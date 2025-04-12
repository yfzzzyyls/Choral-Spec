import logging
import torch
from concurrent import futures
import grpc

from inference import model_loader
from transformers import AutoTokenizer
from grpc_comm import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)

class BatchedTargetSession:
    """
    Holds a [batch_size, seq_len] tensor for multiple sequences in parallel, plus a 'finished' mask.
    """
    def __init__(self, input_ids_2d: torch.Tensor):
        # shape [batch_size, seq_len]
        self.current_ids = input_ids_2d.clone()
        # keep track of how many tokens each row actually has (since seq_len can differ)
        # we'll do that via an array 'row_lengths'
        self.batch_size = input_ids_2d.size(0)
        self.seq_len = input_ids_2d.size(1)
        # infer actual row lengths by searching trailing pad if needed
        # for demo, assume no pad => all rows same length
        self.row_lengths = [self.seq_len]*self.batch_size
        self.finished_mask = [False]*self.batch_size
        self.tokens_generated = [0]*self.batch_size  # optional
        # you can store a 'past_key_values' of shape [batch_size, ...] if the model actually returns it
        self.past_key_values = None

    def is_all_finished(self):
        return all(self.finished_mask)

    def get_active_indices(self):
        return [i for i, fin in enumerate(self.finished_mask) if not fin]

    def mark_finished(self, row_idx):
        self.finished_mask[row_idx] = True


class SpeculativeServiceServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_path, sequence_length=128):
        self.model = model_loader.load_model(model_path, sequence_length=sequence_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.sessions = {}  # session_id -> BatchedTargetSession
        self.lock = torch.multiprocessing.Lock()

    def StartGeneration(self, request, context):
        """
        Expect user to do something like:
          prompt_text = ""  # or some placeholder
          # But if you want multiple sequences, you'd unify them outside
          # For demonstration, we just do a single row. If you want a real batch, pass a bigger shape.
        """
        session_id = request.session_id
        prompt_text = request.prompt
        max_tokens = request.max_new_tokens
        gamma = request.gamma
        logger.info(f"[session={session_id}] StartGeneration: prompt='{prompt_text}', max_new_tokens={max_tokens}, gamma={gamma}")
        with self.lock:
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, overwriting.")
            if prompt_text:
                # naive approach: single row
                enc = self.tokenizer(prompt_text, return_tensors='pt')
                current_ids = enc["input_ids"]  # shape [1, prompt_len]
            else:
                # user can do some other approach
                current_ids = torch.zeros((1,0), dtype=torch.long)
            # store as a BatchedTargetSession with batch_size=1
            sess = BatchedTargetSession(current_ids)
            self.sessions[session_id] = sess
        return inference_pb2.StartResponse(acknowledged=True)

    def VerifyBatchTokens(self, request, context):
        """
        Multi-sequence, multi-token verification in a single forward pass, like lucidrains.
        For each session in request.sequences, we unify them into one sub-batch [num_rows, chunk_len].
        We produce [num_rows, chunk_len, vocab], compare row by row, token by token. 
        If mismatch, partial accept. The difference from your old code is we handle multiple tokens per row in a single pass.
        """
        results = []
        with self.lock:
            seq_map = {}
            for draft_seq in request.sequences:
                sid = draft_seq.session_id
                if sid not in seq_map:
                    seq_map[sid] = []
                seq_map[sid].append(draft_seq.draft_tokens)

            for sid, token_lists in seq_map.items():
                if sid not in self.sessions:
                    # session not found => all finished
                    for tok_list in token_lists:
                        results.append(inference_pb2.VerifyResult(
                            session_id=sid, tokens_accepted=0, target_token=0, finished=True
                        ))
                    continue

                sess = self.sessions[sid]
                if sess.is_all_finished():
                    for tok_list in token_lists:
                        results.append(inference_pb2.VerifyResult(
                            session_id=sid, tokens_accepted=0, target_token=0, finished=True
                        ))
                    continue

                # We unify token_lists => shape [1, chunk_len], because we are storing a single row. 
                # But if you do a true multi-row approach, you'd have row_mask, etc. 
                # For demonstration, let's assume only one row per session. If you want multiple rows, 
                # store them in BatchedTargetSession with batch_size>1. Then you'd unify them similarly.

                # Flatten token_lists => typically one entry. If multiple, unify them
                all_draft_toks = []
                for arr in token_lists:
                    all_draft_toks.extend(arr)
                if not all_draft_toks:
                    results.append(inference_pb2.VerifyResult(
                        session_id=sid, tokens_accepted=0, target_token=0, finished=sess.is_all_finished()
                    ))
                    continue

                # shape [1, len(all_draft_toks)]
                row_len_before = sess.row_lengths[0]  # single row, index=0 
                # build new_context = sess.current_ids[0,:row_len_before] + all_draft_toks
                old_row = sess.current_ids[0, :row_len_before]
                new_chunk = torch.tensor(all_draft_toks, dtype=old_row.dtype)
                new_row = torch.cat([old_row, new_chunk], dim=0)
                new_len = new_row.size(0)

                # unify => shape [1, new_len]
                device_ = next(self.model.parameters(), None)
                if device_ is not None:
                    device_ = device_.device
                else:
                    device_ = torch.device('cpu')
                batch_ids = new_row.unsqueeze(0).to(device_)

                with torch.no_grad():
                    out = self.model(batch_ids)
                # shape [1, new_len, vocab]
                if hasattr(out, 'logits'):
                    logits_3d = out.logits
                else:
                    logits_3d = out
                # We get the logit slice corresponding to the newly appended tokens => indices row_len_before..(new_len-1)
                # shape => [1, chunk_len, vocab]
                chunk_len = len(all_draft_toks)
                logits_slice = logits_3d[:, (new_len - chunk_len):, :]  # [1, chunk_len, vocab]

                # We'll do a token-by-token acceptance check
                # If mismatch => partial accept
                accepted_count = 0
                mismatch_tok = 0
                finished_flag = False

                for i_token in range(chunk_len):
                    dtok = all_draft_toks[i_token]
                    row_logits = logits_slice[0, i_token, :]
                    # do a next token pick => for demonstration we do greedy
                    t_id = int(torch.argmax(row_logits, dim=-1).item())
                    if t_id == dtok:
                        # accept
                        accepted_count += 1
                        # if EOS => finished
                        if self.eos_token_id is not None and dtok == self.eos_token_id:
                            finished_flag = True
                            sess.finished_mask[0] = True
                            break
                    else:
                        # mismatch
                        mismatch_tok = t_id
                        # if t_id is eos => mark finished
                        if self.eos_token_id is not None and t_id == self.eos_token_id:
                            finished_flag = True
                            sess.finished_mask[0] = True
                        break

                # we do not do the multi-row approach here, but you can if you want
                # store partial new_row into sess.current_ids
                # accept tokens_accepted => partial. plus mismatch if any
                new_accepted_part = all_draft_toks[:accepted_count]
                # rebuild the row
                final_len = row_len_before + accepted_count
                # store accepted tokens
                if final_len>sess.current_ids.size(1):
                    # expand
                    old_cols = sess.current_ids.size(1)
                    new_cols = final_len
                    if mismatch_tok != 0:
                        new_cols+=1
                    new_cids = torch.full((1, new_cols), self.tokenizer.pad_token_id or 0, dtype=sess.current_ids.dtype, device=sess.current_ids.device)
                    new_cids[:, :old_cols] = sess.current_ids
                    sess.current_ids = new_cids
                # write the accepted tokens
                for iacc, tval in enumerate(new_accepted_part):
                    sess.current_ids[0, row_len_before + iacc] = tval
                sess.row_lengths[0] = row_len_before + accepted_count

                # forced mismatch
                if mismatch_tok != 0:
                    # store forced mismatch
                    forced_idx = sess.row_lengths[0]
                    if forced_idx>=sess.current_ids.size(1):
                        # expand again
                        old_cols = sess.current_ids.size(1)
                        new_cols = old_cols+1
                        new_cids = torch.full((1, new_cols), self.tokenizer.pad_token_id or 0, dtype=sess.current_ids.dtype, device=sess.current_ids.device)
                        new_cids[:, :old_cols] = sess.current_ids
                        sess.current_ids = new_cids
                    sess.current_ids[0, forced_idx] = mismatch_tok
                    sess.row_lengths[0]+=1
                    if mismatch_tok == self.eos_token_id:
                        finished_flag = True
                        sess.finished_mask[0] = True

                # build result
                rr = inference_pb2.VerifyResult(
                    session_id=sid,
                    tokens_accepted=accepted_count,
                    target_token=mismatch_tok,
                    finished=finished_flag
                )
                results.append(rr)

        return inference_pb2.VerifyBatchResponse(results=results)




    def FinalizeBatchTokens(self, request, context):
        """
        For each row, we finalize the accepted tokens and forced mismatch token, appending them
        to the batch session's context. If we see an EOS, mark that row finished.
        """
        results = []
        with self.lock:
            row_indices = {}
            for seq in request.sequences:
                sid = seq.session_id
                if sid not in self.sessions:
                    logger.warning(f"Session {sid} not found in finalize batch.")
                    results.append(inference_pb2.FinalizeBatchResult(session_id=sid, finished=True))
                    continue
                sess = self.sessions[sid]
                tokens = list(seq.tokens)
                if not tokens:
                    # no tokens => do nothing
                    results.append(inference_pb2.FinalizeBatchResult(session_id=sid, finished=sess.is_all_finished()))
                    continue
                # we do not know which row is which => user might have to pass row index
                # For demonstration, we assume only 1 row in a batch session or 1 call per row
                # If multiple rows => need a row index
                # We'll handle single row approach => find the first non-finished row
                r_idx = None
                for rr in range(sess.batch_size):
                    if not sess.finished_mask[rr]:
                        r_idx = rr
                        break
                if r_idx is None:
                    # all done
                    results.append(inference_pb2.FinalizeBatchResult(session_id=sid, finished=True))
                    continue
                # finalize these tokens
                row_len = sess.row_lengths[r_idx]
                for t in tokens:
                    if row_len >= sess.current_ids.size(1):
                        # expand
                        old_cols = sess.current_ids.size(1)
                        new_cols = old_cols+1
                        new_cids = torch.full((sess.batch_size, new_cols), self.tokenizer.pad_token_id or 0, dtype=sess.current_ids.dtype, device=sess.current_ids.device)
                        new_cids[:, :old_cols] = sess.current_ids
                        sess.current_ids = new_cids
                        sess.seq_len = new_cols
                    sess.current_ids[r_idx, row_len] = t
                    sess.row_lengths[r_idx]+=1
                    row_len+=1
                    if self.eos_token_id is not None and t == self.eos_token_id:
                        sess.finished_mask[r_idx] = True
                results.append(inference_pb2.FinalizeBatchResult(session_id=sid, finished=sess.finished_mask[r_idx]))
        return inference_pb2.FinalizeBatchResponse(results=results)


    ###################################################
    # Single-sequence calls (kept for backward compatibility)
    ###################################################
    def VerifyDraftTokens(self, request, context):
        # unchanged
        sid = request.session_id
        draft_tokens = list(request.draft_tokens)
        logger.info(f"[session={sid}] VerifyDraftTokens: {draft_tokens}")
        with self.lock:
            if sid not in self.sessions:
                return inference_pb2.VerifyResponse(
                    target_probs=[0.0]*len(draft_tokens), finished=True
                )
            sess = self.sessions[sid]
            if sess.is_all_finished():
                return inference_pb2.VerifyResponse(target_probs=[], finished=True)
            if not draft_tokens:
                return inference_pb2.VerifyResponse(target_probs=[], finished=False)

            # fallback single-sequence approach
            target_probs = []
            seq_finished = False
            for t in draft_tokens:
                row_idx = 0  # assume single row
                logits = self._forward_batch(sess, row_idx)
                row_probs = torch.softmax(logits, dim=-1)
                p = float(row_probs[0, t].item())
                target_probs.append(p)
                # we pretend we accepted t
                # etc. This is leftover logic
            return inference_pb2.VerifyResponse(
                target_probs=target_probs,
                finished=seq_finished
            )

    def FinalizeTokens(self, request, context):
        # unchanged single-sequence fallback
        sid = request.session_id
        accepted_count = request.accepted_count
        draft_chunk_size = request.draft_chunk_size
        logger.info(f"[session={sid}] FinalizeTokens: accepted_count={accepted_count}, chunk_size={draft_chunk_size}")
        return inference_pb2.FinalizeResponse(final_token=0, finished=True)

    ###################################################
    # Helper method: forward pass for a single row idx
    ###################################################
    def _forward_batch(self, sess: BatchedTargetSession, row_idx: int):
        # we do a forward pass for shape [1, seq_len_of_that_row]
        # gather that row
        row_len = sess.row_lengths[row_idx]
        row_ids = sess.current_ids[row_idx, :row_len].unsqueeze(0)
        with torch.no_grad():
            out = self.model(row_ids)
        if hasattr(out, 'logits'):
            logits = out.logits
            # shape [1, row_len, vocab]
            return logits[:, -1, :].float()
        else:
            # fallback
            if len(out.shape)==3:
                return out[:, -1, :].float()
            return out.float()


def run_server(model_path, port=50051, sequence_length=128, profile=False):
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Loading target model from {model_path} seq_len={sequence_length}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    servicer = SpeculativeServiceServicer(model_path, sequence_length=sequence_length)
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server_address = f"[::]:{port}"
    logger.info(f"Target server starting on {server_address}")
    server.add_insecure_port(server_address)
    server.start()
    server.wait_for_termination()