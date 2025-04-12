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
        True multi-sequence approach:
          - We unify all sequences in request.sequences into a single sub-batch.
          - We do 1 forward pass (or multiple if the tokens differ in length).
          - For each row, we see how many tokens match or diverge.
        """
        results = []
        with self.lock:
            # Step 1: Group sequences by session_id
            # for each session, we have multiple rows
            seq_map = {}
            for draft_seq in request.sequences:
                sid = draft_seq.session_id
                if sid not in seq_map:
                    seq_map[sid] = []
                seq_map[sid].append(draft_seq.draft_tokens)

            # For each session, unify the new tokens into a single 2D shape [num_rows, max_len]
            for sid, token_lists in seq_map.items():
                if sid not in self.sessions:
                    # no session => all finished
                    for tok_list in token_lists:
                        results.append(inference_pb2.VerifyResult(
                            session_id=sid,
                            tokens_accepted=0,
                            target_token=0,
                            finished=True
                        ))
                    continue

                sess = self.sessions[sid]
                if sess.is_all_finished():
                    # session is done => no tokens accepted
                    for tok_list in token_lists:
                        results.append(inference_pb2.VerifyResult(
                            session_id=sid,
                            tokens_accepted=0,
                            target_token=0,
                            finished=True
                        ))
                    continue

                # We have multiple rows in one session if the user arranged them that way
                # but typically the user might pass one row per session. Let's unify them anyway.
                active_rows = sess.get_active_indices()
                if not active_rows:
                    # everything finished
                    for tok_list in token_lists:
                        results.append(inference_pb2.VerifyResult(
                            session_id=sid,
                            tokens_accepted=0,
                            target_token=0,
                            finished=True
                        ))
                    continue

                # Step 2: We must handle each row's new tokens in parallel.
                # But what if the user gave multiple 'draft_tokens' for the same row? We'll assume 1 call = 1 row each.
                # We'll do a single sub-batch forward pass for each new token index, because each row might get a different token
                # We combine them into shape [num_active, 1], do a forward pass => shape [num_active, vocab]
                # then compare each row's draft token to the target's chosen token.

                # For simplicity, assume each token_lists entry is just 1 row -> we do 1 forward pass per token index if needed.
                # We'll do a minimal approach: we only handle the case # all rows have the same length of draft_tokens => single pass
                # but a robust approach would do a step-by-step loop. For demonstration, let's do step-by-step.

                # flatten: e.g. if we have 2 row's tokens => [[5,6,7],[10,11]] => we do a step loop up to max length=3
                # row0 => [5,6,7], row1 => [10,11,_]
                # each step, we feed [row0_token[step], row1_token[step]] if row1 hasn't ended
                # If mismatch => partial acceptance => we do a 'target_token' insertion
                # This is big. We'll do a simpler approach: we only handle 1 token per row for demonstration.

                # => for now, assume each token_lists has length=1 => single pass
                # We'll implement the real multi-step approach for demonstration
                draft_row_tokens = token_lists[0]  # let's handle the first row's tokens only
                # unify => shape [num_active, len(draft_row_tokens)]
                # But we might have a mismatch if # tokens for each row is different => let's do step-by-step.
                max_len = len(draft_row_tokens)
                # We'll do a step loop  (like _verify_sequence_tokens but for the entire sub-batch)
                accepted_count = 0
                mismatch_token = 0
                seq_finished = False

                # build sub-batch for the single step approach
                # feed the entire current_ids => or feed the last token each time => let's do last token each time
                # This is complicated to do truly multi-step. We'll do the single token approach as a demonstration:
                # If your draft side passes 1 token per row, we do 1 pass => done. If multiple tokens, do multiple passes.

                # Let's assume 'draft_row_tokens' is exactly 1 token
                if len(draft_row_tokens) != len(active_rows):
                    logger.warning("Currently, we only handle exactly 1 new token per active row.")
                    # fallback => accept none
                    for row in active_rows:
                        results.append(inference_pb2.VerifyResult(
                            session_id=sid,
                            tokens_accepted=0,
                            target_token=0,
                            finished=sess.finished_mask[row]
                        ))
                    continue

                # shape [num_active]
                token_tensor = torch.tensor(draft_row_tokens, dtype=torch.long)
                # we do a forward pass with shape [num_active, seq_len+1]? We want the next token distribution
                # Actually we need to feed each row's entire context. We'll store them in 'sess.current_ids' => shape [batch_size, ...]
                # We'll build a sub-batch [num_active, ...] from 'sess.current_ids[active_rows]' + the new token
                # Then do one forward pass => shape [num_active, final_seq_len, vocab], we pick the last token's distribution
                # Then compare to draft token?

                # But we want to see which token the target picks => let's do greedy for demonstration
                # We'll do naive approach => we create a new 'temp_ids' for each row => unify in a single batch => forward => shape [num_active, new_seq_len, vocab], pick the last step

                # build list of new contexts
                new_contexts = []
                for irow, row_idx in enumerate(active_rows):
                    row_len = sess.row_lengths[row_idx]
                    # the row's existing context
                    row_ids = sess.current_ids[row_idx, :row_len]
                    # append the draft's token
                    dtok = draft_row_tokens[irow]
                    dtok_tensor = torch.tensor([dtok], dtype=row_ids.dtype)
                    new_row = torch.cat([row_ids, dtok_tensor], dim=0)  # shape [row_len+1]
                    new_contexts.append(new_row)

                # unify into a padded batch
                max_row_len = max(x.size(0) for x in new_contexts)
                padded_input = []
                for row_idx, row_ids in enumerate(new_contexts):
                    pad_sz = max_row_len - row_ids.size(0)
                    if pad_sz > 0:
                        pad_tensor = torch.full((pad_sz,), self.tokenizer.pad_token_id or 0, dtype=row_ids.dtype)
                        cat_row = torch.cat([row_ids, pad_tensor], dim=0)
                        padded_input.append(cat_row)
                    else:
                        padded_input.append(row_ids)
                # shape [num_active, max_row_len]
                batched_ids = torch.stack(padded_input, dim=0)  # 2D
                device_ = next(self.model.parameters()).device
                batched_ids = batched_ids.unsqueeze(1) if len(batched_ids.shape)==1 else batched_ids
                batched_ids = batched_ids.to(device_)

                with torch.no_grad():
                    out = self.model(batched_ids)
                # shape [num_active, max_row_len, vocab]
                # pick the last token's distribution => out[:, -1, :]
                if hasattr(out, 'logits'):
                    logits_all = out.logits
                else:
                    logits_all = out
                # shape [num_active, max_row_len, vocab]
                time_dim = logits_all.shape[1]
                # we want the distribution at the last position => index time_dim-1
                final_logits = logits_all[:, time_dim-1, :]  # shape [num_active, vocab]

                # do greedy
                t_ids = torch.argmax(final_logits, dim=-1)  # shape [num_active]
                t_ids = t_ids.cpu().tolist()

                # now compare each row's dtok vs t_id
                # accepted_count for the entire batch is ambiguous => let's produce partial results
                # but we only produce 1 result object per row => that means multiple results in the final
                for irow, row_idx in enumerate(active_rows):
                    # if we do 1 token => the partial acceptance is either 0 or 1
                    dtok = draft_row_tokens[irow]
                    t_id = t_ids[irow]
                    if t_id == dtok:
                        # accept
                        accepted_cnt = 1
                        mismatch_tok = 0
                        # update the session's context => store dtok
                        # expand row
                        row_len = sess.row_lengths[row_idx]
                        row_ids = sess.current_ids[row_idx, :row_len]
                        appended_tok = torch.tensor([dtok], dtype=row_ids.dtype).unsqueeze(0)
                        appended_tok = appended_tok.to(sess.current_ids.device)
                        # we do cat along dim=1 for shape [1, new_len], but we are storing row in 1D => let's store in 2D
                        # Actually we have shape [batch_size, max_seq_len], we do row_idx => so we do:
                        if row_len < sess.current_ids.size(1):
                            sess.current_ids[row_idx, row_len] = dtok
                            sess.row_lengths[row_idx]+=1
                        else:
                            # expand the columns => for demonstration let's do a new bigger tensor
                            old_cols = sess.current_ids.size(1)
                            new_cols = old_cols+1
                            new_cids = torch.full((sess.batch_size, new_cols), self.tokenizer.pad_token_id or 0, dtype=sess.current_ids.dtype, device=sess.current_ids.device)
                            new_cids[:, :old_cols] = sess.current_ids
                            new_cids[row_idx, old_cols] = dtok
                            sess.current_ids = new_cids
                            sess.seq_len = new_cols
                            sess.row_lengths[row_idx]+=1

                        if self.eos_token_id is not None and dtok == self.eos_token_id:
                            sess.finished_mask[row_idx] = True
                        finished_flag = sess.finished_mask[row_idx]
                    else:
                        # mismatch => accepted_count=0, forced token => t_id
                        accepted_cnt = 0
                        mismatch_tok = t_id
                        # store mismatch in the context
                        row_len = sess.row_lengths[row_idx]
                        if row_len < sess.current_ids.size(1):
                            sess.current_ids[row_idx, row_len] = t_id
                            sess.row_lengths[row_idx]+=1
                        else:
                            # expand columns
                            old_cols = sess.current_ids.size(1)
                            new_cols = old_cols+1
                            new_cids = torch.full((sess.batch_size, new_cols), self.tokenizer.pad_token_id or 0, dtype=sess.current_ids.dtype, device=sess.current_ids.device)
                            new_cids[:, :old_cols] = sess.current_ids
                            new_cids[row_idx, old_cols] = t_id
                            sess.current_ids = new_cids
                            sess.seq_len = new_cols
                            sess.row_lengths[row_idx]+=1

                        if self.eos_token_id is not None and t_id == self.eos_token_id:
                            sess.finished_mask[row_idx] = True
                        finished_flag = sess.finished_mask[row_idx]

                    res = inference_pb2.VerifyResult(
                        session_id=sid,
                        tokens_accepted=accepted_cnt,
                        target_token=mismatch_tok,
                        finished=finished_flag
                    )
                    results.append(res)

        # done
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