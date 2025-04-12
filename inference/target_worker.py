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
    We store up to batch_size rows. Each row_i corresponds to one prompt.
    """
    def __init__(self, batch_size=2, initial_seq_len=0, pad_token_id=0):
        # Start with shape [batch_size, initial_seq_len]
        self.batch_size = batch_size
        self.seq_len = initial_seq_len
        self.pad_id = pad_token_id
        if initial_seq_len == 0:
            # shape [batch_size, 0], we'll expand columns as needed
            self.current_ids = torch.zeros((batch_size, 0), dtype=torch.long)
        else:
            self.current_ids = torch.full((batch_size, initial_seq_len), pad_token_id, dtype=torch.long)
        self.row_lengths = [0]*batch_size  # how many tokens each row uses
        self.finished_mask = [False]*batch_size

    def is_all_finished(self):
        return all(self.finished_mask)

    def get_active_indices(self):
        return [i for i, fin in enumerate(self.finished_mask) if not fin]

    def mark_finished(self, row_idx):
        self.finished_mask[row_idx] = True

    def expand_columns_if_needed(self, new_cols):
        """
        Expand self.current_ids to have new_cols columns if new_cols > self.seq_len.
        """
        if new_cols <= self.seq_len:
            return
        old_cols = self.seq_len
        new_tensor = torch.full(
            (self.batch_size, new_cols), self.pad_id, dtype=self.current_ids.dtype, device=self.current_ids.device
        )
        if old_cols>0:
            new_tensor[:, :old_cols] = self.current_ids
        self.current_ids = new_tensor
        self.seq_len = new_cols

    def add_tokens_to_row(self, row_idx, tokens: list):
        """
        Appends these tokens to row_idx, expanding columns if needed.
        """
        old_len = self.row_lengths[row_idx]
        needed = old_len + len(tokens)
        if needed > self.seq_len:
            self.expand_columns_if_needed(needed)
        # write them
        for i, t in enumerate(tokens):
            self.current_ids[row_idx, old_len + i] = t
        self.row_lengths[row_idx]+= len(tokens)


class SpeculativeServiceServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_path, sequence_length=128, batch_size=2):
        self.model = model_loader.load_model(model_path, sequence_length=sequence_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        self.sessions = {}  # session_id -> BatchedTargetSession
        self.lock = torch.multiprocessing.Lock()
        self.batch_size = batch_size
        logger.info(f"Target worker compiled with batch_size={self.batch_size}.")

    def _assign_next_row(self, sess: BatchedTargetSession):
        """
        Return index of a free row (not finished, length=0) or a row that is not used yet.
        If no free row, return None.
        """
        for i in range(sess.batch_size):
            if sess.row_lengths[i] == 0 and not sess.finished_mask[i]:
                return i
        return None

    def _find_unfinished_row(self, sess: BatchedTargetSession):
        """Return the first row that is not finished (the code uses 1 row per finalize)."""
        for i in range(sess.batch_size):
            if not sess.finished_mask[i]:
                return i
        return None

    def StartGeneration(self, request, context):
        session_id = request.session_id
        prompt_text = request.prompt
        max_tokens = request.max_new_tokens
        gamma = request.gamma
        logger.info(f"[session={session_id}] StartGeneration: prompt='{prompt_text}', max_new_tokens={max_tokens}, gamma={gamma}")
        with self.lock:
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, re-initializing.")
            # Create an empty [batch_size,0] session
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            sess = BatchedTargetSession(batch_size=self.batch_size, initial_seq_len=0, pad_token_id=pad_id)
            self.sessions[session_id] = sess
        return inference_pb2.StartResponse(acknowledged=True)

    def VerifyBatchTokens(self, request, context):
        """
        True single-pass partial acceptance. We do exactly one forward pass for the chunk of gamma tokens
        each row proposes, shape [num_rows, old_len + chunk_len].
        We re-encode from scratch here (no caching), but each chunk is only one pass.
        If accepted_count=0 and mismatch=0, we forcibly mismatch the first token => ensures progress.
        """
        import torch
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
                    # session not found => finish
                    for tok_list in token_lists:
                        results.append(inference_pb2.VerifyResult(
                            session_id=sid,
                            tokens_accepted=0,
                            target_token=0,
                            finished=True
                        ))
                    continue
                sess = self.sessions[sid]
                device_ = next(self.model.parameters(), None)
                if device_ is not None:
                    device_ = device_.device
                else:
                    device_ = torch.device('cpu')

                # For demonstration, we handle each row in token_lists sequentially. If you want multi-row,
                # unify them in one pass. We'll do a single row approach for clarity.
                for d_toks in token_lists:
                    # find a row
                    r_idx = self._find_unfinished_row(sess)
                    if r_idx is None:
                        results.append(inference_pb2.VerifyResult(
                            session_id=sid, tokens_accepted=0, target_token=0, finished=True
                        ))
                        continue

                    if not d_toks:
                        # no tokens => skip
                        results.append(inference_pb2.VerifyResult(
                            session_id=sid,
                            tokens_accepted=0,
                            target_token=0,
                            finished=sess.finished_mask[r_idx]
                        ))
                        continue

                    old_len = sess.row_lengths[r_idx]
                    old_ids = sess.current_ids[r_idx, :old_len]
                    chunk_len = len(d_toks)
                    # build new context
                    new_row = torch.cat([old_ids, torch.tensor(d_toks, dtype=old_ids.dtype)], dim=0)
                    new_len = new_row.size(0)

                    # single pass => shape [1, new_len, vocab]
                    with torch.no_grad():
                        out = self.model(new_row.unsqueeze(0).to(device_))
                    if hasattr(out, 'logits'):
                        logits_3d = out.logits
                    else:
                        logits_3d = out
                    if logits_3d.dim()==2:
                        logits_3d = logits_3d.unsqueeze(1)  # => [1, new_len, vocab]

                    # partial acceptance
                    accepted_count=0
                    mismatch_tok=0
                    fin=False
                    row_logits_slice = logits_3d[0, (new_len - chunk_len):, :] # the last chunk_len positions
                    for i_token in range(chunk_len):
                        dtok = d_toks[i_token]
                        row_logits = row_logits_slice[i_token, :]
                        t_id = int(torch.argmax(row_logits).item())
                        if t_id == dtok:
                            accepted_count+=1
                            if t_id == self.eos_token_id:
                                fin=True
                                sess.finished_mask[r_idx]=True
                                break
                        else:
                            mismatch_tok = t_id
                            if mismatch_tok==self.eos_token_id:
                                fin=True
                                sess.finished_mask[r_idx]=True
                            break
                    # failsafe mismatch if accepted=0 & mismatch=0 => force mismatch
                    if accepted_count==0 and mismatch_tok==0 and chunk_len>0:
                        mismatch_tok = (d_toks[0]+1) % row_logits_slice.shape[1]

                    # finalize in session => store accepted tokens + forced mismatch
                    accepted_tokens = d_toks[:accepted_count]
                    sess.add_tokens_to_row(r_idx, accepted_tokens)
                    if mismatch_tok!=0:
                        sess.add_tokens_to_row(r_idx, [mismatch_tok])
                        if mismatch_tok == self.eos_token_id:
                            fin=True
                            sess.finished_mask[r_idx]=True

                    res = inference_pb2.VerifyResult(
                        session_id=sid,
                        tokens_accepted=accepted_count,
                        target_token=mismatch_tok,
                        finished=fin
                    )
                    results.append(res)

        return inference_pb2.VerifyBatchResponse(results=results)



    def FinalizeBatchTokens(self, request, context):
        """
        For each sequence, store the given tokens into the next available row or the first unfinished row.
        If we see EOS, we mark that row as finished.
        """
        results = []
        with self.lock:
            sid_to_tokens = {}
            for seq in request.sequences:
                sid = seq.session_id
                if sid not in sid_to_tokens:
                    sid_to_tokens[sid] = []
                sid_to_tokens[sid].append(seq.tokens)

            for sid, list_of_token_lists in sid_to_tokens.items():
                if sid not in self.sessions:
                    logger.warning(f"Session {sid} not found in finalize batch.")
                    for tok_list in list_of_token_lists:
                        results.append(inference_pb2.FinalizeBatchResult(session_id=sid, finished=True))
                    continue
                sess = self.sessions[sid]
                for tok_list in list_of_token_lists:
                    row_idx = self._find_unfinished_row(sess)
                    if row_idx is None:
                        # all done
                        results.append(inference_pb2.FinalizeBatchResult(session_id=sid, finished=True))
                        continue
                    # store tokens
                    sess.add_tokens_to_row(row_idx, tok_list)
                    # check if eos in tok_list
                    if self.eos_token_id in tok_list:
                        sess.mark_finished(row_idx)
                    fin = sess.finished_mask[row_idx]
                    results.append(inference_pb2.FinalizeBatchResult(session_id=sid, finished=fin))

        return inference_pb2.FinalizeBatchResponse(results=results)

    # Single-sequence calls (legacy)
    def VerifyDraftTokens(self, request, context):
        sid = request.session_id
        draft_tokens = list(request.draft_tokens)
        logger.info(f"[session={sid}] VerifyDraftTokens: {draft_tokens}")
        # fallback single-sequence
        return inference_pb2.VerifyResponse(target_probs=[1.0]*len(draft_tokens), finished=False)

    def FinalizeTokens(self, request, context):
        sid = request.session_id
        accepted_count = request.accepted_count
        draft_chunk_size = request.draft_chunk_size
        logger.info(f"[session={sid}] FinalizeTokens: accepted_count={accepted_count}, chunk_size={draft_chunk_size}")
        # fallback single-sequence
        return inference_pb2.FinalizeResponse(final_token=0, finished=False)

def run_server(model_path, port=50051, sequence_length=128, profile=False):
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Loading target model from {model_path} seq_len={sequence_length}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    servicer = SpeculativeServiceServicer(model_path, sequence_length=sequence_length, batch_size=2)
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server_address = f"[::]:{port}"
    logger.info(f"Target server starting on {server_address}")
    server.add_insecure_port(server_address)
    server.start()
    server.wait_for_termination()