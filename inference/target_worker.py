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
        We unify all sequences in request.sequences for the same session into a single sub-batch forward pass.
        Each row => one sequence. If multiple, we handle them in parallel. We do partial acceptance.
        Includes fix for 'too many indices for tensor of dimension 2' by reshaping 2D -> 3D.
        """
        logger.debug(f"VerifyBatchTokens called with {len(request.sequences)} sequences")
        results = []
        with self.lock:
            seq_map = {}
            for draft_seq in request.sequences:
                sid = draft_seq.session_id
                if sid not in seq_map:
                    seq_map[sid] = []
                seq_map[sid].append(draft_seq.draft_tokens)

            # For each session => unify multiple token arrays
            for sid, token_lists in seq_map.items():
                if sid not in self.sessions:
                    # session not found => all finished
                    for tok_list in token_lists:
                        results.append(inference_pb2.VerifyResult(
                            session_id=sid, tokens_accepted=0, target_token=0, finished=True
                        ))
                    continue

                sess = self.sessions[sid]
                # get device from model's first parameter
                device_ = next(self.model.parameters(), None)
                if device_ is not None:
                    device_ = device_.device
                else:
                    device_ = torch.device('cpu')

                # We assign each token_list to the next unfinished row
                active_rows = []
                row_toks = []
                for tok_list in token_lists:
                    r_idx = self._find_unfinished_row(sess)
                    if r_idx is None:
                        # all rows finished => accept none
                        results.append(inference_pb2.VerifyResult(
                            session_id=sid, tokens_accepted=0, target_token=0, finished=True
                        ))
                        continue
                    active_rows.append(r_idx)
                    row_toks.append(tok_list)

                if not active_rows:
                    continue

                max_chunk = max(len(arr) for arr in row_toks) if row_toks else 0
                if max_chunk == 0:
                    # nothing to verify
                    for (r_idx, arr) in zip(active_rows, row_toks):
                        results.append(inference_pb2.VerifyResult(
                            session_id=sid,
                            tokens_accepted=0,
                            target_token=0,
                            finished=sess.finished_mask[r_idx]
                        ))
                    continue

                # Build new context for each row => [old_len + chunk_len]
                new_contexts = []
                chunk_lens = []
                old_lens = []
                for irow, r_idx in enumerate(active_rows):
                    old_len = sess.row_lengths[r_idx]
                    old_lens.append(old_len)
                    old_ids = sess.current_ids[r_idx, :old_len]
                    draft_arr = row_toks[irow]
                    draft_tensor = torch.tensor(draft_arr, dtype=old_ids.dtype)
                    new_row = torch.cat([old_ids, draft_tensor], dim=0)
                    new_contexts.append(new_row)
                    chunk_lens.append(len(draft_arr))

                # Pad to unify => shape [num_active_rows, max_len]
                max_len = max(x.size(0) for x in new_contexts)
                padded_input = []
                for row_data in new_contexts:
                    pad_needed = max_len - row_data.size(0)
                    if pad_needed > 0:
                        pad_tensor = torch.full((pad_needed,), sess.pad_id, dtype=row_data.dtype)
                        row_data = torch.cat([row_data, pad_tensor], dim=0)
                    padded_input.append(row_data)
                batched_ids = torch.stack(padded_input, dim=0).to(device_)  # shape [num_active_rows, max_len]

                # Forward pass => shape [num_active_rows, max_len, vocab] or [num_active_rows, vocab]
                with torch.no_grad():
                    out = self.model(batched_ids)

                if hasattr(out, 'logits'):
                    logits_3d = out.logits
                else:
                    logits_3d = out

                # **Fix**: If it's 2D => [num_active_rows, vocab], reshape to [num_active_rows, max_len, vocab]
                if logits_3d.dim() == 2:
                    # If we only appended chunk_lens[i] tokens, we likely have max_len=someVal
                    # but if we got 2D shape => means max_len=1 or the model is ignoring context
                    # We'll interpret as just a single time dimension
                    # So reshape => [num_active_rows, 1, vocab]
                    logits_3d = logits_3d.unsqueeze(1)

                # partial acceptance
                for irow, r_idx in enumerate(active_rows):
                    draft_arr = row_toks[irow]
                    c_len = chunk_lens[irow]
                    old_len = old_lens[irow]
                    new_len = old_len + c_len
                    # slice out the region => shape [c_len, vocab]
                    # if c_len=1 => shape [1, vocab], no error
                    start_idx = new_len - c_len
                    end_idx = new_len
                    # ensure we don't exceed the time dim
                    time_dim = logits_3d.shape[1]
                    if end_idx > time_dim:
                        end_idx = time_dim
                    row_logits_slice = logits_3d[irow, start_idx:end_idx, :]

                    accepted_count = 0
                    mismatch_tok = 0
                    fin_flag = False

                    for t_i in range(c_len):
                        # safety check
                        if t_i >= row_logits_slice.shape[0]:
                            # no more logits => accept none
                            break
                        dtok = draft_arr[t_i]
                        row_logits = row_logits_slice[t_i, :]
                        t_id = int(torch.argmax(row_logits, dim=-1).item())
                        if t_id == dtok:
                            accepted_count += 1
                            if self.eos_token_id is not None and dtok == self.eos_token_id:
                                fin_flag = True
                                sess.mark_finished(r_idx)
                                break
                        else:
                            mismatch_tok = t_id
                            if self.eos_token_id is not None and t_id == self.eos_token_id:
                                fin_flag = True
                                sess.mark_finished(r_idx)
                            break

                    # store accepted tokens
                    accepted_tokens = draft_arr[:accepted_count]
                    sess.add_tokens_to_row(r_idx, accepted_tokens)
                    if mismatch_tok != 0:
                        sess.add_tokens_to_row(r_idx, [mismatch_tok])
                        if mismatch_tok == self.eos_token_id:
                            fin_flag = True
                            sess.mark_finished(r_idx)

                    vr = inference_pb2.VerifyResult(
                        session_id=sid,
                        tokens_accepted=accepted_count,
                        target_token=mismatch_tok,
                        finished=fin_flag
                    )
                    results.append(vr)

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