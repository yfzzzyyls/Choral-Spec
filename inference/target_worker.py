import logging
import torch
from concurrent import futures
import grpc

from inference import model_loader
from transformers import AutoTokenizer
from grpc_comm import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)

class TargetSession:
    def __init__(self, input_ids):
        # store the current context (tensor of shape [1, seq_len])
        self.current_ids = input_ids
        self.finished = False
        self.tokens_generated = 0
        # store partial chunk from last verify if needed
        self.last_draft_chunk = None

        # optional: store the target model's past_key_values if we want to skip re-encoding.
        self.past_key_values = None

class SpeculativeServiceServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_path, sequence_length=128):
        self.model = model_loader.load_model(model_path, sequence_length=sequence_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.sessions = {}  # session_id -> TargetSession
        self.lock = torch.multiprocessing.Lock()

    def StartGeneration(self, request, context):
        session_id = request.session_id
        prompt_text = request.prompt
        max_tokens = request.max_new_tokens
        gamma = request.gamma
        logger.info(f"[session={session_id}] StartGeneration: prompt='{prompt_text}', max_new_tokens={max_tokens}, gamma={gamma}")
        with self.lock:
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, overwriting.")
            if prompt_text:
                enc = self.tokenizer(prompt_text, return_tensors='pt')
                current_ids = enc["input_ids"]
            else:
                current_ids = torch.zeros((1,0), dtype=torch.long)
            sess = TargetSession(current_ids)
            # optionally, we could precompute the model's past_key_values for the prompt.
            self.sessions[session_id] = sess
        return inference_pb2.StartResponse(acknowledged=True)

    ############################################
    # BATCH calls for multi-seq speculative decode
    ############################################
    def VerifyBatchTokens(self, request, context):
        """
        For each sequence in request.sequences:
          - draft_tokens: the newly proposed tokens from the draft.
          - we compare them to the target model's next tokens.
        We'll respond with tokens_accepted, target_token if mismatch, and finished.
        """
        results = []
        with self.lock:
            for seq in request.sequences:
                sid = seq.session_id
                draft_tokens = list(seq.draft_tokens)

                if sid not in self.sessions:
                    logger.warning(f"Session {sid} not found.")
                    results.append(inference_pb2.VerifyResult(
                        session_id=sid,
                        tokens_accepted=0,
                        target_token=0,
                        finished=True
                    ))
                    continue
                sess = self.sessions[sid]
                if sess.finished:
                    results.append(inference_pb2.VerifyResult(
                        session_id=sid,
                        tokens_accepted=0,
                        target_token=0,
                        finished=True
                    ))
                    continue
                if not draft_tokens:
                    results.append(inference_pb2.VerifyResult(
                        session_id=sid,
                        tokens_accepted=0,
                        target_token=0,
                        finished=False
                    ))
                    continue

                accepted_count, mismatch_token, seq_finished = self._verify_sequence_tokens(sess, draft_tokens)
                results.append(inference_pb2.VerifyResult(
                    session_id=sid,
                    tokens_accepted=accepted_count,
                    target_token=mismatch_token,
                    finished=seq_finished
                ))

        return inference_pb2.VerifyBatchResponse(
            results=results
        )

    def FinalizeBatchTokens(self, request, context):
        """
        We commit the accepted tokens on the target side so it updates its internal context.
        If a mismatch token is included, that also gets appended.
        """
        results = []
        with self.lock:
            for seq in request.sequences:
                sid = seq.session_id
                tokens = list(seq.tokens)
                if sid not in self.sessions:
                    logger.warning(f"Session {sid} not found in finalize batch.")
                    results.append(inference_pb2.FinalizeBatchResult(session_id=sid, finished=True))
                    continue
                sess = self.sessions[sid]
                if sess.finished:
                    results.append(inference_pb2.FinalizeBatchResult(session_id=sid, finished=True))
                    continue

                # incorporate these tokens into sess.current_ids
                for t in tokens:
                    appended_tok = torch.tensor([[t]], dtype=sess.current_ids.dtype)
                    sess.current_ids = torch.cat([sess.current_ids, appended_tok], dim=1)
                    if self.eos_token_id is not None and t == self.eos_token_id:
                        sess.finished = True
                results.append(inference_pb2.FinalizeBatchResult(session_id=sid, finished=sess.finished))
        return inference_pb2.FinalizeBatchResponse(results=results)

    ############################################
    # Single-sequence calls (kept for compatibility)
    ############################################
    def VerifyDraftTokens(self, request, context):
        sid = request.session_id
        draft_tokens = list(request.draft_tokens)
        logger.info(f"[session={sid}] VerifyDraftTokens: {draft_tokens}")
        with self.lock:
            if sid not in self.sessions:
                return inference_pb2.VerifyResponse(
                    target_probs=[0.0]*len(draft_tokens), finished=True
                )
            sess = self.sessions[sid]
            if sess.finished:
                return inference_pb2.VerifyResponse(target_probs=[], finished=True)
            if not draft_tokens:
                return inference_pb2.VerifyResponse(target_probs=[], finished=False)

            # For simplicity, we compute the probability of each token by stepping.
            # But in real usage, we might do a single pass.
            target_probs = []
            seq_finished = False
            # naive approach: for each token, run model once
            for t in draft_tokens:
                if sess.finished:
                    seq_finished = True
                    break
                logits = self._forward_target(sess)
                row_probs = torch.softmax(logits, dim=-1)
                p = float(row_probs[0, t].item())
                target_probs.append(p)
                # we pretend we accepted t for now
                appended_tok = torch.tensor([[t]], dtype=sess.current_ids.dtype)
                sess.current_ids = torch.cat([sess.current_ids, appended_tok], dim=1)
                if self.eos_token_id is not None and t == self.eos_token_id:
                    sess.finished = True
                    seq_finished = True
                    break

            return inference_pb2.VerifyResponse(
                target_probs=target_probs,
                finished=seq_finished
            )

    def FinalizeTokens(self, request, context):
        sid = request.session_id
        accepted_count = request.accepted_count
        draft_chunk_size = request.draft_chunk_size
        logger.info(f"[session={sid}] FinalizeTokens: accepted_count={accepted_count}, chunk_size={draft_chunk_size}")
        with self.lock:
            if sid not in self.sessions:
                return inference_pb2.FinalizeResponse(final_token=0, finished=True)
            sess = self.sessions[sid]
            if sess.finished:
                return inference_pb2.FinalizeResponse(final_token=0, finished=True)
            if sess.last_draft_chunk:
                chunk = sess.last_draft_chunk
                accepted = chunk[:accepted_count]
                for t in accepted:
                    appended_tok = torch.tensor([[t]], dtype=sess.current_ids.dtype)
                    sess.current_ids = torch.cat([sess.current_ids, appended_tok], dim=1)
                    if self.eos_token_id is not None and t == self.eos_token_id:
                        sess.finished = True
                fallback_token = 0
                if accepted_count < draft_chunk_size:
                    fallback_token = self._generate_one_token(sess)
                sess.last_draft_chunk = None
                if fallback_token != 0 and self.eos_token_id is not None and fallback_token == self.eos_token_id:
                    sess.finished = True
                return inference_pb2.FinalizeResponse(final_token=fallback_token, finished=sess.finished)
            else:
                # no chunk stored => nothing accepted => generate one
                fallback_token = self._generate_one_token(sess)
                return inference_pb2.FinalizeResponse(final_token=fallback_token, finished=sess.finished)

    def GenerateFull(self, request, context):
        # optional baseline
        return super().GenerateFull(request, context)

    ###########################
    # Internal helper methods
    ###########################
    def _verify_sequence_tokens(self, sess: TargetSession, draft_tokens: list):
        """
        Step through draft_tokens one by one, let the target model sample.
        Compare to draft. Return (accepted_count, mismatch_token, seq_finished).
        """
        accepted_count = 0
        mismatch_token = 0
        seq_finished = False

        # We'll do a token-by-token loop.
        device = next(self.model.parameters()).device
        for i, dtok in enumerate(draft_tokens):
            if sess.finished:
                seq_finished = True
                break

            # forward target for next token
            logits = self._forward_target(sess)
            row_probs = torch.softmax(logits, dim=-1)

            # sample from row_probs to simulate the target's next token
            # or do greedy. For demonstration, let's do greedy:
            t_id = int(torch.argmax(row_probs, dim=-1)[0].item())
            # compare t_id vs dtok
            if t_id == dtok:
                # accept this token
                accepted_count += 1
                # append dtok to sess.current_ids
                appended_tok = torch.tensor([[dtok]], dtype=sess.current_ids.dtype, device=device)
                sess.current_ids = torch.cat([sess.current_ids, appended_tok.cpu()], dim=1)
                if self.eos_token_id is not None and dtok == self.eos_token_id:
                    sess.finished = True
                    seq_finished = True
                    break
            else:
                # mismatch
                mismatch_token = t_id
                # we do not append dtok to the session. Instead, we append mismatch_token.
                appended_tok = torch.tensor([[t_id]], dtype=sess.current_ids.dtype)
                sess.current_ids = torch.cat([sess.current_ids, appended_tok], dim=1)
                if self.eos_token_id is not None and t_id == self.eos_token_id:
                    sess.finished = True
                    seq_finished = True
                break

        return accepted_count, mismatch_token, sess.finished

    def _forward_target(self, sess: TargetSession):
        """
        Gets next logits from the target model for the next token. Could do caching.
        For demonstration, we re-run from the entire current_ids each time or do partial.
        """
        # naive approach: we do a single step forward by feeding the last token
        # or if it's empty, feed the entire prompt.
        with torch.no_grad():
            out = self.model(sess.current_ids)
        if hasattr(out, 'logits'):
            # shape [1, seq_len, vocab]
            return out.logits[:, -1, :]
        else:
            # fallback
            if len(out.shape) == 3:
                return out[:, -1, :]
            return out

    def _generate_one_token(self, sess: TargetSession):
        logits = self._forward_target(sess)
        t_id = int(torch.argmax(logits, dim=-1)[0].item())
        appended_tok = torch.tensor([[t_id]], dtype=sess.current_ids.dtype)
        sess.current_ids = torch.cat([sess.current_ids, appended_tok], dim=1)
        if self.eos_token_id is not None and t_id == self.eos_token_id:
            sess.finished = True
        sess.tokens_generated += 1
        return t_id


def _extract_logits_all(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits.float()
    out_t = outputs
    if len(out_t.shape) == 3:
        return out_t.float()
    elif len(out_t.shape) == 2:
        return out_t.unsqueeze(1).float()
    elif len(out_t.shape) == 1:
        return out_t.unsqueeze(0).unsqueeze(0).float()
    else:
        raise ValueError(f"Unhandled shape for model output: {out_t.shape}")


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


def run_local(model_path, prompt="", max_new_tokens=50, sequence_length=128, profile=False):
    pass