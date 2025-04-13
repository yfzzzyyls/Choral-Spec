import grpc
import time
import logging
from speculative_pb2 import LoadModelRequest, StartSessionRequest, GenerateDraftRequest, CheckTokenRequest, GenerateTargetRequest, AppendTokenRequest, UpdateDraftContextRequest
from speculative_pb2_grpc import DraftServiceStub, TargetServiceStub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpeculativeDecoder")

class SpeculativeDecoder:
    def __init__(self, draft_address, target_address, draft_length=4):
        # Connect to gRPC services
        self.draft_channel = grpc.insecure_channel(draft_address)
        self.target_channel = grpc.insecure_channel(target_address)
        self.draft_stub = DraftServiceStub(self.draft_channel)
        self.target_stub = TargetServiceStub(self.target_channel)
        self.draft_length = draft_length
        self.initialized = False

    def load_models(self, draft_model_path, target_model_path, max_length=1024, draft_tp=1, target_tp=1):
        """Load draft and target models on their respective workers."""
        # Load draft model
        logger.info(f"Requesting draft model load: {draft_model_path}")
        resp1 = self.draft_stub.LoadModel(LoadModelRequest(
            model_path=draft_model_path,
            n_positions=max_length,
            batch_size=1,
            tp_degree=draft_tp,
            amp="bf16"
        ))
        if not resp1.success:
            raise RuntimeError(f"Draft model load failed: {resp1.message}")
        # Load target model
        logger.info(f"Requesting target model load: {target_model_path}")
        resp2 = self.target_stub.LoadModel(LoadModelRequest(
            model_path=target_model_path,
            n_positions=max_length,
            batch_size=1,
            tp_degree=target_tp,
            amp="bf16"
        ))
        if not resp2.success:
            raise RuntimeError(f"Target model load failed: {resp2.message}")
        logger.info("Models loaded successfully on draft and target workers.")
        self.initialized = True

    def generate(self, prompts, max_new_tokens=50, stop_on_eos=True, eos_token_id=None):
        """Perform speculative decoding for a batch of prompts."""
        if not self.initialized:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        num_sequences = len(prompts)
        # Initialize sessions for all prompts
        session_pairs = []  # list of (draft_session, target_session)
        for i, prompt_ids in enumerate(prompts):
            draft_session_id = f"draft-{i+1}"
            target_session_id = f"target-{i+1}"
            # Start session on draft worker
            dr = self.draft_stub.StartSession(StartSessionRequest(session_id=draft_session_id, input_ids=prompt_ids))
            # Start session on target worker
            tr = self.target_stub.StartSession(StartSessionRequest(session_id=target_session_id, input_ids=prompt_ids))
            if not dr.success or not tr.success:
                raise RuntimeError(f"Failed to start session for prompt {i}: draft={dr.message}, target={tr.message}")
            session_pairs.append((draft_session_id, target_session_id))
        # Prepare output token sequences (including initial prompt)
        outputs = [list(prompt_ids) for prompt_ids in prompts]
        finished = [False] * num_sequences
        total_tokens_generated = [0] * num_sequences

        start_time = time.time()
        iteration = 0
        # Main decoding loop
        while True:
            iteration += 1
            logger.info(f"=== Speculative Iteration {iteration} ===")
            active_indices = [idx for idx, fin in enumerate(finished) if not fin]
            if not active_indices:
                break  # no active sequences
            # Request draft model to generate speculative tokens for all active sequences
            active_draft_sessions = [session_pairs[idx][0] for idx in active_indices]
            draft_resp = self.draft_stub.GenerateDraft(GenerateDraftRequest(session_ids=active_draft_sessions,
                                                                           draft_length=self.draft_length))
            # Map session_id to draft outputs for easy lookup
            draft_outputs = {out.session_id: (list(out.tokens), list(out.probabilities)) for out in draft_resp.outputs}
            # For each active sequence, perform acceptance sampling
            for idx in active_indices:
                draft_sid, target_sid = session_pairs[idx]
                if draft_sid not in draft_outputs:
                    # No output (possibly error or already finished)
                    continue
                draft_tokens, draft_probs = draft_outputs[draft_sid]
                accepted_count = 0
                reject_index = None
                # Go through each speculative token
                for j, token in enumerate(draft_tokens, start=1):
                    # Check acceptance for token j
                    p_req = CheckTokenRequest(session_id=target_sid, token_id=token, draft_prob=draft_probs[j-1])
                    p_resp = self.target_stub.CheckTokenProbability(p_req)
                    target_prob = p_resp.target_prob
                    # Draw a uniform random number for acceptance
                    r = torch.rand(1).item()
                    if r > (target_prob / (draft_probs[j-1] if draft_probs[j-1] > 0 else 1e-8)):
                        # Reject this token
                        reject_index = j
                        accepted_count = j - 1
                        logger.info(f"Sequence {idx}: Rejected draft token at position {j} (token id {token}).")
                        break
                    else:
                        # Accept this token
                        append_resp = self.target_stub.AppendToken(AppendTokenRequest(session_id=target_sid, token_id=token))
                        accepted_count = j
                        if not append_resp.success:
                            logger.error(f"Failed to append token {token} to target session {target_sid}")
                            reject_index = j
                            break
                # Determine if break occurred
                if reject_index is None:
                    # All draft tokens accepted
                    logger.info(f"Sequence {idx}: All {accepted_count} draft tokens accepted.")
                # Now generate the next token from the target model (with distribution adjustment if needed)
                if reject_index is not None:
                    # Draft token at reject_index was not accepted: use distribution adjustment for next token
                    # Prepare draft distribution at the break point (for token at reject_index)
                    # The draft distribution at this point corresponds to small model's probabilities for token j
                    # We have draft_probs for each token, but not the full vector. We need the full q distribution vector.
                    # To obtain it, we can roll back the draft model to state after accepted_count and get its last_logits
                    # from before it predicted the rejected token.
                    # Simpler approach: since draft worker stored state_cache_stack, we can rely on UpdateDraftContext to provide that state's logits.
                    pass```python
    def generate(self, prompts, max_new_tokens=50, stop_on_eos=True, eos_token_id=None):
        # ... (initialization as above) ...
        while True:
            iteration += 1
            logger.info(f"=== Speculative Iteration {iteration} ===")
            active_indices = [idx for idx, fin in enumerate(finished) if not fin]
            if not active_indices:
                break  # all sequences finished
            # Generate speculative draft tokens for all active sequences
            active_draft_sessions = [session_pairs[idx][0] for idx in active_indices]
            draft_resp = self.draft_stub.GenerateDraft(GenerateDraftRequest(session_ids=active_draft_sessions,
                                                                           draft_length=self.draft_length))
            draft_outputs = {out.session_id: (list(out.tokens), list(out.probabilities)) for out in draft_resp.outputs}
            # Process each active sequence
            for idx in active_indices:
                if finished[idx]:
                    continue
                draft_sid, target_sid = session_pairs[idx]
                if draft_sid not in draft_outputs:
                    continue  # skip if no output (error or finished)
                draft_tokens, draft_probs = draft_outputs[draft_sid]
                accepted_count = 0
                reject_index = None
                # Acceptance sampling loop for this sequence
                for j, (token, q_prob) in enumerate(zip(draft_tokens, draft_probs), start=1):
                    # Query target model probability for this token
                    check_req = CheckTokenRequest(session_id=target_sid, token_id=token, draft_prob=q_prob)
                    check_resp = self.target_stub.CheckTokenProbability(check_req)
                    p = check_resp.target_prob  # target model probability of the draft token
                    # Acceptance decision
                    r = torch.rand(1).item()
                    if r > (p / (q_prob + 1e-8)):
                        # Reject draft token j
                        reject_index = j
                        accepted_count = j - 1
                        logger.info(f"Sequence {idx}: Rejecting draft token #{j} (ID={token}) (p={p:.2e}, q={q_prob:.2e}, r={r:.2f})")
                        break
                    else:
                        # Accept draft token j
                        append_resp = self.target_stub.AppendToken(AppendTokenRequest(session_id=target_sid, token_id=token))
                        if not append_resp.success:
                            logger.error(f"Failed to append token {token} to target session {target_sid}")
                        outputs[idx].append(token)
                        total_tokens_generated[idx] += 1
                        accepted_count = j
                        # If this token was an EOS and we stop on EOS, mark sequence finished
                        if stop_on_eos and eos_token_id is not None and token == eos_token_id:
                            finished[idx] = True
                            break
                # Determine if speculative draft was fully accepted or broke early
                if reject_index is None:
                    # All draft tokens accepted
                    logger.info(f"Sequence {idx}: Accepted all {accepted_count} draft tokens.")
                if finished[idx]:
                    # If finished during acceptance (EOS encountered), skip finalization for this sequence
                    continue
                # Finalization: generate one token from target model
                if reject_index is not None:
                    # Draft token at reject_index was not accepted -> use adjusted distribution
                    # Obtain full draft distribution at the break state from the draft worker
                    accepted_sid = draft_sid
                    # Rollback draft worker state to 'accepted_count' tokens, then get distribution
                    self.draft_stub.UpdateDraftContext(UpdateDraftContextRequest(
                        session_id=accepted_sid, accepted_count=accepted_count, new_token=0  # new_token=0 as a placeholder to rollback only
                    ))
                    # (We assume UpdateDraftContext when new_token=0 will rollback state without adding a token.
                    # Alternatively, a dedicated RPC could provide the distribution.)
                    # Now get the draft model distribution at this rollback state
                    # (We would have a method like GetDraftDistribution to retrieve last_logits probabilities.)
                    draft_dist = []
                    try:
                        dist_resp = self.draft_stub.GetDraftDistribution(GetDraftDistributionRequest(session_id=accepted_sid))
                        draft_dist = list(dist_resp.probabilities)
                    except Exception as e:
                        logger.error(f"Failed to get draft distribution for session {accepted_sid}: {e}")
                    target_req = GenerateTargetRequest(session_id=target_sid, draft_distribution=draft_dist)
                else:
                    # No rejection, use target's own distribution
                    target_req = GenerateTargetRequest(session_id=target_sid)
                target_resp = self.target_stub.GenerateTargetToken(target_req)
                target_token = target_resp.token_id
                outputs[idx].append(target_token)
                total_tokens_generated[idx] += 1
                logger.info(f"Sequence {idx}: Target model generated token ID {target_token} after speculative decoding.")
                # Update the draft model with rollback and the new target token
                update_resp = self.draft_stub.UpdateDraftContext(UpdateDraftContextRequest(
                                    session_id=draft_sid, accepted_count=accepted_count, new_token=target_token))
                if not update_resp.success:
                    logger.error(f"Draft context update failed for session {draft_sid}: {update_resp.message}")
                # Check termination conditions for this sequence
                if (stop_on_eos and eos_token_id is not None and target_token == eos_token_id) or (total_tokens_generated[idx] >= max_new_tokens):
                    finished[idx] = True
            # End of iteration
            if all(finished[token_idx] for token_idx in range(num_sequences) if total_tokens_generated[token_idx] >= max_new_tokens):
                # Stop if all sequences have either finished or hit max tokens
                if all(finished):
                    break
        elapsed = time.time() - start_time
        logger.info(f"Speculative decoding completed in {elapsed:.2f} seconds for {num_sequences} sequence(s).")
        # Optionally, log throughput
        total_generated = sum(total_tokens_generated)
        logger.info(f"Total new tokens generated: {total_generated}, throughput: {total_generated/elapsed:.2f} tokens/sec")
        return outputs