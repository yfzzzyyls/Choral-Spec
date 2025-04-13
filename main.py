import argparse
import logging
import sys
import torch
import grpc
import time

from grpc_comm.inference_pb2 import (
    LoadModelRequest,
    StartSessionRequest,
    GenerateDraftRequest,
    CheckTokenRequest,
    GenerateTargetRequest,
    AppendTokenRequest,
    UpdateDraftContextRequest
)
from grpc_comm.inference_pb2_grpc import DraftServiceStub, TargetServiceStub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpeculativeDecoder")

class SpeculativeDecoder:
    def __init__(self, draft_address, target_address, gamma=4):
        """
        Connect to the small (draft) and large (target) model gRPC services.
        `gamma` is how many tokens to generate from the draft in each chunk.
        """
        self.draft_channel = grpc.insecure_channel(draft_address)
        self.target_channel = grpc.insecure_channel(target_address)
        self.draft_stub = DraftServiceStub(self.draft_channel)
        self.target_stub = TargetServiceStub(self.target_channel)
        self.gamma = gamma
        self.initialized = False

    def load_models(self, draft_model_path, target_model_path,
                    max_length=1024, draft_tp=1, target_tp=1):
        """Tell each worker to load/compile its model."""
        # 1) Draft
        logger.info(f"Loading draft model at '{draft_model_path}' (sequence_length={max_length})")
        dresp = self.draft_stub.LoadModel(LoadModelRequest(
            model_path=draft_model_path,
            n_positions=max_length,
            batch_size=1,
            tp_degree=draft_tp,
            amp="bf16"
        ))
        if not dresp.success:
            raise RuntimeError(f"Draft model load failed: {dresp.message}")

        # 2) Target
        logger.info(f"Loading target model at '{target_model_path}' (sequence_length={max_length})")
        tresp = self.target_stub.LoadModel(LoadModelRequest(
            model_path=target_model_path,
            n_positions=max_length,
            batch_size=1,
            tp_degree=target_tp,
            amp="bf16"
        ))
        if not tresp.success:
            raise RuntimeError(f"Target model load failed: {tresp.message}")

        logger.info("Models loaded successfully on both draft and target workers.")
        self.initialized = True

    def generate(
        self,
        prompts,
        max_new_tokens=50,
        eos_token_id=None,
        stop_on_eos=True
    ):
        """
        Example multi-sequence speculative decode. Each prompt is a list of token IDs (ints).
        We'll produce up to `max_new_tokens` for each sequence.

        Approach:
        - Initialize sessions on both draft and target for each sequence
        - Repeatedly call `GenerateDraft(gamma tokens)`, do acceptance check, append accepted tokens to the target,
          and finalize with a target token if mismatch. Then rollback the draft state as needed.
        """

        if not self.initialized:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        num_sequences = len(prompts)

        # 1) Start sessions for each sequence on both draft & target
        session_pairs = []
        for i, input_ids in enumerate(prompts):
            draft_sid = f"draft_{i+1}"
            target_sid = f"target_{i+1}"

            dr = self.draft_stub.StartSession(StartSessionRequest(
                session_id=draft_sid,
                input_ids=input_ids
            ))
            tr = self.target_stub.StartSession(StartSessionRequest(
                session_id=target_sid,
                input_ids=input_ids
            ))
            if not dr.success:
                raise RuntimeError(f"StartSession failed on draft seq#{i}: {dr.message}")
            if not tr.success:
                raise RuntimeError(f"StartSession failed on target seq#{i}: {tr.message}")
            session_pairs.append((draft_sid, target_sid))

        # The final outputs (including prompts)
        outputs = [list(p) for p in prompts]
        finished = [False]*num_sequences
        tokens_generated = [0]*num_sequences

        # main loop
        iteration = 0
        start_time = time.time()

        while True:
            # Are we done?
            active_indices = [idx for idx, fin in enumerate(finished) if not fin]
            if not active_indices:
                logger.info("All sequences are finished.")
                break

            iteration += 1
            logger.info(f"=== Iteration {iteration} => Generating {self.gamma} tokens from draft for all active sequences ===")

            # gather the draft session IDs
            active_draft_sids = [session_pairs[i][0] for i in active_indices]
            # 2) generate up to gamma tokens on the draft side
            dreq = GenerateDraftRequest(
                session_ids=active_draft_sids,
                draft_length=self.gamma
            )
            dresp = self.draft_stub.GenerateDraft(dreq)

            # parse the response
            # each item => session_id, tokens[], probabilities[]
            draft_map = {}
            for out in dresp.outputs:
                draft_map[out.session_id] = (list(out.tokens), list(out.probabilities))

            # 3) acceptance check each sequence
            for i in active_indices:
                draft_sid, target_sid = session_pairs[i]
                if draft_sid not in draft_map:
                    logger.info(f"Draft worker did not return anything for sid={draft_sid}. Possibly an error or already finished.")
                    finished[i] = True
                    continue
                d_tokens, d_probs = draft_map[draft_sid]
                accept_count = 0
                mismatch_index = None
                logger.info(f"Speculative tokens for seq#{i}: {d_tokens}")
                # for each token j in the draft chunk
                for j, dtok in enumerate(d_tokens):
                    # check acceptance
                    # call target CheckTokenProbability
                    c_req = CheckTokenRequest(
                        session_id=target_sid,
                        token_id=dtok,
                        draft_prob=d_probs[j]
                    )
                    c_resp = self.target_stub.CheckTokenProbability(c_req)
                    p = c_resp.target_prob
                    q = d_probs[j] if d_probs[j]>0 else 1e-8
                    ratio = p / q
                    # random acceptance
                    r = float(torch.rand(1).item())
                    if r > ratio:
                        # mismatch
                        mismatch_index = j
                        accept_count = j
                        logger.info(f"[seq#{i}] Mismatch at token j={j}, token_id={dtok}, ratio={ratio:.4f}, r={r:.4f}")
                        break
                    else:
                        # accept token
                        # call AppendToken in target
                        a_resp = self.target_stub.AppendToken(
                            AppendTokenRequest(session_id=target_sid, token_id=dtok)
                        )
                        if not a_resp.success:
                            logger.error(f"AppendToken failed for seq#{i}, token={dtok}")
                            mismatch_index = j
                            break
                        # add dtok to final output
                        outputs[i].append(dtok)
                        tokens_generated[i]+=1
                        accept_count=j+1
                        # check eos
                        if (stop_on_eos and eos_token_id is not None and dtok==eos_token_id) or (tokens_generated[i]>=max_new_tokens):
                            finished[i] = True
                            break
                # if we finished during acceptance check
                if finished[i]:
                    # skip final step for this sequence
                    continue

                # 4) final step => if mismatch
                if mismatch_index is not None:
                    # we accepted accept_count tokens
                    # now we want the target to produce the fallback token from its distribution minus q
                    # first rollback draft to after accept_count tokens
                    rollback_req = UpdateDraftContextRequest(
                        session_id=draft_sid,
                        accepted_count=accept_count,
                        new_token=0
                    )
                    rb_resp = self.draft_stub.UpdateDraftContext(rollback_req)
                    if not rb_resp.success:
                        logger.error(f"Rollback failed on draft seq#{i}: {rb_resp.message}")
                    # then we want target to produce the next token from adjusted distribution
                    # but we haven't actually retrieved the full draft distribution. If we truly
                    # want p-q correction, we need it. We'll skip real p-q correction for brevity,
                    # or you can store the distribution in the draft worker's last_logits. 
                    # We'll just ask the target to sample from its own distribution.
                    t_req = GenerateTargetRequest(session_id=target_sid)
                    t_resp = self.target_stub.GenerateTargetToken(t_req)
                    fallback_token = t_resp.token_id
                    logger.info(f"[seq#{i}] mismatch => fallback token from target is {fallback_token}")
                    # add fallback token to output
                    outputs[i].append(fallback_token)
                    tokens_generated[i]+=1
                    # also update the draft context with that fallback token
                    up_resp = self.draft_stub.UpdateDraftContext(UpdateDraftContextRequest(
                        session_id=draft_sid,
                        accepted_count=accept_count,  # confirm how many tokens we accepted
                        new_token=fallback_token
                    ))
                    if not up_resp.success:
                        logger.error(f"Draft context update after mismatch failed: {up_resp.message}")
                    if (stop_on_eos and eos_token_id is not None and fallback_token==eos_token_id) or (tokens_generated[i]>=max_new_tokens):
                        finished[i] = True
                else:
                    # no mismatch => we accepted the entire chunk ( accept_count == gamma ), 
                    # we can ask the target for the next token if we want
                    # but in standard lucidrains approach, we skip the next token from target if we trust the entire block.
                    # For demonstration, let's do a single token from target to finalize
                    # (some users skip it to minimize calls).
                    if not finished[i]:
                        t_req = GenerateTargetRequest(session_id=target_sid)
                        t_resp = self.target_stub.GenerateTargetToken(t_req)
                        t_token = t_resp.token_id
                        logger.info(f"[seq#{i}] block accepted => next token from target: {t_token}")
                        outputs[i].append(t_token)
                        tokens_generated[i]+=1
                        up_resp = self.draft_stub.UpdateDraftContext(UpdateDraftContextRequest(
                            session_id=draft_sid,
                            accepted_count=accept_count,
                            new_token=t_token
                        ))
                        if not up_resp.success:
                            logger.error(f"Draft context update after block acceptance failed: {up_resp.message}")
                        if (stop_on_eos and eos_token_id is not None and t_token==eos_token_id) or (tokens_generated[i]>=max_new_tokens):
                            finished[i] = True

            # check if all finished or exceed tokens
            all_done = True
            for i in range(num_sequences):
                if not finished[i] and tokens_generated[i]<max_new_tokens:
                    all_done=False
                    break
            if all_done:
                break

        # end loop
        logger.info("Speculative decoding done.")
        total_time = time.time() - start_time
        total_gen = sum(tokens_generated)
        logger.info(f"Generated total {total_gen} new tokens in {total_time:.2f} s => throughput = {total_gen/total_time:.2f} t/s")
        return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft_model", type=str, required=True,
                        help="Path or ID for the draft model on the draft worker.")
    parser.add_argument("--target_model", type=str, required=True,
                        help="Path or ID for the target model on the target worker.")
    parser.add_argument("--draft_server", type=str, default="localhost:50051",
                        help="Draft worker gRPC address.")
    parser.add_argument("--target_server", type=str, default="localhost:50052",
                        help="Target worker gRPC address.")
    parser.add_argument("--gamma", type=int, default=4,
                        help="Number of tokens the draft speculates each iteration.")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum new tokens to generate.")
    parser.add_argument("--prompt_text", type=str, default="",
                        help="Optional single prompt text. If empty, we use a default test prompt.")
    parser.add_argument("--prompt_file", type=str, default="",
                        help="Optional file with prompts (one prompt per line).")
    parser.add_argument("--sequence_length", type=int, default=128,
                        help="Max sequence length for model compilation.")
    args = parser.parse_args()

    # Build the orchestrator
    dec = SpeculativeDecoder(draft_address=args.draft_server,
                             target_address=args.target_server,
                             gamma=args.gamma)

    # 1) Load models on each worker
    dec.load_models(args.draft_model, args.target_model,
                    max_length=args.sequence_length,
                    draft_tp=1,  # or customize
                    target_tp=1) # or customize

    # 2) Gather prompts
    prompts = []
    if args.prompt_file:
        # read each line as a single prompt
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line:
                    # Convert line to token IDs. For demonstration, let's do a naive whitespace split
                    # or you can do your real tokenizer if you want
                    # But here's a dummy approach: convert each char => its ord, for demonstration
                    # In real usage, you should pass actual token IDs from your HF tokenizer
                    tokens = [ord(c) for c in line]
                    prompts.append(tokens)
    elif args.prompt_text.strip():
        # single prompt from command line
        line=args.prompt_text.strip()
        tokens = [ord(c) for c in line]  # dummy approach
        prompts.append(tokens)
    else:
        # fallback: a default test prompt
        default_prompt="Once upon a time, "
        tokens=[ord(c) for c in default_prompt]
        prompts.append(tokens)
        logger.info(f"No prompt_text or prompt_file given; using default test prompt: {default_prompt!r}")

    # 3) Speculative decode
    outputs=dec.generate(prompts, max_new_tokens=args.max_new_tokens,
                         eos_token_id=None, stop_on_eos=False)

    # 4) Print final results
    logger.info("=== FINAL DECODED OUTPUTS ===")
    for i, out_ids in enumerate(outputs):
        # naive decode from our dummy approach (chr)
        text=''.join(chr(t) for t in out_ids)
        logger.info(f"Sequence {i} => {text}")

if __name__=="__main__":
    main()