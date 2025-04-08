import logging
import time
import torch
from concurrent import futures
import grpc
from inference import model_loader
from transformers import AutoTokenizer
from grpc_comm import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)

class SpeculativeServiceServicer(inference_pb2_grpc.SpeculativeServiceServicer):
    def __init__(self, model_path, sequence_length=128):
        # Load or compile the target model for inference
        self.model = model_loader.load_model(model_path, sequence_length=sequence_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.current_ids = None
        self.max_tokens = 0
        self.tokens_generated = 0
        # Identify EOS token if present
        try:
            self.eos_token_id = self.tokenizer.eos_token_id
        except:
            self.eos_token_id = None

    def StartGeneration(self, request, context):
        """Initialize generation with the given prompt and optional max token limit."""
        prompt_text = request.prompt or ""
        max_tokens = request.max_new_tokens
        logger.info(f"StartGeneration called with prompt: \"{prompt_text}\", max_new_tokens: {max_tokens}")
        # Encode prompt into input IDs and reset generation state
        self.current_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
        self.max_tokens = max_tokens
        self.tokens_generated = 0
        return inference_pb2.StartResponse(acknowledged=True)

    def VerifyDraftChunk(self, request, context):
        """Verify a chunk of draft-predicted tokens against the target model's next tokens."""
        draft_tokens = list(request.draft_tokens)
        logger.info(f"VerifyDraftChunk called with draft_tokens (IDs): {draft_tokens}")

        if self.current_ids is None:
            # No active generation session
            logger.error("No active generation context. Call StartGeneration first.")
            return inference_pb2.VerifyChunkResponse(all_matched=False, match_count=0, correct_token=0, finished=True)

        num_tokens = len(draft_tokens)
        current_len = self.current_ids.shape[1]
        try:
            # Generate the next `num_tokens` tokens with the target model from the current context
            output = self.model.sample(self.current_ids, sequence_length=current_len + num_tokens)
        except Exception as e:
            logger.error(f"Target model generation failed: {e}")
            return inference_pb2.VerifyChunkResponse(all_matched=False, match_count=0, correct_token=0, finished=True)

        # Extract the newly generated tokens from the target model
        target_seq = output[0] if isinstance(output, (list, tuple)) else output[0]
        target_new_ids = target_seq[current_len:]
        target_new_ids = [int(t) for t in target_new_ids]
        logger.info(f"Target model predicted tokens (IDs): {target_new_ids}")

        # Initialize comparison results
        match_count = 0
        all_matched = True
        correct_token_id = 0
        finished = False

        # If EOS was generated by target, mark finished and truncate target_new_ids at EOS
        if self.eos_token_id is not None and self.eos_token_id in target_new_ids:
            eos_index = target_new_ids.index(self.eos_token_id)
            target_new_ids = target_new_ids[:eos_index + 1]
            finished = True  # target hit EOS
            logger.info(f"Target model generated EOS at position {eos_index} in the chunk.")

        # Compare tokens one by one
        for i in range(min(len(draft_tokens), len(target_new_ids))):
            if draft_tokens[i] == target_new_ids[i]:
                match_count += 1
            else:
                all_matched = False
                correct_token_id = target_new_ids[i]
                break
        else:
            # If we exited normally (no break), and lengths differ:
            if len(draft_tokens) != len(target_new_ids):
                # The draft provided fewer tokens than target output or vice versa
                all_matched = False
                # If target output is longer and no mismatch yet, take next token as mismatch
                if len(target_new_ids) > len(draft_tokens):
                    correct_token_id = target_new_ids[len(draft_tokens)]
            # If lengths are equal and no mismatch, all_matched remains True (correct_token_id stays 0)

        # Update target context with matched part (plus correct token if mismatch occurred)
        accepted_ids = draft_tokens[:match_count]
        if not all_matched and correct_token_id != 0:
            accepted_ids.append(correct_token_id)
        if accepted_ids:
            new_tokens_tensor = torch.tensor([accepted_ids], dtype=self.current_ids.dtype)
            self.current_ids = torch.cat([self.current_ids, new_tokens_tensor], dim=1)
            self.tokens_generated += len(accepted_ids)
        if all_matched and len(draft_tokens) == len(target_new_ids):
            # Accept all draft tokens (they matched completely)
            self.tokens_generated += len(draft_tokens)

        logger.info(f"VerifyDraftChunk result: all_matched={all_matched}, match_count={match_count}, correct_token_id={correct_token_id}, finished={finished}")
        return inference_pb2.VerifyChunkResponse(all_matched=all_matched, match_count=match_count, correct_token=correct_token_id, finished=finished)

    def VerifyDraftToken(self, request, context):
        """Legacy single-token verification: delegate to VerifyDraftChunk for one token."""
        draft_token_id = request.draft_token_id
        chunk_req = inference_pb2.VerifyChunkRequest(draft_tokens=[draft_token_id])
        # Reuse the chunk verification logic for single token
        chunk_resp = self.VerifyDraftChunk(chunk_req, context)
        # Convert chunk response to legacy VerifyResponse format
        match = chunk_resp.all_matched  # True if token matched
        correct_token_id = chunk_resp.correct_token  # target token if mismatch
        return inference_pb2.VerifyResponse(match=match, correct_token_id=correct_token_id)

    def GenerateFull(self, request, context):
        """Generate a full continuation for the given prompt using the target model (one-shot)."""
        prompt_text = request.prompt or ""
        max_tokens = request.max_new_tokens
        logger.info(f"GenerateFull called with prompt: \"{prompt_text}\", max_new_tokens: {max_tokens}")
        # Reset and encode prompt
        self.current_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
        self.tokens_generated = 0
        output = self.model.sample(self.current_ids, sequence_length=self.current_ids.shape[1] + max_tokens)
        generated_ids = output[0][self.current_ids.shape[1]:] if isinstance(output, (list, tuple)) else output[0][self.current_ids.shape[1]:]
        generated_ids = [int(t) for t in generated_ids]
        logger.info(f"Target model one-shot generated tokens: {generated_ids}")
        # Return only a dummy token_id (not used in this implementation)
        return inference_pb2.GenerateResponse(token_id=(generated_ids[0] if generated_ids else 0))
    
def run_server(model_path, port=50051, sequence_length=128, profile=False):
    """Launch the gRPC server hosting the target model for speculative decoding."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = SpeculativeServiceServicer(model_path, sequence_length=sequence_length)
    inference_pb2_grpc.add_SpeculativeServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"Target server starting on port {port} (sequence_length={sequence_length})")
    server.start()
    server.wait_for_termination()

# (Optional) A local run function for target-only generation, used by main.py for profiling single-model performance
def run_local(model_path, prompt="", max_new_tokens=50, sequence_length=128, profile=False):
    """Run the target model locally (without gRPC) to generate text for a prompt."""
    logger.info("Running target model locally for output verification/profiling.")
    model = model_loader.load_model(model_path, sequence_length=sequence_length)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids if prompt else torch.zeros((1,0), dtype=torch.long)
    output_text = ""
    tokens_generated = 0
    start_time = time.time() if profile else None
    for i in range(max_new_tokens):
        try:
            output = model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        except Exception as e:
            logger.error(f"Target model generation failed: {e}")
            break
        token_id = int(output[0, -1]) if not isinstance(output, (list, tuple)) else int(output[0][-1])
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=True)
        print(f"Token {i+1}: {repr(token_text)}", flush=True)
        output_text += token_text
        # Append new token to input_ids for next iteration
        new_token_tensor = torch.tensor([[token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
        tokens_generated += 1
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            logger.info("EOS token encountered, stopping generation.")
            break
    # Profiling logs
    if profile and start_time is not None:
        total_time = time.time() - start_time
        throughput = tokens_generated / total_time if total_time > 0 else float('inf')
        logger.info(f"Target model generation completed in {total_time:.2f} seconds.")
        logger.info(f"Tokens generated: {tokens_generated}, Throughput: {throughput:.2f} tokens/sec")
        # Save performance metrics to CSV/JSON
        csv_file = f"performance_target_only_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        json_file = csv_file.replace(".csv", ".json")
        try:
            with open(csv_file, "w") as cf:
                cf.write("total_latency,tokens_generated,throughput,avg_token_time,token_match_rate\n")
                avg_time = (total_time / tokens_generated) if tokens_generated > 0 else 0.0
                cf.write(f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_time:.6f},N/A\n")
            metrics = {
                "total_latency": total_time,
                "tokens_generated": tokens_generated,
                "throughput": throughput,
                "token_match_rate": None
            }
            with open(json_file, "w") as jf:
                import json
                json.dump(metrics, jf, indent=2)
            logger.info(f"Performance metrics saved to {csv_file} and {json_file}")
        except Exception as e:
            logger.error(f"Failed to write performance metrics: {e}")
    print("\n=== Final Output ===\n" + (prompt + output_text))
    return output_text
