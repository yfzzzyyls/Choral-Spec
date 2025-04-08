import logging
import grpc
from grpc_comm import inference_pb2_grpc
from inference.model_loader import load_model
from inference.speculative import speculative_decode
from transformers import AutoTokenizer
import torch
import time
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_client(draft_model_name: str, target_host: str = "localhost", port: int = 50051,
               prompt: str = "", target_model_name: str = None, max_new_tokens: int = 50,
               sequence_length: int = 128, profile: bool = False, no_target: bool = False):
    logger.info(f"Loading draft model '{draft_model_name}' (sequence_length={sequence_length})...")
    draft_model = load_model(draft_model_name, sequence_length=sequence_length)
    tokenizer_source = target_model_name or draft_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)

    if not prompt:
        logger.error("No prompt provided for draft client.")
        return

    # STANDALONE DRAFT (no_target=True)
    if no_target:
        logger.info(f"Starting draft-only generation for prompt: {repr(prompt)}")
        start_time = time.time() if profile else None
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        output_text = ""
        tokens_generated = 0

        for i in range(max_new_tokens):
            output = draft_model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
            if isinstance(output, (list, tuple)):
                token_id = int(output[0][-1])
            else:
                token_id = int(output[0, -1])

            # decode with correct spacing
            token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=True)
            print(f"Token {i+1}: {repr(token_text)}", flush=True)
            output_text += token_text
            new_token_tensor = torch.tensor([[token_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
            tokens_generated += 1

        total_time = time.time() - start_time if profile else 0.0
        if profile:
            throughput = tokens_generated / total_time if total_time > 0 else float('inf')
            logger.info(f"Draft-only run done in {total_time:.2f} s, tokens={tokens_generated}, throughput={throughput:.2f} t/s")
            csv_file = f"performance_draft_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            json_file = csv_file.replace('.csv', '.json')
            try:
                with open(csv_file, 'w', newline='') as f:
                    f.write("total_latency,tokens_generated,throughput,avg_token_time,token_match_rate\n")
                    avg_time = (total_time / tokens_generated) if tokens_generated > 0 else 0.0
                    f.write(f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_time:.6f},N/A\n")
                metrics = {
                    "total_latency": total_time,
                    "tokens_generated": tokens_generated,
                    "throughput": throughput,
                    "token_match_rate": None
                }
                with open(json_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Performance metrics saved to {csv_file} and {json_file}")
            except Exception as e:
                logger.error(f"Failed to save profiling data: {e}")

        full_output = prompt + output_text
        print("\n=== Final Output ===\n" + full_output)
        return

    # SPECULATIVE DECODING with target
    logger.info(f"Connecting to target server at {target_host}:{port}...")
    channel = grpc.insecure_channel(f"{target_host}:{port}")
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)
    logger.info(f"Starting speculative decoding for prompt: {prompt!r}")

    profile_data = {} if profile else None
    start_time = time.time() if profile else None

    # do the entire decode
    generated_text = speculative_decode(draft_model, tokenizer, stub, prompt,
                                        max_new_tokens=max_new_tokens, profile_data=profile_data)

    if profile:
        total_time = time.time() - start_time
        tokens_generated = profile_data.get('tokens_generated', 0)
        match_count = profile_data.get('match_count', 0)
        if tokens_generated > 1:
            match_rate = match_count / (tokens_generated - 1)
        else:
            match_rate = None
        throughput = tokens_generated / total_time if total_time > 0 else float('inf')
        logger.info(f"Speculative decoding completed in {total_time:.2f}s, tokens={tokens_generated}, match_rate={match_rate}, throughput={throughput:.2f} t/s")

        csv_file = f"performance_speculative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        json_file = csv_file.replace('.csv', '.json')
        try:
            with open(csv_file, 'w', newline='') as f:
                f.write("total_latency,tokens_generated,throughput,avg_token_time,token_match_rate\n")
                avg_time = total_time / tokens_generated if tokens_generated > 0 else 0.0
                rate_str = "N/A" if match_rate is None else f"{match_rate:.6f}"
                f.write(f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_time:.6f},{rate_str}\n")
            metrics = {
                "total_latency": total_time,
                "tokens_generated": tokens_generated,
                "throughput": throughput,
                "token_match_rate": match_rate
            }
            with open(json_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Performance metrics saved to {csv_file} and {json_file}")
        except Exception as e:
            logger.error(f"Failed to save profiling data: {e}")
    else:
        logger.info("Speculative decoding completed (no profiling).")

    full_output = prompt + generated_text
    print("\n=== Final Output ===\n" + full_output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Draft worker for speculative decoding")
    parser.add_argument("--model", type=str, required=True, help="Draft model path")
    parser.add_argument("--target_host", type=str, default="localhost", help="Target server hostname")
    parser.add_argument("--port", type=int, default=50051, help="Target server port")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--target_model", type=str, help="Path/name of target model for tokenizer alignment")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="How many tokens to generate total")
    parser.add_argument("--sequence_length", type=int, default=128, help="Sequence length for compilation")
    parser.add_argument("--profile", action="store_true", help="Enable total-time profiling")
    parser.add_argument("--no_target", action="store_true", help="Run draft alone (no target contact)")
    args = parser.parse_args()

    run_client(
        draft_model_name=args.model,
        target_host=args.target_host,
        port=args.port,
        prompt=args.prompt,
        target_model_name=args.target_model,
        max_new_tokens=args.max_new_tokens,
        sequence_length=args.sequence_length,
        profile=args.profile,
        no_target=args.no_target
    )
