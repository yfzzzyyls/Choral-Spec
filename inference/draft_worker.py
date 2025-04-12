import logging
import grpc
import os
import time
import json
import threading
import uuid
from datetime import datetime

from grpc_comm import inference_pb2_grpc, inference_pb2, grpc_client
from inference.model_loader import load_model
from inference.speculative import speculative_decode, speculative_decode_batch
from transformers import AutoTokenizer
import torch

logger = logging.getLogger(__name__)


def save_perf_stats(perf_stats: dict, file_prefix: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{file_prefix}_{timestamp}.csv"
    json_path = f"{file_prefix}_{timestamp}.json"
    try:
        with open(csv_path, "w", newline='') as cf:
            cf.write("total_time,tokens_generated,throughput,avg_token_time,token_match_rate\n")
            total_time = perf_stats.get("total_time", 0.0)
            tokens_generated = perf_stats.get("tokens_generated", 0)
            throughput = perf_stats.get("throughput", 0.0)
            avg_token_time = perf_stats.get("avg_token_time", 0.0)
            token_match_rate = perf_stats.get("token_match_rate", None)
            line = f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_token_time:.6f},{token_match_rate}\n"
            cf.write(line)
        with open(json_path, "w") as jf:
            json.dump(perf_stats, jf, indent=2)
        logger.info(f"Performance metrics saved to {csv_path} and {json_path}")
    except Exception as e:
        logger.error(f"Failed to save performance data: {e}")


def run_batched_prompt_file(
    draft_model_name: str,
    target_host: str = "localhost",
    port: int = 50051,
    prompt_text_file: str = "",
    target_tokenizer: str = None,
    max_new_tokens: int = 50,
    sequence_length: int = 128,
    gamma: int = 4,
    profile: bool = False,
    no_target: bool = False,
    top_p: float = 0.9,
    temperature: float = 1.0
):
    """
    Reads N prompts from prompt_text_file, merges them into a single batch of size N 
    (for example, 2 if you have exactly 2 prompts), and uses one session_id for the entire batch. 
    We call StartGeneration once, do a single "batch finalize" of the prompts, 
    then call speculative_decode_batch to proceed.
    """
    import os
    import torch
    import logging
    from transformers import AutoTokenizer
    from grpc_comm import inference_pb2, inference_pb2_grpc, grpc_client
    from inference.model_loader import load_model
    from datetime import datetime
    logger = logging.getLogger(__name__)

    if not os.path.exists(prompt_text_file):
        logger.error(f"Prompt text file not found: {prompt_text_file}")
        return

    with open(prompt_text_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        logger.error("No valid lines in the prompt file.")
        return

    batch_size = len(prompts)
    logger.info(f"Found {batch_size} prompts in {prompt_text_file}. This code assumes you compiled batch_size={batch_size}.")

    # Load local draft
    logger.info(f"Loading draft model '{draft_model_name}' (sequence_length={sequence_length}) for batched decoding...")
    draft_model = load_model(draft_model_name, sequence_length=sequence_length)
    tokenizer_source = target_tokenizer or draft_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if no_target:
        logger.info("No target usage. Just run the draft model locally (not implemented batch).")
        return

    address = f"{target_host}:{port}"
    logger.info(f"Connecting to target server at {address}...")
    import grpc
    channel = grpc.insecure_channel(address)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)

    # We now create exactly ONE session_id for the entire batch
    import uuid
    session_id = int(uuid.uuid4()) & 0xFFFFFFFF
    # Start generation with a placeholder prompt
    # (the actual prompt is passed in a finalize step below)
    stub.StartGeneration(
        inference_pb2.StartRequest(
            session_id=session_id,
            prompt="(placeholder for multi-prompt batch)",
            max_new_tokens=max_new_tokens,
            gamma=gamma
        )
    )
    logger.info(f"Created single session_id={session_id} for all {batch_size} prompts.")

    # Now, unify all prompts into a single batch of shape [batch_size, seq_len] on the draft side
    enc = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=False
    )
    input_ids_batch = enc["input_ids"]  # [batch_size, seq_len]
    attention_mask_batch = enc["attention_mask"]  # [batch_size, seq_len]

    # We'll do an initial 'FinalizeBatchTokens' to pass the entire prompt to the target in one shot.
    # That way, the target side has shape [batch_size, prompt_len].
    # We treat each row as separate "tokens" for finalize
    # In your older code, finalize expects a single row per request => let's adapt 
    # We'll unify them as separate calls or just unify them in a single call if your proto supports repeated sequences
    # For demonstration, do multiple calls: each row => one FinalizeSequence
    finalize_seq_msgs = []
    for i in range(batch_size):
        # extract the real prompt tokens
        row = input_ids_batch[i, : attention_mask_batch[i].sum()].tolist()
        finalize_seq_msgs.append((session_id, row))

    # finalize them
    final_resps = grpc_client.finalize_batch_tokens(stub, finalize_seq_msgs)
    # the target now has shape [batch_size, prompt_len], if your code merges them. 
    # Or if your code stores them as single row, you'd do advanced merging in the target.

    logger.info("All prompts have been finalized onto the target's context. Now we do the speculative decode loop.")
    from inference.speculative import speculative_decode_batch
    final_texts, perf_stats = speculative_decode_batch(
        draft_model,
        tokenizer,
        stub,
        input_ids_batch,
        attention_mask_batch,
        max_new_tokens=max_new_tokens,
        gamma=gamma,
        profile=profile,
        top_p=top_p,
        temperature=temperature,
        session_ids=[session_id]*batch_size  # same session for all rows
    )

    print("\n=== Final Outputs (TRUE BATCH) ===")
    for i, text in enumerate(final_texts):
        print(f"[Prompt {i}]:\n{text}\n")

    # optionally save perf stats
    if profile and perf_stats:
        from datetime import datetime
        import json
        file_prefix = "performance_speculative_batch"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"{file_prefix}_{timestamp}.csv"
        json_path = f"{file_prefix}_{timestamp}.json"
        try:
            with open(csv_path, "w", newline='') as cf:
                cf.write("total_time,tokens_generated,throughput,avg_token_time,token_match_rate\n")
                total_time = perf_stats.get("total_time", 0.0)
                tokens_generated = perf_stats.get("tokens_generated", 0)
                throughput = perf_stats.get("throughput", 0.0)
                avg_token_time = perf_stats.get("avg_token_time", 0.0)
                token_match_rate = perf_stats.get("token_match_rate", None)
                line = f"{total_time:.6f},{tokens_generated},{throughput:.6f},{avg_token_time:.6f},{token_match_rate}\n"
                cf.write(line)
            with open(json_path, "w") as jf:
                json.dump(perf_stats, jf, indent=2)
            logger.info(f"Performance metrics saved to {csv_path} and {json_path}")
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")


def run_client(draft_model_name: str,
               target_host: str = "localhost",
               port: int = 50051,
               prompt: str = "",
               target_tokenizer: str = None,
               max_new_tokens: int = 50,
               sequence_length: int = 128,
               gamma: int = 4,
               profile: bool = False,
               no_target: bool = False,
               top_p: float = 0.9,
               temperature: float = 1.0):
    logger.info(f"Loading draft model '{draft_model_name}' (sequence_length={sequence_length})...")
    draft_model = load_model(draft_model_name, sequence_length=sequence_length)
    tokenizer_source = target_tokenizer or draft_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
    if not prompt:
        logger.error("No prompt provided.")
        return
    if no_target:
        return _run_standalone_draft(draft_model, tokenizer, prompt, max_new_tokens, profile)
    else:
        address = f"{target_host}:{port}"
        logger.info(f"Connecting to target server at {address}...")
        channel = grpc.insecure_channel(address)
        stub = inference_pb2_grpc.SpeculativeServiceStub(channel)
        session_id = _gen_session_id()
        stub.StartGeneration(
            inference_pb2.StartRequest(
                session_id=session_id,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                gamma=gamma
            )
        )
        logger.info(f"Starting speculative decoding (single) for prompt: '{prompt}'")
        generated_text, perf_stats = speculative_decode(
            draft_model, tokenizer, stub, prompt, max_new_tokens, gamma,
            profile=profile, top_p=top_p, temperature=temperature,
            session_id=session_id
        )
        full_output = prompt + generated_text
        print("\n=== Final Output ===\n" + full_output)
        if profile and perf_stats:
            save_perf_stats(perf_stats, file_prefix="performance_speculative")
        return full_output


def _run_standalone_draft(draft_model, tokenizer, prompt, max_new_tokens, profile):
    output_text = ""
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    tokens_generated = 0
    start_time = time.time() if profile else None
    for i in range(max_new_tokens):
        try:
            output = draft_model.sample(input_ids, sequence_length=input_ids.shape[1] + 1)
        except Exception as e:
            logger.error(f"Draft model generation failed: {e}")
            break
        token_id = int(output[0, -1]) if not isinstance(output, (list, tuple)) else int(output[0][-1])
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=True)
        output_text += token_text
        new_token_tensor = torch.tensor([[token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
        tokens_generated += 1
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
    end_time = time.time() if profile else None
    if profile and start_time is not None:
        total_time = end_time - start_time
        throughput = tokens_generated / total_time if total_time > 0 else float('inf')
        logger.info(f"Draft model generation completed in {total_time:.2f} seconds. Throughput={throughput:.2f} t/s")
    full_output = prompt + output_text
    print("\n=== Final Output ===\n" + full_output)
    return full_output


def run_concurrent_clients(draft_model_name: str,
                           target_host: str = "localhost",
                           port: int = 50051,
                           prompt_text_file: str = "",
                           target_tokenizer: str = None,
                           max_new_tokens: int = 50,
                           sequence_length: int = 128,
                           gamma: int = 4,
                           profile: bool = False,
                           no_target: bool = False,
                           top_p: float = 0.9,
                           temperature: float = 1.0):
    """
    Old concurrency approach (1 thread per prompt). We keep it for backward compatibility.
    """
    if not os.path.exists(prompt_text_file):
        logger.error(f"Prompt text file not found: {prompt_text_file}")
        return
    with open(prompt_text_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        logger.error("No valid lines in prompt file.")
        return
    logger.info(f"Loading draft model '{draft_model_name}' (sequence_length={sequence_length}) for concurrency...")
    draft_model = load_model(draft_model_name, sequence_length=sequence_length)
    tokenizer_source = target_tokenizer or draft_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
    address = f"{target_host}:{port}"
    channel = None
    stub = None
    if not no_target:
        logger.info(f"Connecting to target server at {address} for concurrency...")
        channel = grpc.insecure_channel(address)
        stub = inference_pb2_grpc.SpeculativeServiceStub(channel)
    results = [None]*len(prompts)
    threads = []

    def worker(idx, prompt_text):
        if no_target:
            out = _run_standalone_draft(draft_model, tokenizer, prompt_text, max_new_tokens, profile)
            results[idx] = out
        else:
            session_id = _gen_session_id()
            stub.StartGeneration(
                inference_pb2.StartRequest(
                    session_id=session_id,
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                    gamma=gamma
                )
            )
            logger.info(f"[Thread-{idx}] Starting speculative decoding with session_id={session_id}")
            gen_text, perf_stats = speculative_decode(
                draft_model, tokenizer, stub,
                prompt_text, max_new_tokens, gamma,
                profile=profile, top_p=top_p, temperature=temperature,
                session_id=session_id
            )
            full_output = prompt_text + gen_text
            results[idx] = full_output
            if profile and perf_stats:
                prefix = f"performance_speculative_prompt{idx}"
                save_perf_stats(perf_stats, file_prefix=prefix)

    for i, prompt_text in enumerate(prompts):
        t = threading.Thread(target=worker, args=(i, prompt_text), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\n=== Final Batched Outputs ===")
    for i, text in enumerate(results):
        print(f"\n[Prompt {i} Output]:\n{text}")


def _gen_session_id():
    return int(uuid.uuid4()) & 0xFFFFFFFF