import random
import torch
import logging

from grpc_comm import grpc_client

logger = logging.getLogger(__name__)

############################################################
# Single-sequence speculative_decode (existing, unchanged) #
############################################################
def speculative_decode(
    draft_model,
    tokenizer,
    stub,
    prompt,
    max_new_tokens,
    gamma,
    profile=False,
    top_p=0.9,
    temperature=1.0,
    session_id=0
):
    """
    Previous single-sequence speculative decoding logic.
    Left intact for backward compatibility.
    """
    output_tokens = []
    tokens_generated = 0
    finished = False

    accepted_tokens_total = 0
    target_tokens_total = 0

    # Tokenize the user prompt for the first pass
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids

    past = None

    import time
    start_t = time.time() if profile else None

    while not finished and tokens_generated < max_new_tokens:
        # The draft model proposes up to 'gamma' tokens
        draft_tokens = []
        draft_probs = []
        past_states = [past]

        for i in range(gamma):
            if past is None:
                if output_tokens:
                    input_ids = torch.tensor([output_tokens], dtype=torch.long)
                else:
                    input_ids = prompt_ids
            else:
                if output_tokens:
                    last_token_id = torch.tensor([[output_tokens[-1]]], dtype=torch.long)
                else:
                    last_token_id = prompt_ids
                input_ids = last_token_id

            outputs = draft_model(input_ids=input_ids, use_cache=True, past_key_values=past)
            try:
                logits = outputs.logits[0, -1, :]
            except AttributeError:
                logits = outputs[0]

            logits = logits / temperature
            logits = torch.clamp(logits, min=-1e10, max=1e10)

            probs = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all():
                probs = torch.ones_like(probs)
                probs /= probs.sum()

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            if torch.any(cumulative_probs >= top_p):
                cutoff_index = torch.where(cumulative_probs >= top_p)[0][0].item()
            else:
                cutoff_index = len(sorted_probs) - 1

            top_probs = sorted_probs[:cutoff_index+1]
            top_indices = sorted_indices[:cutoff_index+1]

            top_sum = top_probs.sum()
            if not torch.isfinite(top_sum) or top_sum <= 1e-9:
                top_probs = torch.ones_like(top_probs)
                top_sum = top_probs.sum()

            top_probs = top_probs / top_sum
            top_probs = torch.clamp(top_probs, min=0.0, max=1.0)

            choice_index = torch.multinomial(top_probs, 1).item()
            token_id = int(top_indices[choice_index].item())
            token_prob = float(top_probs[choice_index].item())

            draft_tokens.append(token_id)
            draft_probs.append(token_prob)
            output_tokens.append(token_id)
            tokens_generated += 1

            new_past = getattr(outputs, "past_key_values", None)
            past_states.append(new_past)
            past = new_past

            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                finished = True
                break
            if tokens_generated >= max_new_tokens:
                break

        if len(draft_tokens) > 0 and tokens_generated > max_new_tokens:
            overshoot = tokens_generated - max_new_tokens
            draft_tokens = draft_tokens[:-overshoot]
            draft_probs = draft_probs[:-overshoot]
            output_tokens = output_tokens[:-overshoot]
            tokens_generated = max_new_tokens
            finished = True

        if not draft_tokens:
            break

        target_probs, target_finished = grpc_client.verify_draft_tokens(
            stub, draft_tokens, session_id=session_id
        )
        if target_finished and len(target_probs) < len(draft_tokens):
            draft_tokens = draft_tokens[:len(target_probs)]
            draft_probs = draft_probs[:len(target_probs)]
            finished = True

        accept_count = 0
        break_point = False
        import random
        for idx, token_id in enumerate(draft_tokens):
            p_target = float(target_probs[idx]) if idx < len(target_probs) else 0.0
            p_draft = float(draft_probs[idx]) if idx < len(draft_probs) else 1e-9
            ratio = p_target / p_draft if p_draft > 0 else 0.0
            if ratio > 1.0:
                ratio = 1.0

            if random.random() < ratio:
                accept_count += 1
                accepted_tokens_total += 1
                if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                    finished = True
                    break_point = True
                    break
            else:
                break_point = True
                break

        if break_point and accept_count < len(draft_tokens):
            unaccepted = len(draft_tokens) - accept_count
            while unaccepted > 0:
                output_tokens.pop()
                tokens_generated -= 1
                unaccepted -= 1
            past = past_states[accept_count]

        final_token_id, finalize_finished = grpc_client.finalize_tokens(
            stub, accept_count, len(draft_tokens), session_id=session_id
        )
        if final_token_id != 0:
            output_tokens.append(final_token_id)
            tokens_generated += 1
            target_tokens_total += 1
            if tokenizer.eos_token_id is not None and final_token_id == tokenizer.eos_token_id:
                finished = True

        if finalize_finished or tokens_generated >= max_new_tokens:
            finished = True

    generated_text = tokenizer.decode(output_tokens[-tokens_generated:]) if output_tokens else ""

    perf_stats = {}
    if profile:
        end_t = time.time()
        total_time = end_t - start_t
        tokens_generated_total = accepted_tokens_total + target_tokens_total
        throughput = tokens_generated_total / total_time if total_time>0 else 0.0
        perf_stats["total_time"] = total_time
        perf_stats["tokens_generated"] = tokens_generated_total
        perf_stats["throughput"] = throughput
        perf_stats["avg_token_time"] = total_time / tokens_generated_total if tokens_generated_total>0 else 0.0

    total_output_tokens = accepted_tokens_total + target_tokens_total
    if total_output_tokens > 0:
        match_rate = accepted_tokens_total / total_output_tokens
        logger.info(
            f"Speculative decoding match rate: {match_rate:.2%} "
            f"(Draft accepted: {accepted_tokens_total}, Target generated: {target_tokens_total})"
        )
        perf_stats["token_match_rate"] = match_rate

    return generated_text, perf_stats

###############################################
# Multi-sequence true batch decode (new code) #
###############################################

###############################################
# Multi-sequence true batch decode (new code) #
###############################################

def speculative_decode_batch(
    draft_model,
    tokenizer,
    stub,
    input_ids_batch: torch.Tensor,
    attention_mask_batch: torch.Tensor,
    max_new_tokens: int = 50,
    gamma: int = 4,
    profile: bool = False,
    top_p: float = 0.9,
    temperature: float = 1.0,
    session_ids=None
):
    """
    True single-pass multi-sequence decoding for multiple tokens per chunk (gamma) 
    across multiple sequences (batch_size). Each chunk => shape [batch_size, gamma].
    Then we do exactly ONE forward pass => [batch_size, gamma, vocab].
    This is the lucidrains approach for real multi-sequence, partial acceptance.

    NOTE: We do NOT use 'draft_model.device'. Instead, we get device via next(draft_model.parameters()).device.
    """

    import time
    import random

    start_t = time.time() if profile else None

    # Determine device from model parameters (LlamaForSampling has no .device attribute)
    device_params = next(draft_model.parameters(), None)
    device = device_params.device if device_params is not None else torch.device('cpu')

    input_ids_batch = input_ids_batch.to(device)
    attention_mask_batch = attention_mask_batch.to(device)
    batch_size = input_ids_batch.size(0)

    # If no session_ids, create them
    if session_ids is None:
        from uuid import uuid4
        session_ids = [int(uuid4()) & 0xFFFFFFFF for _ in range(batch_size)]

    # Extract initial prompts
    output_tokens = []
    lengths = attention_mask_batch.sum(dim=1).tolist()
    for i in range(batch_size):
        prompt_len = lengths[i]
        prompt_ids = input_ids_batch[i, :prompt_len].tolist()
        output_tokens.append(prompt_ids)

    new_tokens_count = [0]*batch_size
    finished_mask = [False]*batch_size

    accepted_tokens_total = [0]*batch_size
    forced_tokens_total = [0]*batch_size

    def all_done():
        return all(finished_mask)

    while not all_done():
        logger.debug("Entering decode loop iteration. Active indices:")
        active_indices = [i for i, fin in enumerate(finished_mask) if not fin]
        logger.debug(f"  active_indices={active_indices}")
        if not active_indices:
            logger.debug("No active sequences left. Breaking.")
            break

        # how many tokens left for each row
        tokens_left = [max_new_tokens - new_tokens_count[i] for i in range(batch_size)]
        # row_chunk_size[i] = min(tokens_left[i], gamma) if row i not finished
        row_chunk_size = []
        for i in range(batch_size):
            if finished_mask[i]:
                row_chunk_size.append(0)
            else:
                row_chunk_size.append(min(tokens_left[i], gamma))

        max_chunk = max(row_chunk_size)
        logger.debug(f"row_chunk_size={row_chunk_size}, max_chunk={max_chunk}")

        # If nobody can produce more tokens, break out
        if max_chunk == 0:
            logger.debug("max_chunk=0 => No more tokens to generate. Stopping decode.")
            break

        # We'll do step_i from 0..(max_chunk-1)
        for step_i in range(max_chunk):
            # Build sub-batch of shape [batch_size, 1]
            gather_input = []
            for i in range(batch_size):
                if row_chunk_size[i] > step_i:
                    # propose one new token for row i
                    last_token_id = output_tokens[i][-1] if new_tokens_count[i] > 0 else output_tokens[i][-1]
                    gather_input.append(last_token_id)
                else:
                    # pad
                    gather_input.append(tokenizer.pad_token_id or 0)

            logger.debug(f"  step {step_i}: gather_input={gather_input}")
            input_tensor = torch.tensor(gather_input, dtype=torch.long, device=device).unsqueeze(1)
            draft_out = draft_model(input_ids=input_tensor)
            if isinstance(draft_out, (tuple, list)):
                logits_3d = draft_out[0]
            else:
                logits_3d = draft_out

            if logits_3d.dim() == 2:
                # shape [batch_size, vocab]
                logits_3d = logits_3d.unsqueeze(1)  # => [batch_size, 1, vocab]

            # top-p sample each row that is still generating at step_i
            for b_i in range(batch_size):
                if row_chunk_size[b_i] > step_i:
                    row_logits = logits_3d[b_i, -1, :] / temperature
                    row_probs = torch.softmax(row_logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(row_probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=0)
                    cutoff_ix = (cumsum >= top_p).nonzero(as_tuple=True)
                    if len(cutoff_ix[0])>0:
                        cut = cutoff_ix[0][0].item()
                    else:
                        cut = len(sorted_probs)-1
                    keep_probs = sorted_probs[:cut+1]
                    keep_ix = sorted_indices[:cut+1]
                    ssum = keep_probs.sum()
                    if ssum<1e-9:
                        keep_probs = torch.ones_like(keep_probs)
                        ssum = keep_probs.sum()
                    keep_probs = keep_probs / ssum
                    choice = torch.multinomial(keep_probs, 1).item()
                    new_token = keep_ix[choice].item()
                    output_tokens[b_i].append(new_token)
                    new_tokens_count[b_i]+=1

        # Now we have row_chunk_size[i] newly proposed tokens for each row => block_props
        block_props = {}
        for i in range(batch_size):
            cl = row_chunk_size[i]
            if cl>0:
                block_toks = output_tokens[i][-cl:]
            else:
                block_toks = []
            block_props[i] = block_toks
        logger.debug(f"Block proposals => {block_props}")

        # call verify
        verify_list = []
        for i in range(batch_size):
            if not finished_mask[i] and block_props[i]:
                verify_list.append((session_ids[i], block_props[i]))
        if not verify_list:
            logger.debug("No draft tokens to verify for this chunk. Continuing decode loop.")
            continue

        logger.debug(f"Calling verify_batch_tokens with {verify_list}")
        verify_results = grpc_client.verify_batch_tokens(stub, verify_list)
        logger.debug(f"verify_results={verify_results}")
        result_map = {}
        for r in verify_results:
            sid = r['session_id']
            result_map[sid] = {
                'tokens_accepted': r['tokens_accepted'],
                'target_token': r['target_token'],
                'finished': r['finished']
            }

        finalize_data = {}
        for i in range(batch_size):
            if i not in block_props or not block_props[i]:
                continue
            if finished_mask[i]:
                continue
            sid = session_ids[i]
            if sid not in result_map:
                continue
            accepted_count = result_map[sid]['tokens_accepted']
            mismatch_token = result_map[sid]['target_token']
            row_finished = result_map[sid]['finished']
            chunk_toks = block_props[i]

            not_accepted = len(chunk_toks) - accepted_count
            accepted_tokens_total[i]+= accepted_count

            while not_accepted>0:
                output_tokens[i].pop()
                new_tokens_count[i]-=1
                not_accepted-=1

            forced_tok = None
            if mismatch_token != 0:
                forced_tok = mismatch_token
                forced_tokens_total[i]+=1
                output_tokens[i].append(forced_tok)
                new_tokens_count[i]+=1

            if row_finished or (tokenizer.eos_token_id is not None and len(output_tokens[i])>0 and output_tokens[i][-1]==tokenizer.eos_token_id):
                finished_mask[i] = True

            fin_list = chunk_toks[:accepted_count]
            if forced_tok is not None:
                fin_list.append(forced_tok)
            if fin_list:
                finalize_data[i] = fin_list

        # finalize
        final_seq_msgs = []
        for i, tokens_to_fin in finalize_data.items():
            final_seq_msgs.append((session_ids[i], tokens_to_fin))
        if final_seq_msgs:
            fresps = grpc_client.finalize_batch_tokens(stub, final_seq_msgs)
            logger.debug(f"finalize_batch_tokens => {fresps}")
            for fr in fresps:
                sid = fr['session_id']
                fin = fr['finished']
                i2 = session_ids.index(sid)
                if fin:
                    finished_mask[i2] = True

        # check max
        for i in range(batch_size):
            if not finished_mask[i] and new_tokens_count[i]>=max_new_tokens:
                finished_mask[i] = True

    end_t = time.time() if profile else None
    perf_stats = {}
    if profile:
        total_time = end_t - start_t
        tot_acc = sum(accepted_tokens_total)
        tot_forced = sum(forced_tokens_total)
        tot_tokens = tot_acc+tot_forced
        throughput = tot_tokens/total_time if total_time>0 else 0.0
        perf_stats["total_time"] = total_time
        perf_stats["tokens_generated"] = tot_tokens
        perf_stats["throughput"] = throughput
        perf_stats["avg_token_time"] = total_time/tot_tokens if tot_tokens>0 else 0.0
        if tot_tokens>0:
            match_rate = tot_acc/tot_tokens
            perf_stats["token_match_rate"] = match_rate
            logger.info(
                f"[BATCH] Speculative decoding match rate: {match_rate:.2%} "
                f"(Draft accepted: {tot_acc}, Target generated: {tot_forced})"
            )

    final_texts = []
    for i in range(batch_size):
        txt = tokenizer.decode(output_tokens[i], skip_special_tokens=True)
        final_texts.append(txt)

    return final_texts, perf_stats



def _sample_token_id(logits: torch.Tensor, top_p: float, temperature: float):
    logits = logits / temperature
    logits = torch.clamp(logits, min=-1e10, max=1e10)
    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all():
        probs = torch.ones_like(probs)
        probs /= probs.sum()

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    if torch.any(cumulative_probs >= top_p):
        cutoff_index = torch.where(cumulative_probs >= top_p)[0][0].item()
    else:
        cutoff_index = len(sorted_probs) - 1

    top_probs = sorted_probs[:cutoff_index+1]
    top_indices = sorted_indices[:cutoff_index+1]

    top_sum = top_probs.sum()
    if not torch.isfinite(top_sum) or top_sum <= 1e-9:
        top_probs = torch.ones_like(top_probs)
        top_sum = top_probs.sum()

    top_probs = top_probs / top_sum
    choice_index = torch.multinomial(top_probs, 1).item()
    token_id = int(top_indices[choice_index].item())
    return token_id


def _update_seq_in_past(global_past, new_seq_past, seq_idx):
    """
    Replaces the slice corresponding to `seq_idx` in global_past with new_seq_past.
    """
    # each is a tuple of (k,v) shape [batch_size, n_heads, seq_len, head_dim]
    out = []
    for layer_idx, (gk, gv) in enumerate(global_past):
        nk, nv = new_seq_past[layer_idx]
        # gk, gv shape: [batch_size, ...]
        # nk, nv shape: [1, ...]
        # slice out seq_idx
        gk[seq_idx] = nk[0]
        gv[seq_idx] = nv[0]
        out.append((gk, gv))
    return tuple(out)