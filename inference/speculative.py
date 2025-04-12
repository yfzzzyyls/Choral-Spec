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
    TRUE Single-Pass Multi-Sequence decoding: each chunk => EXACTLY one forward pass
    that yields [batch_size, current_length + gamma, vocab], from which we extract
    the last 'gamma' tokens for each row.

    This matches lucidrains' approach more closely:
      - No "for step_i in range(gamma)" loop on the draft side
      - We rely on the model doing a single pass for gamma tokens

    NOTE: For big chunk generation (gamma>1), you must ensure the draft model
    can generate multiple tokens in a single pass. For huggingface, that might
    require a 'generate()' call or a custom approach with 'model(..., use_cache=True, do_sample=True, num_return_sequences=gamma, ???)'.

    Because 'transformers_neuronx.LlamaForSampling' does not natively do
    multi-token single forward pass, we do a pseudo approach: feed [batch_size, old_len+gamma]
    as a single input with a BFS method. This is for demonstration only.
    """

    import time
    import torch
    import random

    logger.info("=== Starting TRUE Single-Pass Multi-Sequence Decoding ===")
    start_t = time.time() if profile else None

    # Get device from the draft model's first parameter
    param = next(draft_model.parameters(), None)
    device = param.device if param is not None else torch.device('cpu')

    batch_size = input_ids_batch.size(0)
    input_ids_batch = input_ids_batch.to(device)
    attention_mask_batch = attention_mask_batch.to(device)

    if session_ids is None:
        from uuid import uuid4
        session_ids = [int(uuid4()) & 0xFFFFFFFF for _ in range(batch_size)]

    # Convert the input into lists of tokens
    # initial output_tokens for each row
    output_tokens = []
    row_lengths = attention_mask_batch.sum(dim=1).tolist()
    for i in range(batch_size):
        row_toks = input_ids_batch[i, : row_lengths[i]].tolist()
        output_tokens.append(row_toks)

    new_tokens_count = [0]*batch_size
    finished_mask = [False]*batch_size

    accepted_tokens_total = [0]*batch_size
    forced_tokens_total = [0]*batch_size

    def all_finished():
        return all(finished_mask)

    while not all_finished():
        # find how many tokens each row can still generate
        tokens_left = [max_new_tokens - new_tokens_count[i] for i in range(batch_size)]
        chunk_sizes = []
        for i in range(batch_size):
            if finished_mask[i]:
                chunk_sizes.append(0)
            else:
                chunk_sizes.append(min(tokens_left[i], gamma))

        max_chunk = max(chunk_sizes)
        if max_chunk == 0:
            # no one can produce more tokens => break
            break

        # Build a single input of shape [batch_size, max_old_len + max_chunk]
        # We do so by each row:
        # row context = existing output_tokens + pad for the chunk
        # We'll rely on the model generating the last 'max_chunk' tokens for each row
        # In real lucidrains, you'd do an incremental approach with KV caching. For demonstration,
        # we do a single pass re-encode: [batch_size, old_len_i + chunk_sizes[i]] => pad to max => shape [batch_size, context_len].
        old_lens = []
        for i in range(batch_size):
            old_lens.append(len(output_tokens[i]))

        max_old_len = max(old_lens)
        # The final context length for each row = old_len[i] + chunk_sizes[i], up to (max_old_len + max_chunk)
        final_context_len = max_old_len + max_chunk

        # build padded batch
        padded_input = []
        for i in range(batch_size):
            row_context_len = old_lens[i] + chunk_sizes[i]
            row_data = output_tokens[i]
            # if chunk_sizes[i]>0, we must pad out with dummy tokens to let the model produce them
            # but 'transformers' by default doesn't do partial generation easily. We'll do a naive approach:
            # replicate the last token chunk_sizes[i] times. This is a hack, because real single pass multi-token
            # would do some form of BFS. We do a simpler approach: row_data + (chunk_sizes[i] dummy tokens).
            # We'll forcibly fill them with pad_token_id or last token
            # so we have the shape [row_context_len].
            # Then we'll pad to final_context_len.

            row_cp = row_data[:]
            # add chunk_sizes[i] pad tokens
            row_cp += [tokenizer.pad_token_id]*(chunk_sizes[i])
            # Now row_cp has length old_lens[i]+chunk_sizes[i].
            # if that < final_context_len => pad more
            needed_pad = final_context_len - (old_lens[i] + chunk_sizes[i])
            if needed_pad>0:
                row_cp += [tokenizer.pad_token_id]*needed_pad
            # row_cp now has length final_context_len
            padded_input.append(torch.tensor(row_cp, dtype=torch.long, device=device))

        batched_ids = torch.stack(padded_input, dim=0)  # shape [batch_size, final_context_len]

        # forward pass in one shot => shape [batch_size, final_context_len, vocab]
        with torch.no_grad():
            # This might not do exactly gamma new tokens, but we treat the last chunk_sizes[i] positions as the newly generated tokens
            outputs = draft_model(batched_ids)
        if isinstance(outputs, (tuple, list)):
            logits_3d = outputs[0]
        else:
            logits_3d = outputs

        # shape => [batch_size, final_context_len, vocab]
        if logits_3d.dim()==2:
            # single step => [batch_size, vocab], expand
            logits_3d = logits_3d.unsqueeze(1)

        # For each row, sample the last chunk_sizes[i] tokens from the last chunk_sizes[i] positions
        # those positions => range( old_lens[i], old_lens[i]+chunk_sizes[i] ) in logits_3d
        # then do top-p sampling for each token
        proposed_tokens = {i: [] for i in range(batch_size)}
        for i in range(batch_size):
            c_size = chunk_sizes[i]
            old_len_i = old_lens[i]
            if c_size<=0:
                continue
            for step_i in range(c_size):
                pos = old_len_i + step_i
                row_logits = logits_3d[i, pos, :] / temperature
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
                proposed_tokens[i].append(new_token)

        # now we have up to chunk_sizes[i] newly proposed tokens for each row
        # we do one verifyBatchTokens call
        verify_list = []
        for i in range(batch_size):
            if not finished_mask[i] and proposed_tokens[i]:
                verify_list.append((session_ids[i], proposed_tokens[i]))

        if not verify_list:
            # nothing to verify => all done
            break

        verify_results = grpc_client.verify_batch_tokens(stub, verify_list)
        # parse
        result_map = {}
        for r in verify_results:
            sid = r['session_id']
            result_map[sid] = {
                'tokens_accepted': r['tokens_accepted'],
                'target_token': r['target_token'],
                'finished': r['finished']
            }

        # partial acceptance => rollback or forced mismatch
        finalize_data = {}
        for i in range(batch_size):
            if i not in proposed_tokens or not proposed_tokens[i]:
                continue
            if finished_mask[i]:
                continue
            sid = session_ids[i]
            if sid not in result_map:
                continue
            accepted_count = result_map[sid]['tokens_accepted']
            mismatch_token = result_map[sid]['target_token']
            row_finished = result_map[sid]['finished']
            c_toks = proposed_tokens[i]

            not_accepted = len(c_toks)-accepted_count
            accepted_tokens_total[i]+= accepted_count
            # remove unaccepted from output_tokens
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

            if row_finished or (tokenizer.eos_token_id is not None and len(output_tokens[i])>0 and output_tokens[i][-1] == tokenizer.eos_token_id):
                finished_mask[i] = True

            # finalize tokens
            fin_list = c_toks[:accepted_count]
            if forced_tok is not None:
                fin_list.append(forced_tok)
            if fin_list:
                finalize_data[i] = fin_list

        # finalize
        final_msg = []
        for i,flist in finalize_data.items():
            final_msg.append((session_ids[i], flist))
        if final_msg:
            fresp = grpc_client.finalize_batch_tokens(stub, final_msg)
            for rr in fresp:
                sid = rr['session_id']
                fin = rr['finished']
                i2 = session_ids.index(sid)
                if fin:
                    finished_mask[i2] = True

        # check if we exceed max
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
            logger.info(f"[TRUE BATCH] Speculative decoding match rate: {match_rate:.2%} (Draft accepted: {tot_acc}, Target generated: {tot_forced})")

    # decode
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