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
    True multi-sequence speculative decoding for a batch of prompts.

    :param draft_model: The draft model (LlamaForSampling) loaded on Trainium.
    :param tokenizer: The same tokenizer used for both draft and target.
    :param stub: gRPC stub to the target service.
    :param input_ids_batch: [batch_size, seq_len] integer tensor of padded prompts.
    :param attention_mask_batch: [batch_size, seq_len] binary mask (1 for real tokens, 0 for pad).
    :param max_new_tokens: The maximum number of tokens to generate.
    :param gamma: Number of draft tokens to propose before verifying.
    :param profile: If True, measure time/perf.
    :param top_p: Nucleus sampling threshold for the draft model.
    :param temperature: Sampling temperature for the draft model.
    :param session_ids: list of session IDs (one per sequence) for the target side.

    :return: A list of token ID lists (or final texts) for each sequence.
    """
    if session_ids is None:
        # If not provided, generate one for each sequence.
        batch_size = input_ids_batch.size(0)
        from uuid import uuid4
        session_ids = [(int(uuid4()) & 0xFFFFFFFF) for _ in range(batch_size)]

    batch_size = input_ids_batch.size(0)
    device = draft_model.device if hasattr(draft_model, 'device') else torch.device('cpu')
    input_ids_batch = input_ids_batch.to(device)
    attention_mask_batch = attention_mask_batch.to(device)

    # We'll track the output tokens for each sequence in a list of lists
    output_token_lists = [[] for _ in range(batch_size)]
    # Keep track of how many new tokens each sequence has produced so far
    new_tokens_count = [0]*batch_size
    # Track if each sequence is finished
    finished_mask = [False]*batch_size

    # 1) Run the draft model on the entire batch of prompts to initialize the caches
    with torch.no_grad():
        draft_out = draft_model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            use_cache=True
        )
    # shape: [batch_size, seq_len, vocab]
    # store the past key values for each sequence
    past_key_values_draft = draft_out.past_key_values  # tuple of (k,v) for each layer

    # For convenience, we'll store them as separate structures for each sequence.
    # Actually, we can keep them in a single batched structure if the library supports indexing.

    # Transfer logits for last position for each sequence (where actual tokens end)
    # But some sequences may have different prompt lengths.
    # We'll do one token generation step if needed.

    # We'll also let the target know about the initial prompt tokens. We'll do that by finalizing
    # the existing prompt tokens once, so the target is aligned. Or we can do the StartGeneration logic.

    # 2) Keep decoding until all sequences are finished or we hit max_new_tokens.
    import time
    start_time = time.time() if profile else None

    # We'll store how many total tokens were accepted vs forced for debugging.
    accepted_tokens_total = [0]*batch_size
    forced_tokens_total = [0]*batch_size

    def all_finished():
        return all(finished_mask)

    # We assume the user has already called stub.StartGeneration(...) for each session_id.
    # If not, they'd do so externally. We'll do partial logging.

    # Main loop:
    while not all_finished():
        # Construct a smaller sub-batch of active sequences
        active_indices = [i for i, fin in enumerate(finished_mask) if not fin]
        if not active_indices:
            break  # all done

        # We'll collect up to gamma tokens for each active sequence.
        # We'll do a token-by-token approach in a batched manner.
        # 1) Create a placeholder for new tokens for each sequence.
        proposed_tokens = {i: [] for i in active_indices}  # draft-proposed tokens
        # We'll store the updated caches in a local structure for each token step.

        for step in range(gamma):
            # Identify sequences that are still active in this block
            block_active_indices = [idx for idx in active_indices if len(proposed_tokens[idx]) == step]
            if not block_active_indices:
                break  # no one left in this gamma block

            # For these sequences, we feed in their last token if step>0, else we reuse the final logits from prompt.
            # But since we typically need a forward pass for each step, we do a single batched pass.

            # We gather the last token for each block-active sequence if step>0. If step=0, we treat it as
            # we need the next token from the last prompt logits.

            if step == 0:
                # We'll do a forward pass with a dummy input = the last real token for each sequence.
                # Because we might not have the final logits from the draft_out above for each seq specifically.
                # Alternatively, we can slice the logits from draft_out, but let's do a consistent approach.

                # Construct input_ids of shape [block_active_size, 1]
                last_tokens = []
                new_past = []
                for idx in block_active_indices:
                    seq_len = attention_mask_batch[idx].sum().item()  # actual prompt length
                    # the last real token is input_ids_batch[idx, seq_len-1]
                    last_tok_id = input_ids_batch[idx, seq_len-1].unsqueeze(0)
                    last_tokens.append(last_tok_id)
                tokens_tensor = torch.stack(last_tokens, dim=0)  # shape [block_active_size, 1]

                # gather sub-batch of past_key_values
                # each layer is (k,v) shaped [batch_size, n_heads, seq_len, head_dim]
                # we'll need to index them by block_active_indices.
                sub_past = []
                for layer in past_key_values_draft:
                    k, v = layer  # shape [batch_size, n_heads, seq_len, head_dim]
                    sub_k = k[block_active_indices].clone()
                    sub_v = v[block_active_indices].clone()
                    sub_past.append((sub_k, sub_v))
                sub_past = tuple(sub_past)

                with torch.no_grad():
                    step_out = draft_model(
                        input_ids=tokens_tensor.to(device),
                        use_cache=True,
                        past_key_values=sub_past
                    )
                step_logits = step_out.logits  # shape [block_active_size, 1, vocab]
                new_sub_past = step_out.past_key_values

                # now sample a token for each sequence in block_active_indices
                for b_i, seq_idx in enumerate(block_active_indices):
                    logits_1 = step_logits[b_i, -1, :]
                    token_id = _sample_token_id(logits_1, top_p, temperature)
                    proposed_tokens[seq_idx].append(token_id)

                # We must now integrate new_sub_past back into the global past_key_values_draft.
                # We'll do it by rewriting the slices.
                for layer_idx, (k, v) in enumerate(past_key_values_draft):
                    new_k, new_v = new_sub_past[layer_idx]
                    k[block_active_indices] = new_k
                    v[block_active_indices] = new_v

            else:
                # gather last step's newly proposed token for each block-active sequence
                last_tokens = []
                for idx in block_active_indices:
                    t_id = proposed_tokens[idx][-1]
                    last_tokens.append(torch.tensor([t_id], dtype=torch.long, device=device))
                tokens_tensor = torch.stack(last_tokens, dim=0)  # shape [block_active_size, 1]

                # slice out their past states from past_key_values_draft
                sub_past = []
                for layer in past_key_values_draft:
                    k, v = layer
                    sub_k = k[block_active_indices].clone()
                    sub_v = v[block_active_indices].clone()
                    sub_past.append((sub_k, sub_v))
                sub_past = tuple(sub_past)

                with torch.no_grad():
                    step_out = draft_model(
                        input_ids=tokens_tensor,
                        use_cache=True,
                        past_key_values=sub_past
                    )
                step_logits = step_out.logits  # shape [block_active_size, 1, vocab]
                new_sub_past = step_out.past_key_values

                for b_i, seq_idx in enumerate(block_active_indices):
                    logits_1 = step_logits[b_i, -1, :]
                    token_id = _sample_token_id(logits_1, top_p, temperature)
                    proposed_tokens[seq_idx].append(token_id)

                # rewrite the global past
                for layer_idx, (k, v) in enumerate(past_key_values_draft):
                    new_k, new_v = new_sub_past[layer_idx]
                    k[block_active_indices] = new_k
                    v[block_active_indices] = new_v

            # handle EOS if any
            eos_hits = []
            for seq_idx in block_active_indices:
                if tokenizer.eos_token_id is not None and proposed_tokens[seq_idx][-1] == tokenizer.eos_token_id:
                    # this seq is done in this block
                    eos_hits.append(seq_idx)

            # remove eos hits from active set for remainder of this gamma block
            if eos_hits:
                for eidx in eos_hits:
                    # done in the block, do not generate more tokens
                    active_indices.remove(eidx)
        # end of gamma block generation

        # Now we do a single verifyBatchTokens call
        verify_seq_list = []
        for i in range(batch_size):
            if not finished_mask[i] and proposed_tokens[i]:
                verify_seq_list.append((session_ids[i], proposed_tokens[i]))
        if not verify_seq_list:
            # no new tokens proposed
            break

        verify_results = grpc_client.verify_batch_tokens(stub, verify_seq_list)
        # Each result has: session_id, tokens_accepted, target_token, finished

        # We'll store a map from session_id to (accepted_count, target_token, finished)
        result_map = {}
        for r in verify_results:
            sid = r['session_id']
            result_map[sid] = {
                'accepted_count': r['tokens_accepted'],
                'target_token': r['target_token'],
                'finished': r['finished'],
            }

        # We also prepare finalize requests per sequence
        finalize_data = {}

        # For each sequence we proposed tokens for, interpret the verify result
        for i in range(batch_size):
            if i not in proposed_tokens or not proposed_tokens[i]:
                continue
            if finished_mask[i]:
                continue
            sid = session_ids[i]
            if sid not in result_map:
                # not in results => no tokens accepted
                continue
            rr = result_map[sid]
            accepted_count = rr['accepted_count']
            mismatch_token = rr['target_token']
            seq_finished = rr['finished']

            # get the block tokens
            block_toks = proposed_tokens[i]
            block_len = len(block_toks)

            # accepted tokens = block_toks[:accepted_count]
            accepted_tokens = block_toks[:accepted_count]
            # if mismatch_token != 0, that's the target's chosen token after the break
            # or 0 if no mismatch.

            # remove unaccepted tokens from the local output
            unaccepted = block_len - accepted_count
            # pop them from the draft side
            # But we haven't appended them to output_token_lists yet. We'll do so now only if accepted.

            # append the accepted tokens
            output_token_lists[i].extend(accepted_tokens)
            accepted_tokens_total[i] += len(accepted_tokens)
            new_tokens_count[i] += len(accepted_tokens)

            # handle mismatch token
            if mismatch_token != 0:
                # that means a divergence happened, we must also accept mismatch_token into final output
                output_token_lists[i].append(mismatch_token)
                forced_tokens_total[i] += 1
                new_tokens_count[i] += 1

                # we need to adjust the draft model's cache to reflect the mismatch.
                # roll back to the state after accepted_count tokens in the block

                # We'll do so by forcing a re-compute from the last known good state.
                # simplest approach: re-build the draft state from the final output.
                # But that might be slow.
                # for demonstration, we do a small re-encode. Or we can do a small forward pass for mismatch_token.

                # We'll re-encode from the start if short sequences. Otherwise we do partial.
                # For large sequences, you'd implement a more advanced approach.
                new_context = torch.tensor([output_token_lists[i]], dtype=torch.long, device=device)
                new_mask = torch.ones_like(new_context)
                with torch.no_grad():
                    re_out = draft_model(
                        input_ids=new_context,
                        attention_mask=new_mask,
                        use_cache=True
                    )
                # update the global past at index i
                past_key_values_draft = _update_seq_in_past(
                    past_key_values_draft,
                    re_out.past_key_values,
                    seq_idx=i
                )
            else:
                # all tokens were accepted
                # do nothing special except keep the draft model's existing cache
                pass

            if seq_finished or (tokenizer.eos_token_id is not None and len(output_token_lists[i])>0 and output_token_lists[i][-1] == tokenizer.eos_token_id):
                finished_mask[i] = True

            # finalize tokens for the target side.
            # if mismatch_token != 0, finalize = accepted_tokens + [mismatch_token]
            # else finalize = accepted_tokens only
            finalize_list = accepted_tokens
            if mismatch_token != 0:
                finalize_list = accepted_tokens + [mismatch_token]

            if finalize_list:
                finalize_data[i] = finalize_list

        # Now call finalize_batch_tokens
        finalize_seq_msgs = []
        for i in finalize_data:
            sid = session_ids[i]
            finalize_seq_msgs.append((sid, finalize_data[i]))
        if finalize_seq_msgs:
            finalize_resps = grpc_client.finalize_batch_tokens(stub, finalize_seq_msgs)
            # we can parse if needed. If any are finished, we mark them.
            for r in finalize_resps:
                sid = r['session_id']
                fin = r['finished']
                # find which i has this sid
                i2 = session_ids.index(sid)
                if fin:
                    finished_mask[i2] = True
        # end iteration

        # also check if any sequences have gone beyond max_new_tokens
        for i in range(batch_size):
            if not finished_mask[i] and new_tokens_count[i] >= max_new_tokens:
                finished_mask[i] = True

    # end while

    end_time = time.time() if profile else None
    perf_stats = {}
    if profile:
        total_elapsed = end_time - start_time
        # sum up all tokens
        total_accepted = sum(accepted_tokens_total)
        total_forced = sum(forced_tokens_total)
        total_tokens = total_accepted + total_forced
        throughput = total_tokens / total_elapsed if total_elapsed > 0 else 0.0
        perf_stats["total_time"] = total_elapsed
        perf_stats["tokens_generated"] = total_tokens
        perf_stats["throughput"] = throughput
        perf_stats["avg_token_time"] = total_elapsed / total_tokens if total_tokens>0 else 0.0
        if total_tokens>0:
            match_rate = total_accepted / total_tokens
            perf_stats["token_match_rate"] = match_rate
            logger.info(
                f"[BATCH] Speculative decoding match rate: {match_rate:.2%}"
                f" (Draft accepted: {total_accepted}, Target generated: {total_forced})"
            )

    # decode final text for each sequence
    final_texts = []
    for i in range(batch_size):
        # strip the prompt tokens from the final output, or we can just append to them.
        # we'll do a simple approach: combine all tokens in output_token_lists.
        # if you want to exclude the prompt, you'll need to skip that many tokens.
        text = tokenizer.decode(output_token_lists[i], skip_special_tokens=True)
        final_texts.append(text)

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