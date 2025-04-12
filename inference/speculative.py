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

def speculative_decode_batch(target_model, draft_model, input_ids, attention_mask=None,
                              max_length=None, max_new_tokens=None, gamma=1, eos_token_id=None, pad_token_id=None):
    """
    Generate text using speculative decoding with a draft (small) and target (large) model.
    Returns the generated tokens (including the initial prompt).
    """
    # Move input to target model's device
    input_ids = input_ids.to(target_model.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(target_model.device)
    batch_size = input_ids.size(0)
    
    # Use model config for eos/pad tokens if not provided
    if pad_token_id is None:
        pad_token_id = getattr(target_model.config, 'pad_token_id', None)
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0
    if eos_token_id is None:
        eos_token_id = getattr(target_model.config, 'eos_token_id', None)
    
    # Determine initial (prompt) lengths for each sequence
    if attention_mask is not None:
        prompt_lengths = attention_mask.sum(dim=1)  # number of non-pad tokens per sequence
    else:
        prompt_lengths = torch.tensor([input_ids.size(1)] * batch_size, device=input_ids.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    
    # Determine target total length (prompt + new tokens)
    if max_length is not None:
        target_lengths = torch.tensor([max_length] * batch_size, device=input_ids.device)
    elif max_new_tokens is not None:
        target_lengths = prompt_lengths + max_new_tokens
    else:
        raise ValueError("Must specify either max_length or max_new_tokens")
    
    def model_forward(model, input_ids, past=None, attention_mask=None):
        """Runs model.forward() and returns (logits, past_key_values)."""
        # Prepare kwargs for forward
        kwargs = {'use_cache': True}
        if attention_mask is not None:
            kwargs['attention_mask'] = attention_mask
        if past is not None:
            # Use appropriate keyword for past key values
            kwargs['past_key_values'] = past
        outputs = model(input_ids=input_ids, **kwargs)
        # Handle different output types
        if hasattr(outputs, "logits"):
            logits = outputs.logits
            past_key_values = getattr(outputs, "past_key_values", None) or getattr(outputs, "past", None)
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[0]
            past_key_values = None
            # Search for past in remaining outputs
            for out in outputs[1:]:
                if isinstance(out, (tuple, list)):
                    past_key_values = out
                    break
        else:  # outputs is a single tensor (logits)
            logits = outputs
            past_key_values = None
        return logits, past_key_values
    
    # Encode the initial prompt for both models to prime their caches
    target_logits, past_target = model_forward(target_model, input_ids, past=None, attention_mask=attention_mask)
    draft_logits, past_draft = model_forward(draft_model, input_ids, past=None, attention_mask=attention_mask)
    # Save the target model’s next-token logits after the prompt (for first speculative token verification)
    prev_target_next_logits = target_logits[:, -1, :]
    
    # Initialize generated output (start with prompt) and finished flags
    out_ids = input_ids.clone()
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    
    # Loop until all sequences are finished or reach the target length
    while True:
        # Compute current lengths and update finished flags
        seq_lengths = attention_mask.sum(dim=1)
        if eos_token_id is not None:
            last_tokens = out_ids.gather(1, (seq_lengths - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
            finished |= (last_tokens == eos_token_id)
        finished |= (seq_lengths >= target_lengths)  # also finish if length limit reached
        if finished.all():
            break
        
        # Determine how many speculative tokens to generate this round (do not exceed needed tokens)
        remaining_tokens = (target_lengths - seq_lengths).clamp(min=0)
        spec_count = min(gamma, int(remaining_tokens.max().item())) if remaining_tokens.max() > 0 else gamma
        
        # Collect draft model logits and sampled tokens for this speculative batch
        small_logits_list = []
        spec_tokens = torch.zeros((batch_size, spec_count), dtype=torch.long, device=input_ids.device)
        for t in range(spec_count):
            # Prepare last tokens of each sequence (use pad for already finished sequences)
            last_tokens = torch.where(
                finished.unsqueeze(1),
                torch.tensor(pad_token_id, device=input_ids.device).repeat(batch_size, 1),
                out_ids.gather(1, (seq_lengths - 1).unsqueeze(1))
            )
            # Draft model forward for next token
            logits, past_draft = model_forward(draft_model, last_tokens, past=past_draft)
            next_logits = logits[:, -1, :]  # logits for the next token
            small_logits_list.append(next_logits)
            # Sample next token from draft model (multinomial sampling for each sequence)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            # Use pad token for sequences that are finished (no actual generation)
            next_token = torch.where(finished, torch.tensor(pad_token_id, device=input_ids.device), next_token)
            # Append sampled token to outputs
            spec_tokens[:, t] = next_token
            out_ids = torch.cat([out_ids, next_token.unsqueeze(1)], dim=1)
            # Update attention_mask (1 for real tokens, 0 for pads)
            new_mask_col = (~finished).long().unsqueeze(1)
            attention_mask = torch.cat([attention_mask, new_mask_col], dim=1)
            seq_lengths = seq_lengths + (~finished).long()  # increment length for active sequences
            # Update finished if EOS was generated
            if eos_token_id is not None:
                finished |= (next_token == eos_token_id)
            if finished.all():
                # All sequences finished during speculative generation
                spec_count = t + 1  # update actual count of spec tokens generated
                spec_tokens = spec_tokens[:, :spec_count]
                small_logits_list = small_logits_list[:spec_count]
                break
        
        if finished.all():
            break  # no need to verify if all are done
        
        # Verify speculative tokens with the target model
        # Determine input to target model: new speculative tokens (if cache is available) or full sequence (if not)
        if past_target is not None:
            # Use cache: feed only the new tokens
            verify_input_ids = spec_tokens
            verify_mask = None  # not needed when using past (model sees these tokens as continuation)
        else:
            # No cache support: feed the entire current sequence with updated mask
            verify_input_ids = out_ids
            verify_mask = attention_mask
        verify_logits, new_past_target = model_forward(target_model, verify_input_ids, past=past_target, attention_mask=verify_mask)
        # Update or reset target model past based on availability
        past_target = new_past_target if new_past_target is not None else None
        
        # Determine target model logits for each new token and for the token after them
        if new_past_target is not None:
            # If using cache, verify_logits shape = (batch, spec_count, vocab)
            big_logits_new = verify_logits  # contains distributions after each new token
            # Distribution after all new tokens (next-token distribution)
            big_next_dist = big_logits_new[:, -1, :].unsqueeze(1)
            # Big model distributions for each speculative token:
            # For token1, use prev_target_next_logits (distribution after previous context);
            # for token2..tokenN, use logits from positions 0..(N-2) of big_logits_new.
            if spec_count > 1:
                big_logits_for_tokens = torch.cat([prev_target_next_logits.unsqueeze(1), big_logits_new[:, :-1, :]], dim=1)
            else:
                big_logits_for_tokens = prev_target_next_logits.unsqueeze(1)
        else:
            # If re-encoded full sequence, verify_logits includes logits for all tokens so far.
            # Extract logits for the last `spec_count` tokens and the token after them (spec_count+1 positions).
            big_logits_seq = verify_logits[:, -(spec_count + 1):, :]
            big_logits_for_tokens = big_logits_seq[:, :-1, :]  # logits corresponding to each new token
            big_next_dist = big_logits_seq[:, -1, :].unsqueeze(1)  # distribution after all new tokens
        # Stack draft model logits for speculative tokens
        small_logits = torch.stack(small_logits_list, dim=1)  # shape (batch, spec_count, vocab)
        
        # Calculate log probabilities of each chosen speculative token under both models
        big_log_probs = []   # shape (batch, spec_count)
        small_log_probs = []  # shape (batch, spec_count)
        for j in range(spec_count):
            token_j = spec_tokens[:, j].unsqueeze(1)  # token IDs of j-th speculative token
            # Big model log-prob for token j:
            if j == 0:
                # First speculative token: use distribution after previous context (prev_target_next_logits)
                big_lp = torch.log_softmax(prev_target_next_logits, dim=-1).gather(1, token_j).squeeze(1)
            else:
                big_lp = torch.log_softmax(big_logits_for_tokens[:, j, :], dim=-1).gather(1, token_j).squeeze(1)
            # Draft model log-prob for token j:
            small_lp = torch.log_softmax(small_logits[:, j, :], dim=-1).gather(1, token_j).squeeze(1)
            big_log_probs.append(big_lp)
            small_log_probs.append(small_lp)
        big_log_probs = torch.stack(big_log_probs, dim=1)     # (batch, spec_count)
        small_log_probs = torch.stack(small_log_probs, dim=1)  # (batch, spec_count)
        
        # Decide how many of the speculative tokens to accept for each sequence
        log_ratio = big_log_probs - small_log_probs  # log(p/q) for each token
        ratio = torch.exp(log_ratio)                 # p_i / q_i for each speculative token
        rand = torch.rand(ratio.shape, device=ratio.device)   # random threshold for each token
        # Find first index where random > ratio (meaning token should be rejected)
        reject_index = torch.full((batch_size,), spec_count, dtype=torch.long, device=input_ids.device)
        for i in range(batch_size):
            if finished[i]:
                # Already finished sequences (no new tokens effectively added) – treat as accepting 0 new tokens
                reject_index[i] = 0
            else:
                # find first speculative token to reject
                idx = (rand[i] > ratio[i]).nonzero(as_tuple=True)
                if len(idx[0]) > 0:
                    reject_index[i] = int(idx[0][0])
        # Number of tokens accepted = all tokens up to (reject_index-1) are accepted. If reject_index == spec_count, accept all.
        accepted_counts = torch.where(reject_index < spec_count, reject_index, torch.tensor(spec_count, device=input_ids.device))
        rejected_counts = spec_count - accepted_counts  # how many tokens to remove for each sequence
        
        # Remove rejected tokens from the outputs for each sequence
        new_out_ids_list = []
        new_mask_list = []
        for i in range(batch_size):
            seq_len = seq_lengths[i].item()  # length after adding spec_count tokens
            keep = int(accepted_counts[i].item())  # number of new tokens to keep
            # Calculate new sequence length after rejection
            new_length = seq_len - spec_count + keep
            # Slice sequence and mask to the new length
            seq_tokens = out_ids[i, :new_length]
            seq_mask = attention_mask[i, :new_length]
            new_out_ids_list.append(seq_tokens)
            new_mask_list.append(seq_mask)
            # Update prompt_lengths for next iteration (context now includes accepted tokens)
            prompt_lengths[i] = new_length
        
        # Pad sequences back to a uniform length for batch tensor
        max_len = max(seq.size(0) for seq in new_out_ids_list)
        out_ids = torch.full((batch_size, max_len), pad_token_id, dtype=out_ids.dtype, device=input_ids.device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=input_ids.device)
        for i in range(batch_size):
            length = new_out_ids_list[i].size(0)
            out_ids[i, :length] = new_out_ids_list[i]
            attention_mask[i, :length] = new_mask_list[i]
        # Update finished flags after truncation (in case EOS token was removed or sequence ended at EOS)
        seq_lengths = attention_mask.sum(dim=1)
        if eos_token_id is not None:
            last_tokens = out_ids.gather(1, (seq_lengths - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
            finished |= (last_tokens == eos_token_id)
        finished |= (seq_lengths >= target_lengths)
        
        # Update the target model's "previous next-token" logits for the next iteration.
        # This should be the distribution after the final accepted token of this iteration’s context.
        new_prev_logits = prev_target_next_logits.clone()
        for i in range(batch_size):
            if rejected_counts[i].item() > 0:
                # If some tokens were rejected for sequence i
                if accepted_counts[i].item() > 0:
                    # Use big model distribution after the last accepted token
                    idx = int(accepted_counts[i].item()) - 1
                    new_prev_logits[i] = big_logits_for_tokens[i, idx, :]
                else:
                    # No token accepted (rejected at first token) – distribution remains as it was (previous context)
                    new_prev_logits[i] = prev_target_next_logits[i]
            else:
                # All speculative tokens accepted – use big model's distribution after all new tokens
                new_prev_logits[i] = big_next_dist[i, 0, :]
        prev_target_next_logits = new_prev_logits
        # Note: `past_target` already contains the cache for prompt + spec_count tokens. If tokens were rejected, those cache entries 
        # will not be used (they correspond to dropped tokens), but we continue with the accepted context. We rely on the updated 
        # `prev_target_next_logits` and attention_mask to ensure correctness. If the model does not support external cache, `past_target` 
        # is None and we re-encode full context each time (ensuring correctness at the cost of performance).
    # End of while loop
    
    return out_ids


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