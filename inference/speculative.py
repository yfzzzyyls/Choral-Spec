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
    Performs real batch speculative decoding using a local `draft_model` and a remote target
    via `stub` for verification. This version matches the call signature in `draft_worker.py`.

    :param draft_model: The local draft model (PyTorch module compiled for Neuron). Must have `.device`.
    :param tokenizer: Hugging Face tokenizer.
    :param stub: A SpeculativeServiceStub to the remote target server (for verify/finalize calls).
    :param input_ids_batch: A [batch_size, seq_len] tensor of token IDs.
    :param attention_mask_batch: A [batch_size, seq_len] tensor of 1/0 indicating real/pad tokens.
    :param max_new_tokens: Maximum tokens to generate for each sequence.
    :param gamma: Number of draft tokens to propose before verifying.
    :param profile: If True, measure performance.
    :param top_p: Nucleus sampling threshold for the draft.
    :param temperature: Sampling temperature for the draft.
    :param session_ids: A list of session IDs, one per sequence, for the target.

    :return: (final_texts, perf_stats) where final_texts is a list of decoded strings per sequence.
    """

    import time
    import random

    logger.info(f"Running speculative_decode_batch with batch_size={input_ids_batch.size(0)}")

    # Determine device from draft_model
    device = getattr(draft_model, 'device', torch.device('cpu'))
    # Move inputs to device
    input_ids_batch = input_ids_batch.to(device)
    attention_mask_batch = attention_mask_batch.to(device)
    batch_size = input_ids_batch.size(0)

    if session_ids is None:
        # create a random session ID for each sequence
        from uuid import uuid4
        session_ids = [(int(uuid4()) & 0xFFFFFFFF) for _ in range(batch_size)]

    # We'll track output tokens for each sequence in a python list
    output_tokens = [[] for _ in range(batch_size)]
    # fill output_tokens with the existing prompt tokens from input_ids_batch (excluding pads)
    prompt_lengths = attention_mask_batch.sum(dim=1).tolist()
    for i in range(batch_size):
        plen = prompt_lengths[i]
        init_tokens = input_ids_batch[i, :plen].tolist()
        output_tokens[i].extend(init_tokens)

    finished_mask = [False]*batch_size
    new_tokens_count = [0]*batch_size  # how many new tokens each sequence generated

    start_time = time.time() if profile else None

    accepted_tokens_total = [0]*batch_size
    forced_tokens_total = [0]*batch_size

    def all_finished():
        return all(finished_mask)

    # We'll do repeated cycles: draft proposes gamma tokens, stub verifies, we finalize
    while not all_finished():
        # gather active sequences
        active_indices = [i for i, fin in enumerate(finished_mask) if not fin]
        if not active_indices:
            break

        # propose up to gamma tokens for each active sequence
        proposed_tokens = {i: [] for i in active_indices}

        # We do token-by-token generation for up to gamma steps
        for step in range(gamma):
            if not active_indices:
                break
            # build a sub-batch for the active sequences: each provides its last token as input
            sub_inputs = []
            for seq_idx in active_indices:
                if len(proposed_tokens[seq_idx]) == step:
                    # if step=0, we feed the last prompt token
                    # else, feed the token we just generated
                    if step == 0:
                        # use the last prompt token
                        plen = len(output_tokens[seq_idx])
                        last_token_id = output_tokens[seq_idx][plen-1]
                    else:
                        last_token_id = proposed_tokens[seq_idx][-1]
                    sub_inputs.append(last_token_id)
            sub_batch_size = len(sub_inputs)
            if sub_batch_size == 0:
                break
            # create input_ids of shape [sub_batch_size, 1]
            sub_tokens_tensor = torch.tensor(sub_inputs, dtype=torch.long, device=device).unsqueeze(1)

            # forward pass draft model with sub_batch_size
            # we do not handle a past cache for now (or we do partial?), let's keep it simple
            outputs = draft_model(input_ids=sub_tokens_tensor)
            if isinstance(outputs, tuple) and len(outputs) >= 1:
                # Some models return (logits, ...)
                logits = outputs[0]
            else:
                # If a single tensor
                logits = outputs

            # Handle either [sub_batch_size, 1, vocab] or [sub_batch_size, vocab]
            if logits.dim() == 3:
                # e.g., shape [sub_batch_size, 1, vocab]
                logits = logits[:, -1, :]  # becomes [sub_batch_size, vocab]
            elif logits.dim() == 2:
                # already [sub_batch_size, vocab], do nothing
                pass
            else:
                # unexpected shape
                raise ValueError(f"Draft model returned logits of shape {logits.shape}, expected 2D or 3D.")

            # Now logits is [sub_batch_size, vocab]

            # sample next token for each row
            probs = torch.softmax(logits / temperature, dim=-1)
            # top_p filtering if needed
            next_tokens = []
            for b_i in range(sub_batch_size):
                row_probs = probs[b_i]
                # do nucleus top_p
                sorted_probs, sorted_indices = torch.sort(row_probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                cutoff = (cumulative >= top_p).nonzero()
                if len(cutoff) > 0 and len(cutoff[0]) > 0:
                    cutoff_idx = cutoff[0][0].item()
                else:
                    cutoff_idx = len(sorted_probs)-1
                keep_probs = sorted_probs[:cutoff_idx+1]
                keep_indices = sorted_indices[:cutoff_idx+1]
                keep_sum = keep_probs.sum()
                if keep_sum < 1e-9:
                    keep_probs = torch.ones_like(keep_probs)
                    keep_sum = keep_probs.sum()
                keep_probs = keep_probs / keep_sum
                choice_idx = torch.multinomial(keep_probs, 1).item()
                token_id = keep_indices[choice_idx].item()
                next_tokens.append(token_id)

            # assign these tokens to the correct sequences
            b_i = 0
            newly_finished = []
            for seq_idx in active_indices:
                if len(proposed_tokens[seq_idx]) == step:
                    t_id = next_tokens[b_i]
                    proposed_tokens[seq_idx].append(t_id)
                    b_i += 1

            # check EOS among them
            eos_hits = []
            for seq_idx in active_indices:
                last_t = proposed_tokens[seq_idx][-1]
                # if is EOS, mark finished
                if tokenizer.eos_token_id is not None and last_t == tokenizer.eos_token_id:
                    eos_hits.append(seq_idx)
            if eos_hits:
                for e in eos_hits:
                    active_indices.remove(e)
        # end gamma block

        # now we verify using stub.VerifyBatchTokens
        verify_list = []
        for i in range(batch_size):
            if not finished_mask[i] and proposed_tokens.get(i) and len(proposed_tokens[i])>0:
                verify_list.append((session_ids[i], proposed_tokens[i]))
        if not verify_list:
            break
        verify_results = grpc_client.verify_batch_tokens(stub, verify_list)
        # parse the results
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
            if i not in proposed_tokens or not proposed_tokens[i]:
                continue
            if finished_mask[i]:
                continue
            sid = session_ids[i]
            if sid not in result_map:
                continue
            rr = result_map[sid]
            accepted_count = rr['tokens_accepted']
            mismatch_token = rr['target_token']
            seq_finished = rr['finished']

            block_toks = proposed_tokens[i]
            block_len = len(block_toks)
            accepted_tokens = block_toks[:accepted_count]

            # add accepted tokens to output
            output_tokens[i].extend(accepted_tokens)
            accepted_tokens_total[i] += len(accepted_tokens)
            new_tokens_count[i] += len(accepted_tokens)

            if mismatch_token != 0:
                output_tokens[i].append(mismatch_token)
                forced_tokens_total[i] += 1
                new_tokens_count[i] += 1

            if seq_finished or (tokenizer.eos_token_id is not None and len(output_tokens[i])>0 and output_tokens[i][-1] == tokenizer.eos_token_id):
                finished_mask[i] = True

            # finalize tokens on target side
            finalize_list = accepted_tokens[:]
            if mismatch_token != 0:
                finalize_list += [mismatch_token]
            if finalize_list:
                finalize_data[i] = finalize_list

        # call finalize
        fseq_list = []
        for i in finalize_data:
            sid = session_ids[i]
            fseq_list.append((sid, finalize_data[i]))
        if fseq_list:
            fresp = grpc_client.finalize_batch_tokens(stub, fseq_list)
            for r in fresp:
                sid = r['session_id']
                fin = r['finished']
                i2 = session_ids.index(sid)
                if fin:
                    finished_mask[i2] = True

        # check max_new_tokens
        for i in range(batch_size):
            if not finished_mask[i] and new_tokens_count[i] >= max_new_tokens:
                finished_mask[i] = True

    end_time = time.time() if profile else None
    perf_stats = {}
    if profile:
        total_elapsed = end_time - start_time
        total_accepted = sum(accepted_tokens_total)
        total_forced = sum(forced_tokens_total)
        total_tokens = total_accepted + total_forced
        throughput = total_tokens / total_elapsed if total_elapsed>0 else 0.0
        perf_stats["total_time"] = total_elapsed
        perf_stats["tokens_generated"] = total_tokens
        perf_stats["throughput"] = throughput
        perf_stats["avg_token_time"] = total_elapsed / total_tokens if total_tokens>0 else 0.0
        if total_tokens>0:
            match_rate = total_accepted / total_tokens
            perf_stats["token_match_rate"] = match_rate
            logger.info(
                f"[BATCH] Speculative decoding match rate: {match_rate:.2%} "
                f"(Draft accepted: {total_accepted}, Target generated: {total_forced})"
            )

    # decode final text
    final_texts = []
    for i in range(batch_size):
        text = tokenizer.decode(output_tokens[i], skip_special_tokens=True)
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