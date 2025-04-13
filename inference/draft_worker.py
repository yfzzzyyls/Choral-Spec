import time
import torch
import logging
from concurrent import futures
import grpc
from transformers_neuronx import NeuronAutoModelForCausalLM, NeuronConfig, GenerationConfig, compiler
from transformers_neuronx.llama.model import LlamaForCausalLM

import speculative_pb2
import speculative_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DraftWorker")

class DraftServicer(speculative_pb2_grpc.DraftServiceServicer):
    def __init__(self):
        self.model = None
        self.generation_config = None
        self.sessions = {}  # session_id -> state dict

    def LoadModel(self, request, context):
        """Load and compile the draft model on Neuron cores."""
        model_path = request.model_path
        n_positions = request.n_positions or 1024
        batch_size = request.batch_size or 1
        tp_degree = request.tp_degree or 1
        amp = request.amp or 'bf16'

        try:
            logger.info(f"Loading draft model from {model_path} with batch_size={batch_size}, tp_degree={tp_degree}")
            # Set up Neuron config for draft model
            neuron_config = NeuronConfig(
                # LLaMA uses pre_attention_layer_norm (for Llama2, has_pre_attention_norm=True by default)
                # Adjust if needed for other model types
                padding_side="right",
                # Use standard BSH layout for attention caches (Batch, Seq, Head) for compatibility
                attention_layout="BSH",
                # If using tensor parallel, collectives for all-reduce (BSH as well)
                collectives_layout="BSH",
                # Place input embedding on device to avoid copy overhead
                on_device_embedding=True,
                # Configure on-device generation parameters (sampling) if desired, though we implement sampling in Python
                # We still set them for completeness (not strictly used in manual sampling)
                on_device_generation=GenerationConfig(do_sample=True)
            )
            self.model = NeuronAutoModelForCausalLM.from_pretrained(
                model_path,
                batch_size=batch_size,
                n_positions=n_positions,
                tp_degree=tp_degree,
                amp=amp,
                neuron_config=neuron_config
            )
            compile_start = time.time()
            self.model.to_neuron()  # Compile the model for Neuron
            compile_time = time.time() - compile_start
            logger.info(f"Draft model compiled and loaded in {compile_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load draft model: {e}")
            return speculative_pb2.LoadModelResponse(success=False, message=str(e))
        # Save generation config if provided in on_device_generation for reference
        # In this design, we will implement top_k, top_p, temperature sampling in code, 
        # so we can also store them here for use.
        # (For simplicity, we assume defaults or possibly parse from request if extended.)
        self.generation_config = {
            "top_k": 50,
            "top_p": 0.9,
            "temperature": 1.0,
            "do_sample": True
        }
        return speculative_pb2.LoadModelResponse(success=True, message="Draft model loaded successfully")

    def StartSession(self, request, context):
        """Initialize a decoding session with the given prompt (input IDs)."""
        session_id = request.session_id or f"draft-{len(self.sessions)+1}"
        input_ids = list(request.input_ids)
        if self.model is None:
            return speculative_pb2.StartSessionResponse(session_id=session_id, success=False,
                                                        message="Model not loaded")
        try:
            # Prepare input tensors
            input_tensor = torch.tensor([input_ids], dtype=torch.int64)
            attention_mask = torch.ones_like(input_tensor)
            # Run the model on the prompt to prime the cache
            outputs = self.model(input_ids=input_tensor, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values  # KV cache after processing prompt
            logits = outputs.logits  # shape: (batch=1, seq_len, vocab)
            last_logits = logits[0, -1, :]             # Last distribution after prompt
            # Store session state
            self.sessions[session_id] = {
                "past": past_key_values,
                "last_logits": last_logits,
                "state_cache_stack": []  # will hold intermediate states during draft generation
            }
            logger.info(f"Draft session {session_id} initialized (prompt length = {len(input_ids)} tokens).")
            return speculative_pb2.StartSessionResponse(session_id=session_id, success=True, message="Session started")
        except Exception as e:
            logger.error(f"Error in StartSession for draft model: {e}")
            return speculative_pb2.StartSessionResponse(session_id=session_id, success=False, message=str(e))

    def GenerateDraft(self, request, context):
        """Generate a speculative sequence of draft tokens for one or more sessions."""
        draft_length = request.draft_length
        results = []
        start_time = time.time()
        session_ids = list(request.session_ids)
        if draft_length == 0:
            # Nothing to generate
            for sid in session_ids:
                results.append(speculative_pb2.GenerateDraftResponse.DraftOutput(
                    session_id=sid, tokens=[], probabilities=[]
                ))
            return speculative_pb2.GenerateDraftResponse(outputs=results)
        # Process each session sequentially (could be optimized to batch, see note below)
        for sid in session_ids:
            state = self.sessions.get(sid)
            if state is None or self.model is None:
                # Skip if session or model is not available
                continue
            tokens = []
            probs = []
            # Clear any previous intermediate cache states
            state["state_cache_stack"] = [ None ] * (draft_length + 1)
            # Save initial state (cache after prompt or after last target token integrated)
            # We clone the past to allow rollback without modifying original
            current_past = tuple([tuple([t.clone() for t in layer]) for layer in state["past"]]) if state["past"] else None
            current_last_logits = state["last_logits"].clone() if state["last_logits"] is not None else None
            state["state_cache_stack"][0] = (current_past, current_last_logits)
            try:
                for i in range(1, draft_length+1):
                    # Compute distribution from last logits
                    logits = current_last_logits  # tensor shape (vocab,)
                    # Apply temperature
                    if self.generation_config and "temperature" in self.generation_config:
                        temp = self.generation_config["temperature"]
                    else:
                        temp = 1.0
                    if temp != 1.0:
                        logits = logits / temp
                    # Convert logits to probabilities with softmax
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)
                    # Apply top-p / top-k filtering if configured
                    if self.generation_config:
                        top_p = self.generation_config.get("top_p", 1.0)
                        top_k = self.generation_config.get("top_k", 0)
                    else:
                        top_p = 1.0; top_k = 0
                    # Filter with top_p and top_k
                    if top_p < 1.0:
                        # sort probabilities
                        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        # mask tokens beyond top_p threshold
                        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
                        cutoff_index = min(cutoff_index.item(), sorted_probs.size(-1) - 1)
                        # Compute boolean mask of which indices to keep
                        mask = torch.ones_like(probabilities, dtype=torch.bool)
                        if cutoff_index < probabilities.numel() - 1:
                            cutoff_prob = sorted_probs[cutoff_index]
                            mask[probabilities < cutoff_prob] = False
                        probabilities = probabilities * mask
                        probabilities = probabilities / probabilities.sum()  # renormalize
                    if top_k > 0:
                        # top_k filtering
                        values, indices = torch.topk(probabilities, top_k)
                        min_value = values[-1]
                        mask = probabilities < min_value
                        probabilities = probabilities * (~mask)
                        probabilities = probabilities / probabilities.sum()
                    # Sample next token from filtered distribution
                    if self.generation_config and not self.generation_config.get("do_sample", True):
                        # Greedy: just take argmax if not sampling
                        next_token_id = int(torch.argmax(probabilities))
                        next_token_prob = float(probabilities[next_token_id])
                    else:
                        next_token_id = int(torch.multinomial(probabilities, 1))
                        next_token_prob = float(probabilities[next_token_id])
                    tokens.append(next_token_id)
                    probs.append(next_token_prob)
                    # If not the last token, advance the model state with this token
                    if i < draft_length:
                        input_tensor = torch.tensor([[next_token_id]], dtype=torch.int64)
                        attention_mask = torch.ones_like(input_tensor)
                        # Use current_past to generate next logits
                        out = self.model(input_ids=input_tensor, attention_mask=attention_mask,
                                         past_key_values=current_past, use_cache=True)
                        current_past = out.past_key_values
                        # Update last logits for distribution after this token
                        current_last_logits = out.logits[0, -1, :]
                        # Save the state (past and last_logits) after generating i tokens
                        state["state_cache_stack"][i] = (tuple([tuple([t.clone() for t in layer]) for layer in current_past]),
                                                         current_last_logits.clone())
                # After generating full draft_length tokens, store final state
                state["past"] = current_past
                state["last_logits"] = current_last_logits
                # Save final state as well
                state["state_cache_stack"][draft_length] = (tuple([tuple([t.clone() for t in layer]) for layer in current_past]),
                                                           current_last_logits.clone())
            except Exception as e:
                logger.error(f"Error during draft generation for session {sid}: {e}")
                # If any error, break out (we can return partial tokens or handle error differently)
            # Append result for this session
            results.append(speculative_pb2.GenerateDraftResponse.DraftOutput(
                session_id=sid,
                tokens=tokens,
                probabilities=probs
            ))
            logger.info(f"Draft model generated {len(tokens)} tokens for session {sid}: {tokens}")
        elapsed = time.time() - start_time
        logger.info(f"GenerateDraft completed for {len(session_ids)} session(s) in {elapsed:.3f}s")
        return speculative_pb2.GenerateDraftResponse(outputs=results)

    def UpdateDraftContext(self, request, context):
        """Roll back the draft model state and integrate the target model's new token."""
        session_id = request.session_id
        accepted_count = request.accepted_count
        new_token = request.new_token
        state = self.sessions.get(session_id)
        if state is None:
            return speculative_pb2.UpdateDraftContextResponse(success=False, message="Session not found")
        try:
            # Roll back to state after 'accepted_count' tokens
            if "state_cache_stack" in state and state["state_cache_stack"]:
                past_state, logits_state = state["state_cache_stack"][accepted_count]
                if past_state is not None:
                    # Deep copy to avoid aliasing
                    past_state = tuple([tuple([t.clone() for t in layer]) for layer in past_state])
                state["past"] = past_state
                state["last_logits"] = logits_state.clone() if logits_state is not None else None
            else:
                # If no stack stored (e.g. accepted_count=0 and no tokens generated), state remains as initial
                pass
            # Now append the new target token to the draft model's context
            input_tensor = torch.tensor([[new_token]], dtype=torch.int64)
            attention_mask = torch.ones_like(input_tensor)
            out = self.model(input_ids=input_tensor, attention_mask=attention_mask,
                             past_key_values=state["past"], use_cache=True)
            # Update session state with new past and last logits (distribution after the new token)
            state["past"] = out.past_key_values
            state["last_logits"] = out.logits[0, -1, :]
            # Clear intermediate stack as it's no longer needed after context update
            state["state_cache_stack"] = []
            logger.info(f"Draft session {session_id} rolled back to {accepted_count} accepted tokens and integrated target token {new_token}.")
            return speculative_pb2.UpdateDraftContextResponse(success=True, message="Draft context updated")
        except Exception as e:
            logger.error(f"Error in UpdateDraftContext for session {session_id}: {e}")
            return speculative_pb2.UpdateDraftContextResponse(success=False, message=str(e))

def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    speculative_pb2_grpc.add_DraftServiceServicer_to_server(DraftServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"Starting DraftService gRPC server on port {port}...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Draft model gRPC worker")
    parser.add_argument("--port", type=int, default=50051, help="Port to run the DraftService server on")
    args = parser.parse_args()
    serve(port=args.port)