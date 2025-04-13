import time
import torch
import math
import logging
from concurrent import futures
import grpc
from transformers_neuronx import NeuronAutoModelForCausalLM, NeuronConfig, GenerationConfig

from grpc_comm import inference_pb2
from grpc_comm import inference_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TargetWorker")

class TargetServicer(inference_pb2_grpc.TargetServiceServicer):
    def __init__(self):
        self.model = None
        self.generation_config = None
        self.sessions = {}  # session_id -> state dict

    def LoadModel(self, request, context):
        """Load and compile the target model on Neuron cores."""
        model_path = request.model_path
        n_positions = request.n_positions or 1024
        batch_size = request.batch_size or 1
        tp_degree = request.tp_degree or 1
        amp = request.amp or 'bf16'
        try:
            logger.info(f"Loading target model from {model_path} with batch_size={batch_size}, tp_degree={tp_degree}")
            neuron_config = NeuronConfig(
                padding_side="right",
                attention_layout="BSH",
                collectives_layout="BSH",
                on_device_embedding=True,
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
            self.model.to_neuron()  # Compile model on Neuron cores
            compile_time = time.time() - compile_start
            logger.info(f"Target model compiled and loaded in {compile_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load target model: {e}")
            return inference_pb2.LoadModelResponse(success=False, message=str(e))
        # Save generation config (top_k, top_p, etc.), assume defaults or parse if extended
        self.generation_config = {
            "top_k": 50,
            "top_p": 0.9,
            "temperature": 1.0,
            "do_sample": True
        }
        return inference_pb2.LoadModelResponse(success=True, message="Target model loaded successfully")

    def StartSession(self, request, context):
        """Initialize a target model decoding session with a prompt."""
        session_id = request.session_id or f"target-{len(self.sessions)+1}"
        input_ids = list(request.input_ids)
        if self.model is None:
            return inference_pb2.StartSessionResponse(session_id=session_id, success=False,
                                                        message="Model not loaded")
        try:
            input_tensor = torch.tensor([input_ids], dtype=torch.int64)
            attention_mask = torch.ones_like(input_tensor)
            outputs = self.model(input_ids=input_tensor, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits
            last_logits = logits[0, -1, :]
            # Store session state
            self.sessions[session_id] = {
                "past": past_key_values,
                "last_logits": last_logits
            }
            logger.info(f"Target session {session_id} initialized (prompt length = {len(input_ids)} tokens).")
            return inference_pb2.StartSessionResponse(session_id=session_id, success=True, message="Session started")
        except Exception as e:
            logger.error(f"Error in StartSession for target model: {e}")
            return inference_pb2.StartSessionResponse(session_id=session_id, success=False, message=str(e))

    def CheckTokenProbability(self, request, context):
        """Get the target model probability of a given token at the current context."""
        session_id = request.session_id
        token_id = request.token_id
        draft_prob = request.draft_prob
        state = self.sessions.get(session_id)
        if state is None:
            return inference_pb2.CheckTokenResponse(target_prob=0.0)
        # We assume the target model's last_logits contains the distribution for the next token at this context
        logits = state["last_logits"]
        # Apply same temperature as used in generation config (for consistency if temperature != 1)
        temp = 1.0
        if self.generation_config:
            temp = self.generation_config.get("temperature", 1.0)
        if temp != 1.0:
            logits = logits / temp
        # Compute softmax normalization factor (log-sum-exp for stability)
        # We need the probability of token_id under the target distribution
        # For numerical stability, subtract max logit
        logits_np = logits.cpu().numpy()
        max_logit = float(logits_np.max())
        exp_logits = torch.exp(logits - max_logit)
        exp_sum = float(exp_logits.sum().item())
        token_exp = float(exp_logits[token_id].item())
        target_prob = token_exp / exp_sum
        # Alternatively, we could compute directly via softmax and index:
        # target_prob = float(torch.nn.functional.softmax(logits, dim=-1)[token_id].item())
        return inference_pb2.CheckTokenResponse(target_prob=target_prob)

    def AppendToken(self, request, context):
        """Append an accepted token to the target context (advance the target model state by one token)."""
        session_id = request.session_id
        token_id = request.token_id
        state = self.sessions.get(session_id)
        if state is None:
            return inference_pb2.AppendTokenResponse(success=False)
        try:
            input_tensor = torch.tensor([[token_id]], dtype=torch.int64)
            attention_mask = torch.ones_like(input_tensor)
            out = self.model(input_ids=input_tensor, attention_mask=attention_mask,
                             past_key_values=state["past"], use_cache=True)
            # Update session state: new past and last_logits (distribution after this token)
            state["past"] = out.past_key_values
            state["last_logits"] = out.logits[0, -1, :]
            return inference_pb2.AppendTokenResponse(success=True)
        except Exception as e:
            logger.error(f"AppendToken error for session {session_id}: {e}")
            return inference_pb2.AppendTokenResponse(success=False)

    def GenerateTargetToken(self, request, context):
        """Generate the next token from the target model's current distribution.
           If draft_distribution is provided, perform distribution adjustment (rollback case)."""
        session_id = request.session_id
        state = self.sessions.get(session_id)
        if state is None:
            return inference_pb2.GenerateTargetResponse(token_id=0)
        logits = state["last_logits"]
        # Apply temperature if any
        temp = 1.0
        if self.generation_config:
            temp = self.generation_config.get("temperature", 1.0)
        if temp != 1.0:
            logits = logits / temp
        # Compute target probabilities
        target_probs = torch.nn.functional.softmax(logits, dim=-1)
        if len(request.draft_distribution) > 0:
            # Draft distribution provided: subtract it from target distribution (p - q)
            # Construct tensor for draft distribution
            q = torch.tensor(request.draft_distribution, dtype=torch.float32)
            # Align length (should be full vocab)
            if q.shape != target_probs.shape:
                # If shapes differ (perhaps due to different vocab?), pad or truncate as needed
                min_len = min(q.shape[0], target_probs.shape[0])
                q = q[:min_len]
                q = torch.nn.functional.pad(q, (0, target_probs.shape[0] - min_len))
            p = target_probs
            adjusted = p - q
            # Clip negatives to zero
            adjusted = torch.clamp(adjusted, min=0.0)
            # If all values are zero (unlikely unless q > p on all), fallback to p
            if torch.sum(adjusted) <= 0:
                adjusted = p
            # Renormalize to sum to 1
            adjusted = adjusted / torch.sum(adjusted)
            # Sample token from adjusted distribution
            if self.generation_config and not self.generation_config.get("do_sample", True):
                next_token_id = int(torch.argmax(adjusted))
            else:
                next_token_id = int(torch.multinomial(adjusted, 1))
            next_token_prob = float(adjusted[next_token_id].item())
            logger.info(f"Target model adjusted distribution sampling: token={next_token_id}, prob={next_token_prob:.4f}")
        else:
            # No draft distribution, just sample from target's own distribution
            p = target_probs
            if self.generation_config and not self.generation_config.get("do_sample", True):
                next_token_id = int(torch.argmax(p))
            else:
                # Apply top-p/top-k filtering to target distribution as well (to mirror sampling constraints)
                probs = p.clone()
                top_p = self.generation_config.get("top_p", 1.0) if self.generation_config else 1.0
                top_k = self.generation_config.get("top_k", 0) if self.generation_config else 0
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=0)
                    cutoff_index = torch.searchsorted(cum_probs, top_p)
                    cutoff_index = min(int(cutoff_index), sorted_probs.shape[0]-1)
                    cutoff_prob = sorted_probs[cutoff_index]
                    probs[probs < cutoff_prob] = 0.0
                    probs = probs / probs.sum()
                if top_k > 0:
                    if top_k < probs.numel():
                        threshold_prob = torch.topk(probs, top_k)[0][-1]
                        probs[probs < threshold_prob] = 0.0
                        probs = probs / probs.sum()
                next_token_id = int(torch.multinomial(probs, 1))
            # (We could log target's chosen token probability if needed)
        # Append this token to the target model's context (advance one step)
        input_tensor = torch.tensor([[next_token_id]], dtype=torch.int64)
        attention_mask = torch.ones_like(input_tensor)
        out = self.model(input_ids=input_tensor, attention_mask=attention_mask,
                         past_key_values=state["past"], use_cache=True)
        state["past"] = out.past_key_values
        state["last_logits"] = out.logits[0, -1, :]
        return inference_pb2.GenerateTargetResponse(token_id=next_token_id)

def serve(port=50052):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_TargetServiceServicer_to_server(TargetServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"Starting TargetService gRPC server on port {port}...")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Target model gRPC worker")
    parser.add_argument("--port", type=int, default=50052, help="Port to run the TargetService server on")
    args = parser.parse_args()
    serve(port=args.port)