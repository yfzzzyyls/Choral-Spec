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
        self.generation_config = {}
        self.sessions = {}  # session_id -> dict(past, last_logits)

    def LoadModel(self, request, context):
        model_path = request.model_path
        n_positions = request.n_positions or 128
        batch_size = request.batch_size or 1
        tp_degree = request.tp_degree or 1
        amp = request.amp or 'bf16'
        try:
            logger.info(f"[Target] Loading model={model_path}, n_positions={n_positions}, batch_size={batch_size}, tp_degree={tp_degree}, amp={amp}")
            neuron_cfg = NeuronConfig(
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
                neuron_config=neuron_cfg
            )
            start_compile=time.time()
            self.model.to_neuron()
            compile_time=time.time()-start_compile
            logger.info(f"[Target] Model compiled in {compile_time:.2f}s.")
            return inference_pb2.LoadModelResponse(success=True, message="Target model loaded.")
        except Exception as e:
            logger.error(f"[Target] LoadModel error: {e}")
            return inference_pb2.LoadModelResponse(success=False, message=str(e))

    def StartSession(self, request, context):
        sid=request.session_id
        if not sid:
            sid=f"target-{len(self.sessions)+1}"
        input_ids=list(request.input_ids)
        if self.model is None:
            return inference_pb2.StartSessionResponse(session_id=sid, success=False, message="Target model not loaded.")
        try:
            inpt=torch.tensor([input_ids], dtype=torch.int64)
            attn=torch.ones_like(inpt)
            out=self.model(input_ids=inpt, attention_mask=attn, use_cache=True)
            past=out.past_key_values
            logits=out.logits
            last_logits=logits[0,-1,:]
            self.sessions[sid]={
                "past":past,
                "last_logits":last_logits
            }
            logger.info(f"[Target] StartSession sid={sid}, prompt len={len(input_ids)}")
            return inference_pb2.StartSessionResponse(session_id=sid, success=True, message="Target session started.")
        except Exception as e:
            logger.error(f"[Target] StartSession error: {e}")
            return inference_pb2.StartSessionResponse(session_id=sid, success=False, message=str(e))

    def CheckTokenProbability(self, request, context):
        """
        Return the target model's probability for 'token_id', based on last_logits in that session's state.
        We also have request.draft_prob but we just compare.
        """
        sid=request.session_id
        if sid not in self.sessions:
            return inference_pb2.CheckTokenResponse(target_prob=0.0)
        st=self.sessions[sid]
        logits=st["last_logits"]
        if logits is None:
            return inference_pb2.CheckTokenResponse(target_prob=0.0)
        # temperature
        temp=self.generation_config.get("temperature",1.0) if self.generation_config else 1.0
        if temp!=1.0:
            logits=logits/temp
        p=torch.nn.functional.softmax(logits, dim=-1)
        if request.token_id>=p.shape[0]:
            return inference_pb2.CheckTokenResponse(target_prob=0.0)
        prob=float(p[request.token_id].item())
        return inference_pb2.CheckTokenResponse(target_prob=prob)

    def AppendToken(self, request, context):
        """
        Accept the token => forward one step in the model
        """
        sid=request.session_id
        if sid not in self.sessions:
            return inference_pb2.AppendTokenResponse(success=False)
        st=self.sessions[sid]
        inpt=torch.tensor([[request.token_id]], dtype=torch.int64)
        attn=torch.ones_like(inpt)
        try:
            out=self.model(input_ids=inpt, attention_mask=attn,
                           past_key_values=st["past"], use_cache=True)
            st["past"]=out.past_key_values
            st["last_logits"]=out.logits[0,-1,:]
            return inference_pb2.AppendTokenResponse(success=True)
        except Exception as e:
            logger.error(f"[Target] AppendToken error: {e}")
            return inference_pb2.AppendTokenResponse(success=False)

    def GenerateTargetToken(self, request, context):
        """
        If request.draft_distribution is provided, do p-q distribution logic. Otherwise sample from p alone.
        Then forward one step in the model.
        """
        sid=request.session_id
        if sid not in self.sessions:
            return inference_pb2.GenerateTargetResponse(token_id=0)
        st=self.sessions[sid]
        logits=st["last_logits"]
        if logits is None:
            return inference_pb2.GenerateTargetResponse(token_id=0)
        temp=self.generation_config.get("temperature",1.0)
        if temp!=1.0:
            logits=logits/temp
        p=torch.nn.functional.softmax(logits, dim=-1)
        if len(request.draft_distribution)>0:
            # subtract q from p
            qvals=torch.tensor(list(request.draft_distribution), dtype=torch.float32)
            # align
            if qvals.shape[0]<p.shape[0]:
                # pad q
                padlen=p.shape[0]-qvals.shape[0]
                qvals=torch.nn.functional.pad(qvals, (0,padlen))
            elif qvals.shape[0]>p.shape[0]:
                qvals=qvals[:p.shape[0]]
            adj=p-qvals
            adj=torch.clamp(adj, min=0)
            if torch.sum(adj)==0:
                adj=p
            adj=adj/torch.sum(adj)
            # sample
            next_id=int(torch.multinomial(adj,1).item())
        else:
            # sample from p
            next_id=int(torch.multinomial(p,1).item())
        # append
        inpt=torch.tensor([[next_id]], dtype=torch.int64)
        attn=torch.ones_like(inpt)
        try:
            out=self.model(input_ids=inpt, attention_mask=attn, past_key_values=st["past"], use_cache=True)
            st["past"]=out.past_key_values
            st["last_logits"]=out.logits[0,-1,:]
        except Exception as e:
            logger.error(f"[Target] GenerateTargetToken forward error: {e}")
            return inference_pb2.GenerateTargetResponse(token_id=0)
        return inference_pb2.GenerateTargetResponse(token_id=next_id)

def serve(port=50052):
    server=grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_TargetServiceServicer_to_server(TargetServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"TargetWorker gRPC listening on port {port}")
    server.start()
    server.wait_for_termination()

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50052)
    args=parser.parse_args()
    serve(port=args.port)