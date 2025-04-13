import time
import torch
import math
import logging
from concurrent import futures
import grpc
from transformers_neuronx import NeuronAutoModelForCausalLM, NeuronConfig, GenerationConfig

from grpc_comm import inference_pb2
from grpc_comm import inference_pb2_grpc

##############################################
# Set up logger at DEBUG level
##############################################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("TargetWorker")
logger.setLevel(logging.DEBUG)

class TargetServicer(inference_pb2_grpc.TargetServiceServicer):
    def __init__(self):
        logger.info("[TargetServicer.__init__] Constructing the TargetServicer now.")
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
            logger.info(f"[Target] LoadModel called: model={model_path}, n_positions={n_positions}, batch_size={batch_size}, tp_degree={tp_degree}, amp={amp}")
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
            logger.info(f"[Target] Model compiled successfully in {compile_time:.2f}s.")
            return inference_pb2.LoadModelResponse(success=True, message="Target model loaded.")
        except Exception as e:
            logger.error(f"[Target] LoadModel error: {e}", exc_info=True)
            return inference_pb2.LoadModelResponse(success=False, message=str(e))

    def StartSession(self, request, context):
        sid=request.session_id
        if not sid:
            sid=f"target-{len(self.sessions)+1}"
        input_ids=list(request.input_ids)
        if self.model is None:
            logger.error(f"[Target] StartSession: model not loaded yet.")
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
            logger.info(f"[Target] StartSession sid={sid}, prompt len={len(input_ids)} => stored KV cache.")
            return inference_pb2.StartSessionResponse(session_id=sid, success=True, message="Target session started.")
        except Exception as e:
            logger.error(f"[Target] StartSession error: {e}", exc_info=True)
            return inference_pb2.StartSessionResponse(session_id=sid, success=False, message=str(e))

    def CheckTokenProbability(self, request, context):
        sid=request.session_id
        token_id=request.token_id
        if sid not in self.sessions:
            logger.warning(f"[Target] CheckTokenProbability: no session {sid}")
            return inference_pb2.CheckTokenResponse(target_prob=0.0)
        st=self.sessions[sid]
        logits=st["last_logits"]
        if logits is None:
            logger.warning(f"[Target] No last_logits in session {sid}")
            return inference_pb2.CheckTokenResponse(target_prob=0.0)
        temp=self.generation_config.get("temperature",1.0)
        if temp!=1.0:
            logits=logits/temp
        p=torch.nn.functional.softmax(logits, dim=-1)
        if token_id>=p.shape[0]:
            logger.warning(f"[Target] token_id={token_id} out of range for distribution shape={p.shape}. Returning 0.")
            return inference_pb2.CheckTokenResponse(target_prob=0.0)
        prob=float(p[token_id].item())
        logger.info(f"[Target] CheckTokenProbability sid={sid}, token_id={token_id} => {prob:.4f}")
        return inference_pb2.CheckTokenResponse(target_prob=prob)

    def AppendToken(self, request, context):
        sid=request.session_id
        token_id=request.token_id
        if sid not in self.sessions:
            logger.warning(f"[Target] AppendToken: no session {sid}")
            return inference_pb2.AppendTokenResponse(success=False)
        st=self.sessions[sid]
        inpt=torch.tensor([[token_id]], dtype=torch.int64)
        attn=torch.ones_like(inpt)
        try:
            out=self.model(input_ids=inpt, attention_mask=attn,
                           past_key_values=st["past"], use_cache=True)
            st["past"]=out.past_key_values
            st["last_logits"]=out.logits[0,-1,:]
            logger.info(f"[Target] AppendToken sid={sid}, token_id={token_id}")
            return inference_pb2.AppendTokenResponse(success=True)
        except Exception as e:
            logger.error(f"[Target] AppendToken error for sid={sid}: {e}", exc_info=True)
            return inference_pb2.AppendTokenResponse(success=False)

    def GenerateTargetToken(self, request, context):
        sid=request.session_id
        if sid not in self.sessions:
            logger.warning(f"[Target] GenerateTargetToken: no session {sid}")
            return inference_pb2.GenerateTargetResponse(token_id=0)
        st=self.sessions[sid]
        logits=st["last_logits"]
        if logits is None:
            logger.warning(f"[Target] GenerateTargetToken: no logits for session {sid}")
            return inference_pb2.GenerateTargetResponse(token_id=0)
        temp=self.generation_config.get("temperature",1.0)
        if temp!=1.0:
            logits=logits/temp
        p=torch.nn.functional.softmax(logits, dim=-1)
        draft_distribution=list(request.draft_distribution)
        if draft_distribution:
            logger.info(f"[Target] GenerateTargetToken sid={sid}, applying distribution p-q.")
            qvals=torch.tensor(draft_distribution, dtype=torch.float32)
            # align shapes
            if qvals.shape[0]<p.shape[0]:
                padlen=p.shape[0]-qvals.shape[0]
                qvals=torch.nn.functional.pad(qvals, (0,padlen))
            elif qvals.shape[0]>p.shape[0]:
                qvals=qvals[:p.shape[0]]
            adj=p-qvals
            adj=torch.clamp(adj, min=0)
            if torch.sum(adj)==0:
                adj=p
            adj=adj/torch.sum(adj)
            next_id=int(torch.multinomial(adj,1).item())
        else:
            next_id=int(torch.multinomial(p,1).item())
        # append
        inpt=torch.tensor([[next_id]], dtype=torch.int64)
        attn=torch.ones_like(inpt)
        try:
            out=self.model(input_ids=inpt, attention_mask=attn,
                           past_key_values=st["past"], use_cache=True)
            st["past"]=out.past_key_values
            st["last_logits"]=out.logits[0,-1,:]
            logger.info(f"[Target] GenerateTargetToken sid={sid}, next_id={next_id}")
            return inference_pb2.GenerateTargetResponse(token_id=next_id)
        except Exception as e:
            logger.error(f"[Target] GenerateTargetToken forward error: {e}", exc_info=True)
            return inference_pb2.GenerateTargetResponse(token_id=0)


def serve(port=50052):
    logger.info(f"[TargetWorker] about to create gRPC server on port {port}")
    server=grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_TargetServiceServicer_to_server(TargetServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"[TargetWorker] gRPC listening on port {port}")
    server.start()
    logger.info("[TargetWorker] server.start() called, now waiting for termination.")
    server.wait_for_termination()

# Remove if __name__=="__main__" block, replace with direct run:
logger.info("[TargetWorker MAIN] This file is being run, not just imported.")
logger.info("[TargetWorker] The script is actually running, about to parse args...")
import argparse
parser=argparse.ArgumentParser(description="Target model gRPC worker.")
parser.add_argument("--port",type=int,default=50052,help="Port to run on.")
args=parser.parse_args()

logger.info(f"[TargetWorker MAIN] invoked with --port={args.port}")
logger.info("[TargetWorker] The script is actually running, about to serve...")
serve(port=args.port)