import time
import torch
import logging
from concurrent import futures
import grpc
from transformers_neuronx import NeuronAutoModelForCausalLM, NeuronConfig, GenerationConfig
# Make sure we have a __init__.py in grpc_comm
from grpc_comm import inference_pb2
from grpc_comm import inference_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DraftWorker")

class DraftServicer(inference_pb2_grpc.DraftServiceServicer):
    def __init__(self):
        self.model = None
        self.generation_config = {}
        self.sessions = {}  # session_id -> dict(past, last_logits, state_cache_stack)

    def LoadModel(self, request, context):
        model_path = request.model_path
        n_positions = request.n_positions or 128
        batch_size = request.batch_size or 1
        tp_degree = request.tp_degree or 1
        amp = request.amp or 'bf16'
        try:
            logger.info(f"[Draft] Loading model={model_path}, n_positions={n_positions}, batch_size={batch_size}, tp_degree={tp_degree}, amp={amp}")
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
            start_compile = time.time()
            self.model.to_neuron()
            compile_time = time.time()-start_compile
            logger.info(f"[Draft] Model compiled in {compile_time:.2f}s.")
            return inference_pb2.LoadModelResponse(success=True, message="Draft model loaded.")
        except Exception as e:
            logger.error(f"[Draft] LoadModel error: {e}")
            return inference_pb2.LoadModelResponse(success=False, message=str(e))

    def StartSession(self, request, context):
        sid = request.session_id
        if not sid:
            sid=f"draft-{len(self.sessions)+1}"
        input_ids = list(request.input_ids)
        if self.model is None:
            return inference_pb2.StartSessionResponse(
                session_id=sid, success=False, message="Draft model not loaded."
            )
        try:
            input_tensor = torch.tensor([input_ids], dtype=torch.int64)
            attn = torch.ones_like(input_tensor)
            outputs = self.model(input_ids=input_tensor, attention_mask=attn, use_cache=True)
            past = outputs.past_key_values
            logits = outputs.logits
            last_logits = logits[0, -1, :]
            self.sessions[sid] = {
                "past": past,
                "last_logits": last_logits,
                "state_cache_stack": []
            }
            logger.info(f"[Draft] StartSession sid={sid}, prompt len={len(input_ids)}")
            return inference_pb2.StartSessionResponse(session_id=sid, success=True, message="Draft session started.")
        except Exception as e:
            logger.error(f"[Draft] StartSession error: {e}")
            return inference_pb2.StartSessionResponse(session_id=sid, success=False, message=str(e))

    def GenerateDraft(self, request, context):
        """
        Generate up to draft_length tokens for each session in request.session_ids
        We'll do a naive approach: sequentially process each session (we could do it in a batch, but more complex).
        We store intermediate states in state_cache_stack for rollback usage.
        """
        draft_len = request.draft_length
        session_ids = list(request.session_ids)
        outputs=[]
        if draft_len<=0:
            for sid in session_ids:
                outputs.append(inference_pb2.GenerateDraftResponse.DraftOutput(
                    session_id=sid, tokens=[], probabilities=[]
                ))
            return inference_pb2.GenerateDraftResponse(outputs=outputs)

        for sid in session_ids:
            if sid not in self.sessions:
                # skip or respond with empty?
                outputs.append(inference_pb2.GenerateDraftResponse.DraftOutput(
                    session_id=sid,
                    tokens=[],
                    probabilities=[]
                ))
                continue
            state = self.sessions[sid]
            tokens=[]
            probs=[]
            # Clear + re-init stack with draft_len+1 slots
            state["state_cache_stack"] = [None]*(draft_len+1)
            curr_past = state["past"]
            curr_logits = state["last_logits"]
            # store slot0 => initial
            # we clone everything to allow safe rollback
            if curr_past is not None:
                cpast = tuple([tuple([t.clone() for t in layer]) for layer in curr_past])
            else:
                cpast=None
            if curr_logits is not None:
                clogits=curr_logits.clone()
            else:
                clogits=None
            state["state_cache_stack"][0]=(cpast, clogits)

            try:
                for i in range(1, draft_len+1):
                    # sample from curr_logits
                    logits=curr_logits
                    # (Apply a fixed top_p=0.9, temperature=1.0, or read from self.generation_config)
                    temp = self.generation_config.get("temperature", 1.0) if self.generation_config else 1.0
                    if temp!=1.0:
                        logits=logits/temp
                    p = torch.nn.functional.softmax(logits, dim=-1)
                    # top_p=0.9
                    top_p = self.generation_config.get("top_p", 0.9) if self.generation_config else 0.9
                    if top_p<1.0:
                        sorted_p,sorted_idx=torch.sort(p,descending=True)
                        csum=torch.cumsum(sorted_p,dim=0)
                        cutoff=(csum>=top_p).nonzero(as_tuple=True)
                        if len(cutoff[0])>0:
                            ci=cutoff[0][0].item()
                        else:
                            ci=len(sorted_p)-1
                        keep_idx=sorted_idx[:ci+1]
                        mask=torch.zeros_like(p, dtype=torch.bool)
                        mask[keep_idx]=True
                        p=p*mask
                        p=p/p.sum()
                    # sample
                    next_id=int(torch.multinomial(p,1).item())
                    next_prob=float(p[next_id])
                    tokens.append(next_id)
                    probs.append(next_prob)
                    if i<draft_len:
                        # feed next_id
                        inpt=torch.tensor([[next_id]],dtype=torch.int64)
                        attn=torch.ones_like(inpt)
                        out=self.model(input_ids=inpt, attention_mask=attn, past_key_values=curr_past, use_cache=True)
                        curr_past=out.past_key_values
                        curr_logits=out.logits[0,-1,:]
                        # store snapshot
                        if curr_past is not None:
                            cpy_past=tuple([tuple([t.clone() for t in layer]) for layer in curr_past])
                        else:
                            cpy_past=None
                        cpy_logits=curr_logits.clone()
                        state["state_cache_stack"][i]=(cpy_past, cpy_logits)
                # done generating
                # store final
                state["past"]=curr_past
                state["last_logits"]=curr_logits
                if curr_past is not None:
                    cpy_past=tuple([tuple([t.clone() for t in layer]) for layer in curr_past])
                else:
                    cpy_past=None
                cpy_logits=curr_logits.clone() if curr_logits is not None else None
                state["state_cache_stack"][draft_len]=(cpy_past, cpy_logits)

                outputs.append(
                    inference_pb2.GenerateDraftResponse.DraftOutput(
                        session_id=sid,
                        tokens=tokens,
                        probabilities=probs
                    )
                )
                logger.info(f"[Draft] generate sid={sid} => {tokens}")
            except Exception as e:
                logger.error(f"GenerateDraft error on sid={sid}: {e}")
                # Return partial if some tokens
                outputs.append(
                    inference_pb2.GenerateDraftResponse.DraftOutput(
                        session_id=sid,
                        tokens=tokens,
                        probabilities=probs
                    )
                )
        return inference_pb2.GenerateDraftResponse(outputs=outputs)

    def UpdateDraftContext(self, request, context):
        """
        Rollback and/or integrate the next target token.
        accepted_count => how many of the previously generated tokens we keep
        new_token => the forced token from the target
        """
        sid=request.session_id
        accept_count=request.accepted_count
        forced_tok=request.new_token
        if sid not in self.sessions:
            return inference_pb2.UpdateDraftContextResponse(success=False, message="No session found")
        st=self.sessions[sid]
        # rollback
        scs=st["state_cache_stack"]
        if accept_count<len(scs):
            cpy=scs[accept_count]
            if cpy is not None:
                st["past"], st["last_logits"]=cpy
            else:
                logger.warning(f"[Draft] no stored snapshot for accept_count={accept_count}")
        # if forced_tok != 0 => feed it into the model
        if forced_tok!=0:
            try:
                inpt=torch.tensor([[forced_tok]],dtype=torch.int64)
                attn=torch.ones_like(inpt)
                out=self.model(input_ids=inpt, attention_mask=attn,
                               past_key_values=st["past"], use_cache=True)
                st["past"]=out.past_key_values
                st["last_logits"]=out.logits[0,-1,:]
            except Exception as e:
                logger.error(f"[Draft] error appending forced token {forced_tok} in UpdateDraftContext: {e}")
                return inference_pb2.UpdateDraftContextResponse(success=False, message=str(e))
        # done
        st["state_cache_stack"]=[]
        return inference_pb2.UpdateDraftContextResponse(success=True, message="Updated draft context")

def serve(port=50051):
    server=grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_DraftServiceServicer_to_server(DraftServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"DraftWorker gRPC listening on port {port}")
    server.start()
    server.wait_for_termination()

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=50051)
    args=parser.parse_args()
    serve(port=args.port)