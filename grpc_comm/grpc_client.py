# ==========================
# 2) grpc_client.py
# ==========================
import grpc
from . import inference_pb2
from . import inference_pb2_grpc


def create_stub(target_address):
    channel = grpc.insecure_channel(target_address)
    stub = inference_pb2_grpc.SpeculativeServiceStub(channel)
    return stub

# -----------------------------------------
# BATCH-ORIENTED CLIENT CALLS
# -----------------------------------------

def verify_batch_tokens(stub, sequences):
    # sequences is a list of (session_id, [draft_tokens])
    # build the request
    seq_msgs = []
    for s in sequences:
        session_id, draft_toks = s
        seq_msgs.append(
            inference_pb2.DraftSequence(
                session_id=session_id,
                draft_tokens=draft_toks
            )
        )
    request = inference_pb2.VerifyBatchRequest(sequences=seq_msgs)
    response = stub.VerifyBatchTokens(request)
    # returns a list of results
    results = []
    for r in response.results:
        results.append({
            'session_id': r.session_id,
            'tokens_accepted': r.tokens_accepted,
            'target_token': r.target_token,
            'finished': r.finished,
        })
    return results


def finalize_batch_tokens(stub, sequences):
    # sequences is a list of (session_id, [accepted_tokens])
    seq_msgs = []
    for s in sequences:
        session_id, tok_list = s
        seq_msgs.append(
            inference_pb2.FinalizeSequence(
                session_id=session_id,
                tokens=tok_list
            )
        )
    request = inference_pb2.FinalizeBatchRequest(sequences=seq_msgs)
    response = stub.FinalizeBatchTokens(request)
    results = []
    for r in response.results:
        results.append({
            'session_id': r.session_id,
            'finished': r.finished,
        })
    return results

# -----------------------------------------
# SINGLE-SEQUENCE CLIENT CALLS (existing)
# -----------------------------------------

def verify_draft_tokens(stub, draft_tokens, session_id=0):
    request = inference_pb2.VerifyRequest(
        session_id=session_id,
        draft_tokens=draft_tokens
    )
    response = stub.VerifyDraftTokens(request)
    target_probs = list(response.target_probs)
    finished = response.finished
    return target_probs, finished


def finalize_tokens(stub, accepted_count, draft_chunk_size, session_id=0):
    request = inference_pb2.FinalizeRequest(
        session_id=session_id,
        accepted_count=accepted_count,
        draft_chunk_size=draft_chunk_size
    )
    response = stub.FinalizeTokens(request)
    final_token_id = response.final_token
    finished = response.finished
    return final_token_id, finished