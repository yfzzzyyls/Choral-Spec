# Replace this with the path where you downloaded and saved the model files.
MODEL_PATH="/home/ubuntu/models/llama-3.1-8b/"
# This is where the compiled model will be saved. The same path
# should be used when launching vLLM server for inference.
COMPILED_MODEL_PATH="/home/ubuntu/Choral-Spec/llama-3.1-8b-compiled"

NUM_CORES=2
TP_DEGREE=2
LNC=1

export NEURON_RT_VIRTUAL_CORE_SIZE=$LNC
export NEURON_RT_NUM_CORES=$((NUM_CORES/NEURON_RT_VIRTUAL_CORE_SIZE))
export NEURON_RT_EXEC_TIMEOUT=600
export XLA_DENSE_GATHER_FACTOR=0
export NEURON_RT_INSPECT_ENABLE=0

inference_demo \
    --model-type llama \
    --task-type causal-lm \
        run \
        --model-path $MODEL_PATH \
        --compiled-model-path $COMPILED_MODEL_PATH \
        --torch-dtype bfloat16 \
        --start_rank_id 0 \
        --local_ranks_size $TP_DEGREE \
        --tp-degree $TP_DEGREE \
        --batch-size 1 \
        --max-context-length 128 \
        --seq-len 128 \
        --on-device-sampling \
        --top-k 1 \
        --do-sample \
        --fused-qkv \
        --sequence-parallel-enabled \
        --qkv-kernel-enabled \
        --attn-kernel-enabled \
        --mlp-kernel-enabled \
        --cc-pipeline-tiling-factor 1 \
        --pad-token-id 2 \
        --enable-bucketing \
        --context-encoding-buckets 48 96 \
            --token-generation-buckets 48 96 \
        --prompt "What is annapurna labs?" 2>&1 | tee log