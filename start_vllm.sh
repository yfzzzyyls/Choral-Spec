export NEURON_RT_INSPECT_ENABLE=0
export NEURON_RT_VIRTUAL_CORE_SIZE=1

# These should be the same paths used when compiling the model.
MODEL_PATH="/home/ubuntu/models/llama-3.1-8b/"
COMPILED_MODEL_PATH="/home/ubuntu/Choral-Spec/llama-3.1-8b-compiled/"

export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_COMPILED_ARTIFACTS=$COMPILED_MODEL_PATH
VLLM_RPC_TIMEOUT=100000 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --max-num-seqs 1 \
    --max-model-len 128 \
    --tensor-parallel-size 2 \
    --device neuron \
    --use-v2-block-manager \
    --override-neuron-config "{\"on_device_sampling_config\": {\"do_sample\": true}}" \
    --port 8000 &
PID=$!
echo "vLLM server started with PID $PID"