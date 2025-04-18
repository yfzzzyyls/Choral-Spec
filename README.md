# Distributed Speculative Decoding on AWS Trainium

This repository has been adapted for **multi-device** AWS Trainium usage with **speculative decoding**, using **Meta LLaMA ** (3.2 1B draft + 3.1 8B target) in **bfloat16**.

## Dpendencies

Create a Trainium instance with AWS Neuron SDK using EC2 with the following settings:

1. 1. **Name:** Your Name
   2. **AMI:** Deep Learning AMI Neuron (Ubuntu 22.04)
   3. **Instance type:** trn1.2xlarge
   4. **Key pair (login):** create a new key pair
   5. **Metadata version [under “Advanced details”]:** V2 only (otherwise, you will encounter a not authorized error)
   6. **When connecting to these instances via SSH, use the username of *ubuntu***
2. Activate the Neuron virtual environment to run inference by running `source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate`.

Install dependencies

```
pip install grpcio==1.71.0 grpcio-tools==1.66.2
pip install gevent
```

Install vLLM
NxD Inference supports running models with vLLM. This functionality is available in a fork of the vLLM GitHub repository:

aws-neuron/upstreaming-to-vllm
To run NxD Inference with vLLM, you need to download and install vLLM from this fork. Clone the Neuron vLLM fork.
```
git clone -b neuron-2.22-vllm-v0.7.2 https://github.com/aws-neuron/upstreaming-to-vllm.git
```

Install the Neuron vLLM fork into the virtual environment
```
cd upstreaming-to-vllm
pip install -r requirements-neuron.txt
VLLM_TARGET_DEVICE="neuron" pip install -e .
cd ..
```

We will use llmperf to measure the performance. We will use the load test feature of LLMPerf and measure the performance for accepting 10,000 tokens as input and generating 1500 tokens as output. Install llmperf into the virtual environment.
```
git clone https://github.com/ray-project/llmperf.git
cd llmperf
pip install -e .
```

Fix transformer version:
```
pip uninstall -y transformers
pip install --no-deps transformers==4.45.1
```

Check transformer version:
```
python - <<'PY'
import transformers, inspect
print("Transformers version:", transformers.__version__)
print("has shard_checkpoint:", "shard_checkpoint" in dir(transformers.modeling_utils))
PY
```

Fix tokenizer version:
```
pip uninstall -y tokenizers
pip install --no-deps "tokenizers==0.20.3" 
```

Fix run time error:
comment out lora related import
```
vi /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/lib/python3.10/site-packages/neuronx_distributed_inference/models/config.py
#from neuronx_distributed_inference.modules.lora_serving import LoraServingConfig
```
```
vi /home/ubuntu/Choral-Spec/upstreaming-to-vllm/vllm/worker/neuronx_distributed_model_runner.py
#from neuronx_distributed_inference.modules.lora_serving import (LoraCheckpoint, LoraServingConfig)
```

## Setup

1. **Clone Repo & Install**:

   ```
   git clone https://github.com/yfzzzyyls/Choral-Spec
   ```
2. **Download Models** (1B draft, 3B target) from Hugging Face. For example:

   ```
   cd ~
   mkdir models
   huggingface-cli download --token YOURTOKEN meta-llama/Llama-3.2-1B --local-dir /home/ubuntu/models/llama-3.2-1b
   ```
3. **Optinal: Generate new grpc files**

   ```
   cd Choral-Spec/grpc_comm
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto
   ```

   Notice: in the newly generated inference_pb2_grpc.py, if you have the following code:

   ```
   import inference_pb2 as inference__pb2
   ```

   replace it with:

   ```
   from . import inference_pb2 as inference__pb2
   ```

## **Usage:**

Compile the model

```
python compile_models.py --model-path /home/ubuntu/models/llama-3.1-8b/ --draft-model-path /home/ubuntu/models/llama-3.2-1b/ --output-compiled-dir ./compiled_llama3.1-8b --tp-size 2 --context-length 128 --speculation-length 4
```

Set Artifact path
```
export NEURON_COMPILED_ARTIFACTS=$PWD/compiled_llama3.1-8b
```

Activate the Neuron backend inside vLLM
```
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
```

Tell the runtime how many Neuron‑cores to use
```
export NEURON_RT_NUM_CORES=8
```

### **Optional:**

Clean cache before compile:

```
rm -r /var/tmp/neuron-compile-cache
```

Check neuron toolkit
```
pip list | grep neuron
```

### **Compile & Run the Target Model Server**

```
python main.py --role target --model /home/ubuntu/models/llama-3.2-3b/ --port 50051 --sequence_length 640
```

### **Compile & Run the Draft Model server**

```
python main.py --role draft --model /home/ubuntu/models/llama-3.2-1b/ --target_host 18.190.157.61 --port 50051 --prompt_text prompt.txt --max_new_tokens 128 --gamma 4 --sequence_length 640 --top_p 0.8 --temperature 0.1
```

### **Example Output**

```
INFO:inference.draft_worker:[Thread-2] Starting speculative decoding with session_id=104464132
INFO:inference.draft_worker:[Thread-0] Starting speculative decoding with session_id=3780988024
INFO:inference.draft_worker:[Thread-1] Starting speculative decoding with session_id=1574770097
INFO:inference.speculative:Speculative decoding match rate: 37.50% (Draft accepted: 48, Target generated: 80)
INFO:inference.speculative:Speculative decoding match rate: 22.48% (Draft accepted: 29, Target generated: 100)
INFO:inference.speculative:Speculative decoding match rate: 17.83% (Draft accepted: 23, Target generated: 106)

=== Final Batched Outputs ===

[Prompt 0 Output]:
Once upon a time, there there
......
[Prompt 1 Output]:
......
[Prompt 2 Output]:
......
```

## **Performance Profiling Stats**

```
INFO:inference.verify:Performance metrics saved to performance_target_only_20250408_013547.csv and performance_target_only_20250408_013547.json
```

Performance stats are saved to .cvs and .json files

## **Run a Single Model for Verification**

You can also run either the draft or target model **standalone** (without speculative decoding) to verify its generation output token-by-token. This is useful for debugging and sanity checks to ensure each model behaves as expected given a prompt.

To run the **target model** by itself on a prompt:

```
python main.py --role verify_target --model /home/ubuntu/models/llama-3.2-3b --prompt "Hi, how are you? Tell me about the difference between llama and alpaca." --sequence_length 640 --max_new_tokens 128 --profile
```

This will load the 3B target model and generate 100 tokens continuing the prompt, printing each generated token as it arrives, followed by the full output text.

Similarly, to run the **draft model** by itself:

```
python main.py --role verify_draft --model /home/ubuntu/models/llama-3.2-1b --prompt "Hi, how are you? Tell me about the difference between llama and alpaca." --sequence_length 640 --max_new_tokens 128 --profile
```

This will use the 1B draft model to generate text token-by-token for the given prompt.

*Note:* In verification modes, the model will be compiled on the fly if a compiled Neuron model is not found. By default, **`--sequence_length 128` is used; ensure you use the same sequence length that the model was compiled with (or specify** **`--sequence_length` accordingly) to avoid recompilation. The** `--max_tokens` option controls how many new tokens to generate for the prompt.

## **Advanced Tips**

* **NEURON_RT_VISIBLE_CORES**: If your instance has multiple NeuronCores, you can dedicate certain cores to the draft or server processes:

```
#In terminal 1 (server):export NEURON_RT_VISIBLE_CORES=4-15
python model_service.py ...#In terminal 2 (draft):export NEURON_RT_VISIBLE_CORES=0-3
python draft_client.py ...
```

This can allow parallel execution, improving throughput.

* **Larger Models**: If using LLaMA 7B or bigger, you might need to distribute the model across multiple Neuron cores. That requires advanced compilation with **neuronx-distributed** or optimum-neuron. The approach is similar; just ensure the code references the sharded model.
* **Modifying the Speculative Mechanism**: The draft code uses a simple loop with **use_cache=True**. If you want to do partial or multi-token steps differently, you can adapt the logic in **draft_client.py**
