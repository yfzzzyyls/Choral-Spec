# ============================================
# File: choral_server/api_server.py
# ============================================
#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess

# Ensure vLLM is available before we proceed
try:
    import vllm  # noqa: F401
except ModuleNotFoundError:
    sys.stderr.write(
        "Error: vLLM is not installed in this environment.\n"
        "Install a Neuron‑compatible build, e.g.:\n"
        "  pip install --no-deps vllm==0.4.2\n"
        "or use the pre‑built wheel on the DLAMI:\n"
        "  pip install /opt/aws_neuronx_wheels/vllm-0.4.2*whl\n"
    )
    sys.exit(1)


def main():
    p = argparse.ArgumentParser(
        description='Start vLLM-based Neuron inference server (OpenAI API compatible)'
    )
    # Model paths
    p.add_argument('--model', dest='model_path', required=False,
                   help='Path to the target LLaMA model (e.g. llama3.1-8b)')
    p.add_argument('--speculative-model', dest='spec_model_path', required=False,
                   help='Path to the draft LLaMA model (e.g. llama3.2-1b)')
    # Speculation settings
    p.add_argument('--num-speculative-tokens', type=int, default=7,
                   help='Tokens to speculate ahead each step')
    # Parallelism and context
    p.add_argument('--tensor-parallel-size', type=int, default=8,
                   help='Number of Neuron cores to use for tensor parallelism')
    p.add_argument('--max-model-len', type=int, default=4096,
                   help='Maximum model context length (tokens)')
    # Neuron overrides
    p.add_argument('--override-neuron-config', default='{"enable_fused_speculation":true}',
                   help='JSON string of Neuron config overrides')
    # Server settings
    p.add_argument('--host', default='0.0.0.0', help='Host/interface to bind')
    p.add_argument('--port', type=int, default=8000, help='Port to listen on')
    p.add_argument('--device', default='neuron', help='Device type (neuron|cuda|cpu)')
    args = p.parse_args()

    # Resolve environment defaults
    env = os.environ.copy()
    # Ensure Neuron framework is set for vLLM
    env.setdefault('VLLM_NEURON_FRAMEWORK', 'neuronx-distributed-inference')
    # NEURON_COMPILED_ARTIFACTS must be set externally
    if 'NEURON_COMPILED_ARTIFACTS' not in env:
        print('Error: NEURON_COMPILED_ARTIFACTS env var not set', file=sys.stderr)
        sys.exit(1)

    # Build the vLLM server command
    cmd = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', args.model_path or env.get('MODEL_PATH'),
        '--device', args.device,
        '--tensor-parallel-size', str(args.tensor_parallel_size),
        '--max-model-len', str(args.max_model_len),
        '--speculative-model', args.spec_model_path or env.get('SPECULATIVE_MODEL_PATH'),
        '--num-speculative-tokens', str(args.num_speculative_tokens),
        '--override-neuron-config', args.override_neuron_config,
        '--host', args.host,
        '--port', str(args.port),
    ]

    print('Launching vLLM server:')
    print('  ' + ' '.join(cmd))
    subprocess.run(cmd, env=env)

if __name__ == '__main__':
    main()