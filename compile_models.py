# ============================================
# File: compile_models.py
# ============================================
#!/usr/bin/env python3
import argparse, os, subprocess, sys

def main():
    parser = argparse.ArgumentParser(
        description="Compile a LLaMA model with fused speculative decoding for AWS Trn1"
    )
    parser.add_argument(
        '--model-path', required=True,
        help='Path to the main LLaMA 3.1 8B model (HF directory or checkpoint)'
    )
    parser.add_argument(
        '--draft-model-path', required=True,
        help='Path to the draft LLaMA 3.2 1B model'
    )
    parser.add_argument(
        '--output-compiled-dir', required=True,
        help='Directory to write compiled Neuron artifacts'
    )
    parser.add_argument(
        '--tp-size', type=int, required=True,
        help='Tensor parallel size (number of Neuron cores to use)'
    )
    parser.add_argument(
        '--context-length', type=int, required=True,
        help='Maximum context length (model context window)'
    )
    parser.add_argument(
        '--speculation-length', type=int, required=True,
        help='Number of tokens to speculate ahead (fused speculation)'
    )
    args = parser.parse_args()

    cmd = [
        'inference_demo',
        '--model-type', 'llama',
        '--task-type', 'causal-lm',
        'run',
        '--model-path', args.model_path,
        '--draft-model-path', args.draft_model_path,
        '--compiled-model-path', args.output_compiled_dir,
        '--tp-degree', str(args.tp_size),
        '--max-context-length', str(args.context_length),
        '--seq-len', str(args.context_length + args.speculation_length),
        '--enable-fused-speculation',
        '--speculation-length', str(args.speculation_length),
        '--pad-token-id', '2',
        '--prompt', 'CompileOnly',          # dummy prompt to satisfy CLI
        '--on-device-sampling',
        '--compile-only',
    ]

    env = os.environ.copy()
    print('Running compile command:\n  ' + ' '.join(cmd))
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        sys.exit(ret.returncode)
    print(f'Compilation finished. Artifacts in: {args.output_compiled_dir}')

if __name__ == '__main__':
    main()