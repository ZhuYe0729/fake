"""Full-checkpoint hardware probe; not a formal phase-vLLM measurement."""
import argparse
import json
import os
from pathlib import Path
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'},
        attn_implementation='sdpa', local_files_only=True).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    assert len(model.model.layers) == 64
    assert all(parameter.is_cuda for parameter in model.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    calls = [0] * 64
    handles = []
    for index, layer in enumerate(model.model.layers):
        def observe(module, inputs, output, index=index):
            calls[index] += 1
        handles.append(layer.register_forward_hook(observe))
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'What is the capital of France? Answer briefly.'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors='pt').to('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=16, min_new_tokens=16,
                                 do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    generated = outputs[0, inputs.input_ids.shape[1]:].tolist()
    assert len(generated) == 16 and all(count == 16 for count in calls)
    payload = {'complete': True, 'role': 'hardware/checkpoint feasibility only',
               'formal_phase_runtime_result': False, 'model': str(args.model.resolve()),
               'torch': torch.__version__, 'transformers': transformers.__version__,
               'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
               'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
               'dtype': 'bfloat16', 'attention': 'sdpa', 'layers': 64,
               'layer_forward_calls': calls, 'load_seconds': load_seconds,
               'input_ids': inputs.input_ids[0].tolist(), 'output_ids': generated,
               'output_text': tokenizer.decode(generated),
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
               'limitation': 'HF probe on existing environment; no speed or quality comparison to archived vLLM.'}
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
