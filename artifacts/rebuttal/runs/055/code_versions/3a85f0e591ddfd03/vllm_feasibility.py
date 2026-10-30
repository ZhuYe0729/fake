"""Probe the full raw-HF BF16 model and planned KV capacity in the locked fork."""
import json
import os
from pathlib import Path

from common import MODEL, RUN, write_json
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    output = RUN / 'results/bf16_vllm_feasibility.json'
    assert not output.exists()
    design = json.loads((RUN / 'configs/design.json').read_text())
    settings = design['engine']
    llm = LLM(model=str(MODEL), tokenizer=str(MODEL), dtype='bfloat16',
              enforce_eager=False, enable_prefix_caching=False, enable_chunked_prefill=False,
              compilation_config={'level': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
                                  'cudagraph_capture_sizes': settings['capture_sizes']},
              max_model_len=settings['max_model_len'], max_num_seqs=settings['max_num_seqs'],
              max_num_batched_tokens=settings['max_num_batched_tokens'],
              kv_cache_memory_bytes=settings['kv_gib'] * 1024**3,
              gpu_memory_utilization=.9, skip_mm_profiling=True)
    try:
        ids = json.loads((RUN / 'configs/speed_inputs.json').read_text())
        params = SamplingParams(temperature=0., max_tokens=64, min_tokens=64,
                                ignore_eos=True, detokenize=False)
        answers = llm.generate([TokensPrompt(prompt_token_ids=row) for row in ids], params, use_tqdm=False)
        result = [answer.outputs[0].token_ids for answer in answers]
        assert len(result) == 8 and all(len(row) == 64 for row in result)
        torch.cuda.synchronize()
        write_json(output, {'complete': True, 'role': 'raw-HF BF16 capacity probe only',
            'formal_phase_runtime_result': False, 'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'model': str(MODEL),
            'engine': settings, 'batch': 8, 'prompt_tokens': 2048, 'output_tokens': 64,
            'output_token_ids': result, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
