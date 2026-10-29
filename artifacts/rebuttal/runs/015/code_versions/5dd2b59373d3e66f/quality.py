"""Quality labels for new maps; selection reads validation, never test."""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from common import MODEL, REPO, RUN, WRAPPER, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--split', choices=['validation', 'test'], required=True)
    parser.add_argument('--scenarios', nargs='+', required=True)
    args = parser.parse_args()
    if args.split == 'test':
        assert (RUN / 'configs/selected_reanalysis.json').exists()
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_FLASHINFER_SAMPLER='0', PHASE_HETERO_TRACE='0')
    for key in ['NVFP4_MYTEST_CUTLASS_WRAPPER_PATH', 'MARLIN_NVFP4_MYTEST_CUTLASS_WRAPPER_PATH',
                'PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH']:
        os.environ[key] = str(WRAPPER)
    from graph_bindings import install as bindings
    bindings()
    from runtime_extensions import install
    phase = install()
    from arena_runtime import ArenaRuntime
    runtime = ArenaRuntime(phase)
    import torch
    sys.path.insert(0, str(REPO / 'daso/workflows/paper/runtime'))
    from teacher_force_processor import TeacherForceBatchProcessor
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=str(args.checkpoint), tokenizer=str(MODEL), dtype='bfloat16',
        enforce_eager=False, compilation_config={'level': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
                                                'cudagraph_capture_sizes': [8]},
        enable_prefix_caching=False, enable_chunked_prefill=False, max_model_len=2120,
        max_num_seqs=8, max_num_batched_tokens=16384, kv_cache_memory_bytes=6 * 1024**3,
        gpu_memory_utilization=.9, skip_mm_profiling=True, logits_processors=[TeacherForceBatchProcessor])
    runtime.activate()
    dataset = RUN / f'data/{args.split}_64_2112.pt'
    data = torch.load(dataset, weights_only=True)
    assert tuple(data.shape) == (64, 2112)
    try:
        for scenario in args.scenarios:
            target = RUN / f'results/{args.split}/{args.policy}/{scenario}.json'
            assert not target.exists(), 'Preserve existing quality label.'
            output = 1 if scenario == 'prefill_only' else 64
            rows = []
            start = time.perf_counter()
            captures = RUN / f'results/captures/{args.split}/{args.policy}/{scenario}'
            for offset in range(0, 64, 8):
                prompts, parameters, paths, forced = [], [], [], []
                for i in range(8):
                    block = data[offset + i]
                    ids, targets = block[:2048].tolist(), block[2048:].tolist()
                    capture = captures / f'block_{offset+i:04d}.jsonl'
                    assert not capture.exists()
                    prompts.append(TokensPrompt(prompt_token_ids=ids))
                    paths.append(capture)
                    forced.append(targets)
                    parameters.append(SamplingParams(temperature=0., max_tokens=output, min_tokens=output,
                        ignore_eos=True, detokenize=False,
                        prompt_logprobs=1 if scenario == 'prefill_only' else None,
                        extra_args=None if scenario == 'prefill_only' else {
                            'teacher_force_target_ids': targets, 'teacher_force_capture_path': str(capture)}))
                if phase.current_phase() == 'decode':
                    runtime.restore()
                    runtime.wait_prefill()
                answers = llm.generate(prompts, parameters, use_tqdm=False)
                assert phase.current_phase() == 'decode'
                for i, answer in enumerate(answers):
                    if scenario == 'prefill_only':
                        ids = prompts[i]['prompt_token_ids']
                        values = [float(answer.prompt_logprobs[j][ids[j]].logprob) for j in range(1, 2048)]
                    else:
                        assert answer.outputs[0].token_ids == forced[i]
                        values = [json.loads(line)['logprob'] for line in paths[i].read_text().splitlines()]
                        assert len(values) == 64
                    rows.append({'block': offset + i, 'token_count': len(values),
                                 'nll': -sum(values), 'avg_nll': -sum(values) / len(values)})
                print(args.policy, args.split, scenario, offset + 8, '/64', flush=True)
                write_json(RUN / f'results/quality_progress/{args.policy}_{args.split}_{scenario}.json',
                           {'completed_blocks': offset + 8, 'blocks': rows})
            count = sum(x['token_count'] for x in rows)
            total = sum(x['nll'] for x in rows)
            write_json(target, {'policy': args.policy, 'scenario': scenario, 'mode': args.split,
                'avg_nll': total / count, 'token_count': count, 'blocks': rows,
                'data_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                'quality_score_batch': 8, 'runtime': 'phase-reused arena and native decode graphs',
                'elapsed_seconds': time.perf_counter() - start})
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
