"""Original-protocol training labels and frozen Graph validation/confirmation."""
import argparse
import gc
import hashlib
import json
import os
import sys
import time

from common import MODEL, REPO, RUN, WRAPPER, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--split', choices=['training', 'validation', 'confirmation'], required=True)
    parser.add_argument('--scenarios', nargs='+', required=True)
    args = parser.parse_args()
    if args.split == 'confirmation':
        assert (RUN / 'configs/selected_policies.json').exists()
    training = args.split == 'training'
    assert not training or len(args.scenarios) == 1
    batch = 1 if training and args.scenarios == ['prefill_only'] else 8
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
    length = 2056 if training and batch == 1 else 2120
    batch_tokens = (length if batch == 1 else batch*2112) if training else 16384
    llm = LLM(model=args.checkpoint, tokenizer=str(MODEL), dtype='bfloat16', enforce_eager=training,
        compilation_config=None if training else {'level': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [8]},
        enable_prefix_caching=False, enable_chunked_prefill=False, max_model_len=length,
        max_num_seqs=batch, max_num_batched_tokens=batch_tokens, kv_cache_memory_bytes=(3 if training else 6) * 1024**3,
        gpu_memory_utilization=.90, skip_mm_profiling=True, logits_processors=[TeacherForceBatchProcessor])
    runtime.activate()
    design = json.loads((RUN / 'configs/frozen_design.json').read_text())
    try:
        for scenario in args.scenarios:
            if training:
                dataset = RUN / ('data/calibration_train_100_2048.pt' if scenario == 'prefill_only' else 'data/calibration_128_2112.pt')
                count = 100
                target = RUN / f'results/training/{scenario}/{args.policy}.json'
            else:
                dataset = __import__('pathlib').Path(design['datasets'][args.split]['file'])
                count = design['datasets'][args.split]['blocks']
                target = RUN / f'results/{args.split}/{args.policy}/{scenario}.json'
            assert not target.exists()
            data = torch.load(dataset, weights_only=True)[:count]
            output = 1 if scenario == 'prefill_only' else 64
            rows, started = [], time.perf_counter()
            for start in range(0, count, batch):
                prompts, params, paths, forced = [], [], [], []
                for offset, block in enumerate(data[start:start+batch]):
                    capture = RUN / f'results/captures/{args.split}/{args.policy}/{scenario}/block_{start+offset:04d}.jsonl'
                    assert not capture.exists()
                    ids, targets = block[:2048].tolist(), block[2048:2112].tolist()
                    prompts.append(TokensPrompt(prompt_token_ids=ids))
                    paths.append(capture)
                    forced.append(targets)
                    settings = {'temperature': 0., 'max_tokens': output, 'detokenize': False,
                        'prompt_logprobs': 1 if output == 1 else None,
                        'extra_args': None if output == 1 else {'teacher_force_target_ids': targets, 'teacher_force_capture_path': str(capture)}}
                    if not training:
                        settings.update(min_tokens=output, ignore_eos=True)
                    params.append(SamplingParams(**settings))
                if phase.current_phase() == 'decode':
                    runtime.restore()
                    runtime.wait_prefill()
                answers = llm.generate(prompts, params, use_tqdm=False)
                assert phase.current_phase() == 'decode'
                for offset, answer in enumerate(answers):
                    if output == 1:
                        ids = prompts[offset]['prompt_token_ids']
                        values = [float(answer.prompt_logprobs[j][ids[j]].logprob) for j in range(1, 2048)]
                    else:
                        assert answer.outputs[0].token_ids == forced[offset]
                        values = [json.loads(line)['logprob'] for line in paths[offset].read_text().splitlines()]
                        assert len(values) == 64
                    rows.append({'block': start+offset, 'token_count': len(values), 'nll': -sum(values), 'avg_nll': -sum(values)/len(values)})
                print(args.policy, args.split, scenario, len(rows), '/', count, flush=True)
            total = sum(row['token_count'] for row in rows)
            write_json(target, {'policy': args.policy, 'scenario': scenario, 'mode': args.split,
                'blocks': rows, 'token_count': total, 'avg_nll': sum(row['nll'] for row in rows)/total,
                'data_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(), 'quality_score_batch': batch,
                'enforce_eager': training, 'elapsed_seconds': time.perf_counter()-started,
                'scope': 'Fixed training protocol or new nonoverlapping confirmation; no test-based selection.'})
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
