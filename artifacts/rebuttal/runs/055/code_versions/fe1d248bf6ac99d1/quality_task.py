"""Full original downstream protocols for a frozen BF16/DASO/HAWQ point."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from common import MODEL, RUN, WRAPPER, write_json

EXPECTED = {'wikitext': (62, 'word_perplexity,none'), 'arc_easy': (2376, 'acc,none'),
            'arc_challenge': (1172, 'acc_norm,none'), 'winogrande': (1267, 'acc,none'),
            'mmlu': (14042, 'acc,none')}


def frozen_datasets():
    import datasets
    records = json.loads((RUN / 'configs/likelihood_datasets.json').read_text())
    lookup = {(row['dataset_path'], row['dataset_name']): row for row in records}
    cache = {}
    def load(path, name=None, *args, split=None, **kwargs):
        key = path, name
        assert key in lookup, key
        if key not in cache:
            cache[key] = datasets.load_from_disk(lookup[key]['path'])
        return cache[key][split] if split is not None else cache[key]
    datasets.load_dataset = load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--task', choices=[*EXPECTED, 'cnn_dm_1000', 'dsum', 'IWSLT'], required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()
    formal = json.loads((RUN / 'configs/formal_points.json').read_text())
    assert args.policy in {case['policy_id'] for case in formal['unique_cases']}
    design = json.loads((RUN / 'configs/design.json').read_text())
    policy_path = RUN / f'configs/policies/{args.policy}.json'
    policy = json.loads(policy_path.read_text())
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest() == design['policies'][args.policy]['map_sha256']
    exported = json.loads((args.checkpoint / 'phase_hetero_policy.json').read_text())
    assert exported['method_map'] == policy['method_map']
    output = RUN / f'results/tasks/{args.policy}/{args.task}/{args.attempt}'
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_FLASHINFER_SAMPLER='0',
                      PHASE_HETERO_TRACE='0', CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES='8')
    os.environ['PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH'] = str(WRAPPER)
    from graph_bindings import install as bindings
    bindings()
    from runtime_extensions import install
    phase = install()
    from arena_runtime import ArenaRuntime
    runtime = ArenaRuntime(phase)
    import torch
    started = time.perf_counter()
    model = None
    metadata = {'policy': args.policy, 'task': args.task, 'checkpoint': str(args.checkpoint),
                'policy_sha256': design['policies'][args.policy]['map_sha256'],
                'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'quality_only': True}
    try:
        if args.task not in EXPECTED:
            from vllm import LLM, SamplingParams
            from vllm.inputs import TokensPrompt
            protocol = json.loads((RUN / 'configs/generation_protocol.json').read_text())
            source = RUN / f'data/generation_{args.task}.jsonl'
            assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol['tasks'][args.task]['sha256']
            questions = [json.loads(line) for line in source.read_text().splitlines()][args.begin:args.end]
            assert questions
            model = LLM(model=str(args.checkpoint), tokenizer=str(MODEL), dtype='bfloat16', enforce_eager=False,
                compilation_config={'level': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [8]},
                enable_prefix_caching=False, enable_chunked_prefill=False, max_model_len=4096, max_num_seqs=8,
                max_num_batched_tokens=32768, gpu_memory_utilization=.9, kv_cache_memory_bytes=8 * 1024**3)
            assert len(phase._LAYERS) == 256
            runtime.activate()
            sampling = SamplingParams(temperature=0., max_tokens=256)
            with (output / 'answers.jsonl').open('w') as stream:
                for start in range(0, len(questions), 8):
                    rows = questions[start:start + 8]
                    if phase.current_phase() == 'decode':
                        runtime.restore(); runtime.wait_prefill()
                    answers = model.generate([TokensPrompt(prompt_token_ids=row['prompt_token_ids']) for row in rows],
                                             sampling, use_tqdm=False)
                    assert len(answers) == len(rows) and phase.current_phase() == 'decode'
                    assert len(runtime.scheduled) == 256
                    for offset, (question, answer) in enumerate(zip(rows, answers, strict=True)):
                        value = {'question_id': question['question_id'], 'reference': question['reference'],
                            'prediction': answer.outputs[0].text, 'generated_token_ids': answer.outputs[0].token_ids,
                            'question_index': args.begin + start + offset,
                            'prompt_tokens': len(question['prompt_token_ids'])}
                        assert len(value['generated_token_ids']) <= 256
                        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
                    stream.flush()
                    print(args.policy, args.task, start + len(rows), '/', len(questions), flush=True)
            metadata.update(complete=True, samples=len(questions), begin=args.begin, end=args.end,
                answers=str(output / 'answers.jsonl'), max_new_tokens=256, normal_eos=True,
                batch=8, max_model_len=4096, kv_cache_bytes=8 * 1024**3,
                input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        else:
            frozen_datasets()
            from lm_eval import simple_evaluate
            from lm_eval.models.vllm_causallms import VLLM
            from lm_eval.tasks import TaskManager
            batch = 1 if args.task == 'wikitext' else 4
            lm = VLLM(pretrained=str(args.checkpoint), tokenizer=str(MODEL), dtype='bfloat16', batch_size=batch,
                max_model_len=2048, tensor_parallel_size=1, enforce_eager=True, enable_prefix_caching=False,
                enable_chunked_prefill=False, gpu_memory_utilization=.9, max_num_seqs=batch,
                skip_mm_profiling=True, kv_cache_memory_bytes=3 * 1024**3)
            model = lm.model
            assert len(phase._LAYERS) == 256
            runtime.activate(); runtime.active = False
            result = simple_evaluate(model=lm, tasks=[args.task], num_fewshot=0, batch_size=batch,
                limit=None, log_samples=True, random_seed=0, numpy_random_seed=0, torch_random_seed=0,
                fewshot_random_seed=0, task_manager=TaskManager())
            assert result is not None
            expected, metric = EXPECTED[args.task]
            selected = result.get('results', {}).get(args.task) or result.get('groups', {}).get(args.task)
            value = selected[metric]
            assert math.isfinite(float(value)) and sum(row['effective'] for row in result['n-samples'].values()) == expected
            assert phase.current_phase() == 'prefill' and not runtime.scheduled and runtime.generations == 0
            metadata.update(complete=True, samples=expected, metric=metric, metric_value=value,
                raw_lm_eval=result, num_fewshot=0, limit=None, batch=batch, max_model_len=2048,
                kv_cache_bytes=3 * 1024**3)
        metadata.update(elapsed_seconds=time.perf_counter() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(output / 'result.json', metadata)
    finally:
        if model is not None:
            model.llm_engine.engine_core.shutdown()


if __name__ == '__main__':
    main()
