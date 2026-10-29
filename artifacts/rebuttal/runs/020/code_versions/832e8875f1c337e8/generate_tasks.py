from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu

from common import RUN, WRAPPER, SPEC, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--dataset', choices=['cnn_dm_1000','dsum','IWSLT'], required=True)
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    os.environ['PHASE_HETERO_TRACE'] = '0'
    os.environ['PHASE_HETERO_MYTEST_CUTLASS_WRAPPER_PATH'] = str(WRAPPER)
    os.environ['CUTLASS_WRAPPER_SPARSE_BF16_MAX_MATMUL_CACHE_ENTRIES'] = '16'
    from runtime_extensions import install
    phase = install()
    from graph_bindings import install as graph_bindings
    graph_bindings()
    from arena_runtime import ArenaRuntime
    runtime = ArenaRuntime(phase)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    source = RUN / f'data/generation_{args.dataset}.jsonl'
    questions = [json.loads(line) for line in source.read_text().splitlines()][args.begin:args.end]
    output = RUN / f'results/generation/{args.policy}/{args.dataset}'
    output.mkdir(parents=True, exist_ok=True)
    answers = output / f'answers_{args.begin:04d}_{args.end or "end"}.jsonl'
    if answers.exists():
        raise FileExistsError(answers)
    llm = LLM(model=str(args.checkpoint), dtype='bfloat16', enforce_eager=False,
        compilation_config={'level':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[SPEC['generation_batch']]},
        enable_prefix_caching=False, enable_chunked_prefill=False,
        max_model_len=4096, max_num_seqs=SPEC['generation_batch'], max_num_batched_tokens=SPEC['generation_batch']*4096,
        gpu_memory_utilization=.90, kv_cache_memory_bytes=SPEC['generation_kv_gib']*1024**3)
    from admission_check import check
    admission = check(llm, SPEC['generation_batch'], 3840, 256)
    runtime.activate()
    sampling = SamplingParams(temperature=0., max_tokens=256)
    started = time.perf_counter()
    with answers.open('w') as handle:
        for start in range(0, len(questions), SPEC['generation_batch']):
            if start:
                phase.wait_for_prefill_ready()
            batch = questions[start:start+SPEC['generation_batch']]
            outputs = llm.generate([TokensPrompt(prompt_token_ids=row['prompt_token_ids'])
                                   for row in batch], sampling, use_tqdm=False)
            for question, response in zip(batch, outputs, strict=True):
                record = {'question_id':question['question_id'], 'reference':question['reference'],
                    'prediction':response.outputs[0].text, 'generated_token_ids':list(response.outputs[0].token_ids),
                    'prompt_tokens':len(question['prompt_token_ids'])}
                handle.write(json.dumps(record, ensure_ascii=False)+'\n')
            handle.flush()
            phase.prepare_next_prefill()
            if start % 80 == 0:
                print(args.policy,args.dataset,start+len(batch),'/',len(questions),
                      'seconds',time.perf_counter()-started,flush=True)
    rows = [json.loads(line) for line in answers.read_text().splitlines()]
    predictions = [row['prediction'].strip() or ' ' for row in rows]
    references = [row['reference'] for row in rows]
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    metrics = {'policy_id':args.policy,'dataset':args.dataset,'samples':len(rows),
        'begin':args.begin,'end':args.end,'generation_seconds':time.perf_counter()-started,
        'rougeL_percent':100*sum(scorer.score(ref,pred)['rougeL'].fmeasure
            for ref,pred in zip(references,predictions,strict=True))/len(rows),
        'empty_predictions':sum(not row['prediction'].strip() for row in rows),
        'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'kv_cache_memory_bytes':SPEC['generation_kv_gib']*1024**3, 'quality_batch':SPEC['generation_batch'], 'admission':admission,'max_new_tokens':256,
        'chat_template':SPEC['generation_prompt'],
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    if args.dataset=='IWSLT':
        metrics['sacre_bleu']=corpus_bleu(predictions,[references]).score
    write_json(output/f'metrics_{args.begin:04d}_{args.end or "end"}.json',metrics)
    llm.llm_engine.engine_core.shutdown()
    print(metrics,flush=True)


if __name__=='__main__':main()
