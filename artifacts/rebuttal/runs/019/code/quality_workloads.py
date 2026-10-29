"""Per-load teacher-forced quality, without selecting or tuning any policy."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from common import RUN, write_json
from engine import load, admission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--policy',required=True)
    args = parser.parse_args()
    design = json.loads((RUN/'configs/design.json').read_text())
    llm,phase,runtime,initialization = load(args.checkpoint,quality=True)
    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    try:
        for case in design['workloads']:
            target = RUN/f'results/quality/{args.policy}/{case["label"]}.json'
            if target.exists():continue
            check = admission(llm,case)
            path = Path(case['quality_file'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==case['quality_sha256']
            data = torch.load(path,weights_only=True)
            rows,begin = [],time.perf_counter()
            for start in range(0,len(data),case['batch']):
                blocks = data[start:start+case['batch']]
                prompts,params,captures,forced = [],[],[],[]
                for offset,block in enumerate(blocks):
                    capture = RUN/f'results/captures/{args.policy}/{case["label"]}/{start+offset:04d}.jsonl'
                    assert not capture.exists()
                    ids = block[case['prompt']:].tolist()
                    assert len(ids)==case['output']
                    prompts.append(TokensPrompt(prompt_token_ids=block[:case['prompt']].tolist()))
                    params.append(SamplingParams(temperature=0.,max_tokens=case['output'],min_tokens=case['output'],ignore_eos=True,detokenize=False,
                        extra_args={'teacher_force_target_ids':ids,'teacher_force_capture_path':str(capture)}))
                    captures.append(capture);forced.append(ids)
                if phase.current_phase()=='decode':runtime.restore();runtime.wait_prefill()
                answers = llm.generate(prompts,params,use_tqdm=False)
                for offset,answer in enumerate(answers):
                    assert answer.outputs[0].token_ids==forced[offset]
                    values = [json.loads(line)['logprob'] for line in captures[offset].read_text().splitlines()]
                    assert len(values)==case['output']
                    assert all(math.isfinite(value) for value in values), (args.policy,case['label'],start+offset)
                    rows.append({'block':start+offset,'avg_nll':-sum(values)/len(values),'token_count':len(values)})
                print(args.policy,case['label'],len(rows),'/',len(data),flush=True)
            assert len(rows)==64
            write_json(target,{'complete':True,'policy':args.policy,'workload':case['label'],'blocks':rows,
                'quality_sha256':case['quality_sha256'],'quality_batch':case['batch'],'admission':check,
                'initialization_ms':initialization,'elapsed_seconds':time.perf_counter()-begin})
    finally:llm.llm_engine.engine_core.shutdown()


if __name__ == '__main__':main()
