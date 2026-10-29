"""Fresh mixed-serving campaign with both frozen HAWQ-FP4 controls."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from common import MODEL,RUN,WRAPPER,write_json
from full_quality_queue import invoke

TEMP=Path('/dev/shm/daso_rebuttal_006_extension_20261025')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
    parser=argparse.ArgumentParser();parser.add_argument('--repeat',action='store_true');args=parser.parse_args()
    design_path=RUN/'configs/baseline_extension_design.json'
    if args.repeat:
        assert json.loads((RUN/'results/extension_worker.json').read_text())['status']=='complete'
        original=json.loads(design_path.read_text());design_path=RUN/'configs/counterbalanced_repeat_design.json'
        if not design_path.exists():
            write_json(design_path,{**original,'policies':['uniform_dense_bf16']+list(reversed([p for p in original['policies'] if p!='uniform_dense_bf16'])),
                'first_campaign_design_sha256':hashlib.sha256((RUN/'configs/baseline_extension_design.json').read_bytes()).hexdigest(),
                'repeat_reason':'Standard BF16 endpoint/start approximately .942; repeat ALL frozen methods and cases in reverse order, preserving first campaign and all quality results.'})
    if not design_path.exists():
        original=json.loads((RUN/'configs/design.json').read_text())
        source=RUN.parent/'009/configs/candidate_pool.json';pool=json.loads(source.read_text())
        points=[p for p in pool['points'] if p.get('format')=='fp4_phase_pair' and p['scenario']=='prefill_decode']
        extra=list(dict.fromkeys(p['policy_id'] for p in points));assert len(extra)==2
        for pid in extra:shutil.copy2(RUN.parent/f'009/configs/policies/{pid}.json',RUN/f'configs/policies/{pid}.json')
        # Fixed order interleaves original controls and new HAWQ controls.
        policies=['uniform_dense_bf16','transfer_daso5_measured_prefill_decode_04',extra[0],'uniform_fp8',
            'transfer_daso8_prefill_decode_01',extra[1],'uniform_sq_int8','phase_sq_int8_w4a16_ours']
        write_json(design_path,{**original,'policies':policies,'original_policies':original['policies'],
            'extra_policies':extra,'extra_aliases':points,'baseline_bracket':'BF16 start/end, pool 10 timed bursts per case; report both endpoints',
            'old_design_sha256':hashlib.sha256((RUN/'configs/design.json').read_bytes()).hexdigest(),
            'hawq_pool_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'quality_reuse':'Exact original method maps/canonical states/scales, same data/targets/scheduler and backend; original files immutable. New HAWQ policies freshly scored on all three cases.',
            'selection':'All predeclared FP4 P+D sign/cap variants, two unique maps; no mixed-quality or mixed-speed tuning'})
    design=json.loads(design_path.read_text());stage='repeat' if args.repeat else 'extension';state=RUN/f'results/{stage}_worker.json'
    temp=TEMP.with_name(TEMP.name+'_repeated') if args.repeat else TEMP
    write_json(state,{'status':'running','gpu':6,'pid':os.getpid()})
    jobs=[(p,False) for p in design['policies']]+[('uniform_dense_bf16',True)]
    try:
        for policy,ending in jobs:
            model=Path('/dev/shm/daso_rebuttal_001_20261025/checkpoints')/policy;owned=policy!='uniform_dense_bf16'
            if owned:
                model=temp/'checkpoints'/policy
                done=RUN/f'results/{stage}_complete/{policy}.json'
                if done.exists():continue
                if not (model/'model.safetensors').exists():
                    if model.exists():model.rename(model.with_name(policy+'_previous'))
                    command=[sys.executable,RUN/'code/export_model.py','--model-path',MODEL,'--policy-json',RUN/f'configs/policies/{policy}.json',
                        '--output-dir',model,'--cutlass-wrapper-path',WRAPPER]
                    content=(RUN/f'configs/policies/{policy}.json').read_text()
                    for method in ['sparse_bf16','sparse_nvfp4']:
                        if method in content:command+=['--canonical-'+method.replace('_','-')+'-state',Path('/dev/shm/daso_rebuttal_001_20261025/canonical')/method/'model.pt']
                    invoke(command,policy+'_'+stage+'_export')
                metadata=RUN/f'results/exports_{stage}/{policy}';metadata.mkdir(parents=True,exist_ok=True)
                for p in model.glob('*.json'):shutil.copy2(p,metadata/p.name)
            marker=RUN/f'results/{stage}_complete/{policy}{"_end" if ending else ""}.json'
            if marker.exists():continue
            for budget in [1024,4096]:
                command=[sys.executable,RUN/'code/benchmark.py','--checkpoint',model,'--policy',policy,'--budget',budget,'--extension']
                if args.repeat:command+=['--repeat']
                if ending:command+=['--baseline-end']
                invoke(command,policy+'_'+stage+'_speed_'+str(budget)+('_end' if ending else ''))
            if policy in design['extra_policies'] and not args.repeat:
                for budget in [1024,4096]:invoke([sys.executable,RUN/'code/benchmark.py','--checkpoint',model,'--policy',policy,'--budget',budget,
                    '--extension','--full-quality'],policy+'_extension_quality_'+str(budget))
            write_json(marker,{'complete':True,'gpu':6})
            if owned:
                binary=model/'model.safetensors';size=binary.stat().st_size;binary.unlink()
                write_json(RUN/f'results/exports_{stage}/{policy}/binary_cleanup.json',{'bytes':size,'path':str(binary),
                    'reason':'all extension speed/quality checks complete; all source/log/CSV/result/document/metadata retained'})
            print(policy,'extension complete',flush=True)
        write_json(state,{'status':'complete','gpu':6,'pid':os.getpid()})
    except Exception as e:write_json(state,{'status':'failed','error':repr(e)});raise


if __name__=='__main__':main()
