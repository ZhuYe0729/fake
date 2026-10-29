"""Use GPU 6 after its dense generation jobs for the last frozen DASO representative."""
import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from pathlib import Path

from baseline_worker import TEMP,invoke
from checkpoint_evidence import inspect_checkpoint
from common import MODEL,RUN,VLLM,WRAPPER,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int,required=True)
    args=parser.parse_args()
    while Path(f'/proc/{args.wait_pid}').exists():time.sleep(5)
    selection=RUN/'configs/selected_daso_prefill_only.json'
    while not selection.exists():time.sleep(5)
    policy_id=json.loads(selection.read_text())['policies'][-1]
    state={'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'policy_id':policy_id,'rule':'last predeclared representative; no downstream scores','status':'running'}
    target=RUN/'results/extra_prefill_representative.json';write_json(target,state)
    lock=RUN/'results/locks'/f'{policy_id}.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        tasks=RUN/'results/tasks'/policy_id
        if all((tasks/task/'full/result.json').exists() for task in ['wikitext','winogrande','arc_easy','arc_challenge','mmlu']):
            state['status']='already_complete';write_json(target,state);return
        checkpoint=TEMP/'extra_prefill_checkpoints'/policy_id
        if checkpoint.exists():checkpoint.rename(checkpoint.with_name(checkpoint.name+f'_previous_{time.time_ns()}'))
        policy=RUN/f'configs/policies/{policy_id}.json'
        command=[sys.executable,str(RUN/'code/export_model.py'),'--model-path',str(MODEL),
            '--policy-json',str(policy),'--output-dir',str(checkpoint),'--cutlass-wrapper-path',str(WRAPPER)]
        for method in ['sparse_bf16','sparse_nvfp4']:
            if method in policy.read_text():command+=['--canonical-'+method.replace('_','-')+'-state',str(TEMP/'canonical'/method/'model.pt')]
        invoke(command,RUN/f'logs/extra_prefill_{policy_id}_export.log')
        manifest=RUN/f'configs/extra_manifest_{policy_id}.json'
        write_json(manifest,{'model_path':str(MODEL),'policies':[{'label':policy_id,'kind':'ours',
            'checkpoint':str(checkpoint),'policy_json':str(policy)}]})
        invoke([sys.executable,str(RUN/'code/run_workflow.py'),'tasks','--manifest',str(manifest),
            '--policy',policy_id,'--output-root',str(RUN/'results/tasks'),'--vllm-root',str(VLLM),
            '--cutlass-wrapper-path',str(WRAPPER),'--audit'],RUN/f'logs/extra_prefill_{policy_id}_tasks.log')
        evidence=RUN/f'results/extra_prefill_export/{policy_id}';evidence.mkdir(parents=True,exist_ok=True)
        for source in checkpoint.glob('*.json'):shutil.copy2(source,evidence/source.name)
        inspect_checkpoint(checkpoint,evidence)
        binary=checkpoint/'model.safetensors';size=binary.stat().st_size;binary.unlink()
        write_json(evidence/'temporary_binary_cleanup.json',{'path':str(binary),'bytes':size,
            'reason':'owned extra representative export after successful full tasks; metadata retained'})
    state['status']='complete';write_json(target,state)


if __name__=='__main__':main()
