"""Final, sequential speed measurements on one physical RTX 5090 (GPU 3)."""
import os
import shutil
import sys
import time

from baseline_worker import TEMP,invoke
from checkpoint_evidence import inspect_checkpoint
from common import MODEL,RUN,VLLM,WRAPPER,write_json
from report_results import points


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    while not (RUN/'results/daso/prefill_decode/predicted_points.json').exists():time.sleep(10)
    policies=list(dict.fromkeys(point['policy_id'] for point in points()))
    write_json(RUN/'configs/common_card_speed.json',{'physical_gpu':3,'policies':policies,
        'rule':'all final operating points, sequentially on the same GPU, same fixed 3-GiB KV and 2+5 iterations',
        'source':'configs/final_protocol.json; preserves earlier cross-GPU measurements'})
    for policy_id in policies:
        output=RUN/f'results/speed_common_gpu3/{policy_id}'
        if (output/'summary.csv').exists():continue
        checkpoint=TEMP/'checkpoints'/policy_id
        owned=not policy_id.startswith('uniform_')
        if owned:checkpoint=TEMP/'common_card_checkpoints'/policy_id
        if not (checkpoint/'model.safetensors').exists():
            if checkpoint.exists():checkpoint.rename(checkpoint.with_name(checkpoint.name+f'_previous_{time.time_ns()}'))
            policy=RUN/f'configs/policies/{policy_id}.json'
            command=[sys.executable,str(RUN/'code/export_model.py'),'--model-path',str(MODEL),
                '--policy-json',str(policy),'--output-dir',str(checkpoint),'--cutlass-wrapper-path',str(WRAPPER)]
            for method in ['sparse_bf16','sparse_nvfp4']:
                if method in policy.read_text():command+=['--canonical-'+method.replace('_','-')+'-state',str(TEMP/'canonical'/method/'model.pt')]
            invoke(command,RUN/f'logs/common_gpu3_{policy_id}_export.log')
        invoke([sys.executable,str(RUN/'code/run_workflow.py'),'speed','--checkpoint',str(checkpoint),
            '--scenario','prefill_decode','--output-dir',str(output),'--vllm-root',str(VLLM),
            '--cutlass-wrapper-path',str(WRAPPER),'--warmup-iters','2','--iters','5','--reuse-llm',
            '--gpu-memory-utilization','.85'],RUN/f'logs/common_gpu3_{policy_id}_speed.log')
        evidence=RUN/f'results/common_speed_exports/{policy_id}'
        evidence.mkdir(parents=True,exist_ok=True)
        for source in checkpoint.glob('*.json'):shutil.copy2(source,evidence/source.name)
        inspect_checkpoint(checkpoint,evidence)
        if owned:
            binary=checkpoint/'model.safetensors';size=binary.stat().st_size;binary.unlink()
            write_json(evidence/'temporary_binary_cleanup.json',{'path':str(binary),'bytes':size,
                'reason':'owned common-card export after successful measurement; all metadata retained',
                'rebuild':'code/export_model.py with frozen policy and canonical states'})
        print(policy_id,'common-card speed complete',flush=True)


if __name__=='__main__':main()
