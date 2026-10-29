"""Decompose selected-policy prediction error and calibration-to-test shift."""
import json
import shutil
import sys

from baseline_worker import TEMP,invoke
from checkpoint_evidence import inspect_checkpoint
from common import MODEL,RUN,VLLM,WRAPPER,write_json


def main():
    scenario='prefill_only'
    points=json.loads((RUN/f'results/daso/{scenario}/predicted_points.json').read_text())
    selected=json.loads((RUN/f'configs/selected_daso_{scenario}.json').read_text())['policies']
    policies=list(dict.fromkeys([points[0]['policy_id'],*selected]))
    reference=json.loads((RUN/f'results/calibration/{scenario}/cal_{scenario}_p00.json').read_text())['avg_nll']
    test_reference=json.loads((RUN/'results/nll/uniform_dense_bf16_test_prefill.json').read_text())['avg_nll']
    rows=[]
    for policy_id in policies:
        output=RUN/f'results/selected_policy_calibration/{policy_id}'
        target=output/'nll.json'
        if not target.exists():
            checkpoint=TEMP/'quality_audit_checkpoints'/policy_id
            policy=RUN/f'configs/policies/{policy_id}.json'
            command=[sys.executable,str(RUN/'code/export_model.py'),'--model-path',str(MODEL),
                '--policy-json',str(policy),'--output-dir',str(checkpoint),'--cutlass-wrapper-path',str(WRAPPER)]
            for method in ['sparse_bf16','sparse_nvfp4']:
                if method in policy.read_text():command+=['--canonical-'+method.replace('_','-')+'-state',str(TEMP/'canonical'/method/'model.pt')]
            invoke(command,RUN/f'logs/quality_audit_{policy_id}_export.log')
            invoke([sys.executable,str(RUN/'code/run_workflow.py'),'prefill_nll','--checkpoint',str(checkpoint),
                '--tokenizer',str(MODEL),'--samples',str(RUN/'data/calibration_train_100_2048.pt'),
                '--output',str(target),'--label',policy_id,'--phase-hetero','--policy-json',str(policy),
                '--blocks','100','--vllm-root',str(VLLM),'--cutlass-wrapper-path',str(WRAPPER)],
                RUN/f'logs/quality_audit_{policy_id}_nll.log')
            evidence=output/'export';evidence.mkdir(parents=True,exist_ok=True)
            for source in checkpoint.glob('*.json'):shutil.copy2(source,evidence/source.name)
            inspect_checkpoint(checkpoint,evidence)
            binary=checkpoint/'model.safetensors';size=binary.stat().st_size;binary.unlink()
            write_json(evidence/'temporary_binary_cleanup.json',{'path':str(binary),'bytes':size,
                'reason':'owned diagnostic export after successful calibration-text NLL; all metadata retained'})
        result=json.loads(target.read_text());assert result['token_count']==204700
        point=next(point for point in points if point['policy_id']==policy_id)
        calibration_delta=result['avg_nll']-reference
        test_delta=json.loads((RUN/f'results/nll/{policy_id}_test_prefill.json').read_text())['avg_nll']-test_reference
        rows.append({'policy_id':policy_id,'budget':point['quality_budget'],
            'predicted_delta_nll':point['predicted_delta_nll'],'measured_calibration_delta_nll':calibration_delta,
            'measured_test_delta_nll':test_delta,
            'prediction_error_on_calibration':calibration_delta-point['predicted_delta_nll'],
            'calibration_to_test_shift':test_delta-calibration_delta})
        write_json(RUN/'results/selected_policy_quality_audit.json',{'rule':'zero-budget plus the three predeclared full-task representatives',
            'scope':'diagnostic evaluation only; fitted model and selected policies remain the frozen originals','rows':rows})
        print(rows[-1],flush=True)


if __name__=='__main__':main()
