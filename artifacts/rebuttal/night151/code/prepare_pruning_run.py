"""Freeze Llama-3.1 pruning controls with original calibration specifications."""
import hashlib
import json
import shutil
from pathlib import Path

REPO=Path(__file__).resolve().parents[4]
ROOT=REPO/'artifacts/rebuttal'
RUN=ROOT/'runs/020'
SOURCE=ROOT/'runs/017'


def read(path):return json.loads(path.read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    assert not (RUN/'configs/design.json').exists()
    for name in ['code','configs/policies','data','reference','logs','results']:(RUN/name).mkdir(parents=True,exist_ok=True)
    files=['common.py','arena_runtime.py','eightbit.py','export_model.py','graph_bindings.py','int4_marlin.py',
        'runtime_extensions.py','telemetry.py','launch.py','resources.py','admission_check.py','score_nll.py',
        'benchmark.py','run_workflow.py','evaluate_prefill_tasks_batch8.py','generate_tasks.py','job_queue.py']
    for name in files:shutil.copy2(SOURCE/'code'/name,RUN/'code'/name)
    for directory in ['csrc','compat']:shutil.copytree(SOURCE/'code'/directory,RUN/'code'/directory)
    (RUN/'extensions').symlink_to((SOURCE/'extensions').resolve(),target_is_directory=True)
    for name in ['activation_scales.pt','likelihood_datasets','likelihood_auxiliary_datasets','generation_cnn_dm_1000.jsonl','generation_dsum.jsonl','generation_IWSLT.jsonl']:
        (RUN/'data'/name).symlink_to((SOURCE/'data'/name).resolve())
    model=read(SOURCE/'configs/model.json');model['run']='020'
    model['temp']='/dev/shm/daso_rebuttal_020_20261026'
    write(RUN/'configs/model.json',model)
    design=read(SOURCE/'configs/frozen_design.json')
    write(RUN/'configs/frozen_design.json',{'datasets':{'confirmation':design['datasets']['confirmation']},
        'scope':'Fixed pruning controls only; no fitting, validation or outcome-based selection.'})
    write(RUN/'configs/selected_policies.json',{'scope':'No selection; all pruning methods frozen before outcomes.'})
    template=read(SOURCE/'configs/policies/uniform_dense_bf16.json')
    write(RUN/'configs/policies/uniform_dense_bf16.json',template)
    for method in ['wanda','magnitude']:
        pid='uniform_'+method
        policy={**template,'policy_id':pid,'method_map':{n:{'prefill_method':'sparse_bf16','decode_method':'sparse_bf16'} for n in template['method_map']},
            'canonical_states':{'sparse_bf16':str(RUN/f'data/canonical/{method}/model.pt')}}
        write(RUN/f'configs/policies/{pid}.json',policy)
    for name in ['speed_inputs.json','generation_protocol.json','full_task_protocol.json']:shutil.copy2(SOURCE/'configs'/name,RUN/'configs'/name)
    for kind in ['confirmation','full_quality']:
        write(RUN/f'configs/{kind}_jobs.json',[
            {'job_id':method,'policy':'uniform_'+method,
                **({'scenarios':['prefill_only','prefill_decode']} if kind=='confirmation' else {'kinds':['tasks','generation']})}
            for method in ['wanda','magnitude']])
    for name in ['original_sg_128_2048.pt','original_sg_128_2048.json']:
        shutil.copy2(ROOT/'night151/pruning_calibration'/('017_'+name),RUN/'data'/name)
    protocol=read(RUN/'configs/full_task_protocol.json');protocol['run']='020'
    write(RUN/'configs/full_task_protocol.json',protocol)
    source=ROOT/'runs/007/code/prepare_pruning.py'
    code=source.read_text().replace('Qwen3 adaptation','Llama-3.1 adaptation')
    code=code.replace("RUN/'data/calibration_128_2112.pt'","RUN/'data/original_sg_128_2048.pt'")
    code=code.replace("Path('/dev/shm/daso_rebuttal_007_20261025/canonical')","(RUN/'data/canonical')")
    code=code.replace('Qwen3 layer kwargs','Llama-3.1 layer kwargs')
    (RUN/'code/prepare_pruning.py').write_text(code)
    shutil.copytree(ROOT/'runs/007/reference/wanda',RUN/'reference/wanda')
    path=RUN/'code/launch.py';text=path.read_text()
    old="BASELINE = TEMP / 'checkpoints/uniform_dense_bf16'"
    assert text.count(old)==1
    text=text.replace(old,"BASELINE = Path('/dev/shm/daso_rebuttal_017_20261026/checkpoints/uniform_dense_bf16')")
    old="Path(SPEC['canonical_root']) / method / 'model.pt'"
    assert text.count(old)==1
    text=text.replace(old,"Path(read(RUN / f'configs/policies/{pid}.json')['canonical_states'][method])")
    path.write_text(text)
    path=RUN/'code/job_queue.py'
    path.write_text(path.read_text().replace('with quality_gpus() as available:','with quality_gpus(maximum=2) as available:'))
    write(RUN/'configs/design.json',{'source_run':'017','model_id':model['model_id'],'methods':['wanda','magnitude'],
        'pattern':'dense_2_4','calibration':read(RUN/'data/original_sg_128_2048.json'),
        'source_pruning_code_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'rounds':[['uniform_dense_bf16','uniform_wanda','uniform_magnitude','uniform_dense_bf16_bracket_end'],
                  ['uniform_dense_bf16','uniform_magnitude','uniform_wanda','uniform_dense_bf16_bracket_end'],
                  ['uniform_dense_bf16','uniform_wanda','uniform_magnitude','uniform_dense_bf16_bracket_end']],
        'scope':'Original SparseGPT baseline retained from run017. Fixed Wanda/magnitude controls use the original specified 128x2048 train contexts, 2:4 pattern and SparseBF16 runtime. '
                'Wanda propagates prior pruned layers. All NLL/full tasks and three speed campaigns fresh; no outcome-based selection.'})
    print('Frozen Llama-3.1 pruning controls; Llama2 discrepancy remains separate',flush=True)


if __name__=='__main__':main()
