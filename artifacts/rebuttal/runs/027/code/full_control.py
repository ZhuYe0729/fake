"""Refresh the fixed-prefill SQ-INT8 full control after observed pilot drift."""
import sys
import time
from driver import RUN,invoke,write


def state(status,**details):
    write(RUN/'results/full_control_worker.json',{'status':status,**details})


def main():
    from resources import quality_gpus
    pid='uniform_sq_int8';model=RUN/f'data/checkpoints/{pid}'
    assert (model/'model.safetensors').exists()
    state('waiting',stage='idle_gpu')
    with quality_gpus(maximum=1,minimum=1) as cards:
        gpu=cards[0]
        for dataset in ['cnn_dm_1000','dsum','IWSLT']:
            target=RUN/f'full/results/generation/{pid}/{dataset}'
            if (target/'metrics_0000_end.json').exists():continue
            partial=target/'answers_0000_end.jsonl'
            if partial.exists():partial.rename(partial.with_name('answers_failed_'+str(time.time_ns())+'.jsonl'))
            state('running',stage='full_control_generation',gpu=gpu,dataset=dataset)
            invoke([sys.executable,RUN/'code/generate_full.py','--policy',pid,
                '--checkpoint',model,'--dataset',dataset],'full_control_'+dataset,gpu)
    state('complete',stage='full_control_complete',tasks=3)


if __name__=='__main__':
    try:main()
    except Exception as error:
        state('failed',error=repr(error));raise
