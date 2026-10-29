"""Paired dual-resident control after all primary measurements complete."""
import os
import sys

from orchestrate import RUN,TEMP,invoke,read,write


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    assert read(RUN/'results/worker.json')['status']=='complete'
    design=read(RUN/'configs/resident_control.json');state=RUN/'results/resident_worker.json'
    write(state,{'status':'running','pid':os.getpid(),'gpu':0})
    try:
        for spec in design['rounds']:
            for pid in spec['policies']:
                i=spec['round']
                if (RUN/f'results/resident/node0/round{i}/{pid}.json').exists():continue
                invoke([sys.executable,RUN/'code/benchmark.py','--checkpoint',TEMP/'checkpoints'/pid,
                    '--policy',pid,'--round',i,'--node',0,'--resident'],f'resident_round{i}_{pid}',0)
                print('resident complete',i,pid,flush=True)
        write(state,{'status':'complete','pid':os.getpid(),'gpu':0})
    except Exception as e:write(state,{'status':'failed','error':repr(e)});raise


if __name__=='__main__':main()
