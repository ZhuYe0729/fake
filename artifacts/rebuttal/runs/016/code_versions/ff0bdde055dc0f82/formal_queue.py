"""Three quiet common-card campaigns for every preregistered comparison point."""
import sys
import time

from launch import RUN, checkpoint, cleanup, invoke, read, write


def main():
    assert read(RUN / 'results/confirmation_complete.json')['complete']
    pool = read(RUN / 'configs/allocation_pool.json')
    policies = {pid: {'outputs': [o for s,o in [('prefill_only',1),('prefill_decode',64)] if s in spec['scenarios']]}
        for pid,spec in pool['policies'].items()}
    other = sorted(set(policies)-{'uniform_dense_bf16'})
    rounds = [{'round': n, 'order': ['uniform_dense_bf16',*order]} for n,order in
        enumerate([other,other[::-1],other[7:]+other[:7]])]
    design = {'policies': policies, 'rounds': rounds, 'seed': 144983, 'iterations': 8, 'warmups': 3,
        'speed_inputs_sha256': __import__('hashlib').sha256((RUN / 'configs/speed_inputs.json').read_bytes()).hexdigest()}
    target = RUN / 'configs/speed_design.json'
    if not target.exists():
        write(target, design)
    else:
        assert read(target) == design
    from telemetry import nvml
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    for spec in rounds:
        order = [*spec['order'], 'uniform_dense_bf16']
        for index,pid in enumerate(order):
            name = pid + ('_bracket_end' if index == len(order)-1 else '')
            if (RUN / f'results/speed/node0/round{spec["round"]}/{name}.json').exists():
                continue
            while nvml.nvmlDeviceGetMemoryInfo(handle).used > 512*1024**2:
                time.sleep(10)
            model = checkpoint(pid,0)
            write(RUN / 'results/formal_progress.json', {'status':'running','round':spec['round'],
                'index':index,'total_policies':len(order),'policy':pid})
            invoke([sys.executable,RUN/'code/formal_benchmark.py','--checkpoint',model,'--policy',pid,
                '--round',str(spec['round']),'--record-id',name,'--outputs',*map(str,policies[pid]['outputs'])],
                f'formal_round{spec["round"]}_{name}',0)
            cleanup(model,f'formal_round{spec["round"]}_{name}')
    write(RUN / 'results/formal_complete.json', {'complete':True,'policies':len(policies),'processes':3*(len(policies)+1)})


if __name__ == '__main__':
    main()
