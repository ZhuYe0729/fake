"""Descriptive all-variant quality pairs and completed-campaign speed snapshot."""
import csv
import datetime
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT.parent / 'runs/016'


def read(path):
    return json.loads(path.read_text())


def quality(pid, scene):
    return read(RUN / f'results/confirmation/{pid}/{scene}.json')['blocks']


def speed(pid, output, rounds):
    ratios = []
    for number in rounds:
        base = RUN / f'results/speed/node0/round{number}'
        def samples(name):
            return [s['e2e_ms'] for c in read(base / f'{name}.json')['cases'] if c['output']==output
                    for s in c['samples'] if not s['warmup']]
        anchors = samples('uniform_dense_bf16')+samples('uniform_dense_bf16_bracket_end')
        measured = anchors if pid=='uniform_dense_bf16' else samples(pid)
        ratios.append(statistics.median(anchors)/statistics.median(measured))
    return statistics.median(ratios) if ratios else None


def main():
    pool = read(RUN / 'configs/allocation_pool.json')
    lookup = {(p.get('allocator'),p['scenario'],p.get('phase_mode'),p.get('latency_fraction')):p
              for p in pool['points'] if p['policy_id']}
    rows = []
    for point in pool['points']:
        if point['family'] != 'daso matched':
            continue
        for variant in ['hawq_signed','hawq_clip_zero','hawq_absolute']:
            hawq = lookup.get((variant,point['scenario'],point['phase_mode'],point['latency_fraction']))
            if not hawq:
                continue
            diff = [x['avg_nll']-y['avg_nll'] for x,y in zip(quality(point['policy_id'],point['scenario']),
                quality(hawq['policy_id'],hawq['scenario']),strict=True)]
            mean = statistics.mean(diff)
            two = 2*statistics.stdev(diff)/math.sqrt(len(diff))
            rows.append({'scenario':point['scenario'],'phase_mode':point['phase_mode'],
                'r':point['latency_fraction'],'hawq_variant':variant,'daso_minus_hawq_nll':mean,
                'paired_2se':two,'lower':mean-two,'upper':mean+two,
                'scope':'Descriptive paired windows; no multiple-comparison-adjusted significance claim; every frozen point retained.'})
    with (ROOT / 'qwen016_paired_nll.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rounds = [n for n in range(3) if (RUN / f'results/speed/node0/round{n}/uniform_dense_bf16_bracket_end.json').exists()]
    measured=[]
    for point in read(RUN / 'configs/selected_policies.json')['selected']:
        scene,pid = point['scenario'],point['policy']
        diff = [a['avg_nll']-b['avg_nll'] for a,b in zip(quality(pid,scene),quality('uniform_dense_bf16',scene),strict=True)]
        measured.append({'label':point['label'],'policy':pid,'delta_nll':statistics.mean(diff),
            'paired_2se':2*statistics.stdev(diff)/math.sqrt(len(diff)),
            'completed_speed_rounds':len(rounds),'e2e_speedup':speed(pid,1 if scene=='prefill_only' else 64,rounds)})
    for point in pool['points']:
        if point['family']=='hardware anchor' and point['scenario']=='prefill_decode' and point.get('phase_mode')=='phase_pair':
            measured.append({'label':'shared hardware minimum PD phase pair','policy':point['policy_id'],
                'completed_speed_rounds':len(rounds),'e2e_speedup':speed(point['policy_id'],64,rounds)})
    target=ROOT/'qwen016_preliminary_snapshot.json'
    if target.exists():
        previous=ROOT/'snapshot_history'/target.name
        previous.parent.mkdir(exist_ok=True)
        previous=previous.with_name(str(target.stat().st_mtime_ns)+'_'+target.name)
        target.rename(previous)
    target.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Completed campaigns only, full-task results pending; diagnostic does not change any frozen policy.',
        'rows':measured},indent=2)+'\n')
    print('Completed campaigns',rounds,flush=True)
    for row in measured:
        print(row,flush=True)
    print('Central paired diagnostics',flush=True)
    for row in rows:
        if row['r']==.8:
            print({k:v for k,v in row.items() if k!='scope'},flush=True)


if __name__ == '__main__':
    main()
