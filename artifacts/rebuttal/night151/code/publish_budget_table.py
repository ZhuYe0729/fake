"""Join prospective budget choices with core/additional formal speed campaigns."""
import argparse
import csv
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT.parent/'runs'


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def core_speed(rid,pid,scene):
    output=1 if scene=='prefill_only' else 64
    ratios=[]
    for number in range(3):
        folder=RUNS/f'{rid}/results/speed/node0/round{number}'
        measured=[s for c in read(folder/f'{pid}.json')['cases'] if c['output']==output for s in c['samples'] if not s['warmup']]
        anchors=[s for label in ['uniform_dense_bf16','uniform_dense_bf16_bracket_end']
                 for c in read(folder/f'{label}.json')['cases'] if c['output']==output for s in c['samples'] if not s['warmup']]
        assert len(measured)==8 and len(anchors)==16
        ratios.append(1. if pid=='uniform_dense_bf16' else statistics.median(s['e2e_ms'] for s in anchors)/statistics.median(s['e2e_ms'] for s in measured))
    return statistics.median(ratios)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    for rid in ['021','022']:
        path=RUNS/f'{rid}/results/final_audit.json'
        while not path.exists():
            if not args.wait:raise RuntimeError(f'Waiting for run{rid}')
            time.sleep(10)
        assert read(path)['verified']
    rows=list(csv.DictReader((RUNS/'021/results/budget_points.csv').open()))
    extra={(r['source_run'],r['policy'],r['scenario']):r
        for r in csv.DictReader((RUNS/'022/results/budget_speed.csv').open())}
    for row in rows:
        rid,pid,scene=row['source_run'],row['policy'],row['scenario']
        primary=read(RUNS/f'{rid}/configs/primary_policies.json')['policies']
        if pid in primary:
            row['e2e_speedup']=core_speed(rid,pid,scene);row['speed_campaign']=rid
        else:
            row['e2e_speedup']=float(extra[rid,pid,scene]['e2e_speedup']);row['speed_campaign']='022/'+rid
    assert len(rows)==20
    with (ROOT/'llama_five_budget_complete.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 原Llama五档质量预算与正式速度（R1 AK6a / R5 gkP9）','',
        '021规则与代码在任何017/018 confirmation前固定；同一冻结候选池，按validation ΔNLL+2SE+0.005筛选后取校准E2E最快点。'
        'BF16保护值为0。预算共享相同确认样本，重复选中的配置不是新独立实验。','',
        '正式速度来自017/018核心campaign或022补充campaign，各自使用同模型/同卡BF16首尾锚点归一化；'
        '全部GPU0/NUMA0、同输入B8/S2048/O1或64。跨campaign汇总明确保留来源，不伪称20行同轮。','',
        '| Run | Scene | ε | Policy | Validation guard | Test ΔNLL ±2SE | E2E × | Mean exceeds ε | Campaign |',
        '| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |']
    for row in rows:
        lines.append(f'| {row["source_run"]} | {row["scenario"]} | {float(row["quality_budget"]):.3f} | {row["policy"]} | '
            f'{float(row["guarded_delta_nll"]):.5f} | {float(row["confirmation_delta_nll"]):.5f} ± {2*float(row["confirmation_se"]):.5f} | '
            f'{row["e2e_speedup"]:.3f} | {row["mean_budget_violation"]} | {row["speed_campaign"]} |')
    lines+=['','NLL预算不保证MC、ROUGE或BLEU；完整任务优先覆盖主表ε=.01/.04。经验保护和±2SE不构成任意分布泛化或多重比较保证。','',
        '[完整CSV](llama_five_budget_complete.csv) · [预登记选择及审计](../../../runs/021/README.md) · [补充速度](../../../runs/022/README.md)','']
    for reviewer in ['reviewer1','reviewer5']:
        out=ROOT.parent/'responses'/reviewer/'five_budget_llama_night151';out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'llama_five_budget_complete.csv',out/'llama_five_budget_complete.csv')
        (out/'README.md').write_text('\n'.join(lines))
    (ROOT/'budget_packaging_audit.json').write_text(json.dumps({'rows':20,'sources_sha256':{
        rid:sha(RUNS/f'{rid}/results/final_audit.json') for rid in ['017','018','021','022']},
        'csv_sha256':sha(ROOT/'llama_five_budget_complete.csv')},indent=2)+'\n')
    print('Published 20 complete prospective budget points.',flush=True)


if __name__=='__main__':main()
