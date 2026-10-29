"""Populate the planned matched-allocator table from actual per-point CSV."""
import csv
import re
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]


def main():
    path=BASE/'EXPERIMENT_CHECKLIST.md';text=path.read_text()
    block=re.search(r'(## 表4 — .*?)(?=\n## 表5)',text,re.S)
    lines=block[1].splitlines();data={}
    for model,rid in [('Qwen3','016'),('Llama-3.1','017'),('Llama-2','018')]:
        for row in csv.DictReader((BASE/f'runs/{rid}/results/allocation_points.csv').open()):
            data[model,row['label'],row['scenario']]=row
    changed=0
    for i,line in enumerate(lines):
        if not line.startswith('| '):continue
        cells=[v.strip() for v in line.split('|')[1:-1]]
        if cells[0] not in ['Qwen3','Llama-3.1','Llama-2']:continue
        model,scene,mode,fraction=cells[:4]
        scene='prefill_only' if scene=='Prefill' else 'prefill_decode'
        mode='same_operator' if mode=='同算子' else 'phase_pair'
        labels=[f'{allocator}_{scene}_{mode}_r{float(fraction):g}' for allocator in ['daso','hawq_signed']]
        if any((model,label,scene) not in data for label in labels):continue
        pair=[data[model,label,scene] for label in labels]
        nll='confirmation_delta_nll' if model=='Qwen3' else 'delta_nll'
        ratio='speedup' if model=='Qwen3' else 'e2e_speedup'
        misses='e2e_target_violations' if model=='Qwen3' else 'e2e_target_misses'
        cells[4]=' / '.join(f'{float(p[nll]):.5f}' for p in pair)
        cells[5]=' / '.join(f'{float(p[ratio]):.3f}' if p[ratio] else '待' for p in pair)
        cells[6]=' / '.join(f'{p[misses]}/24' if p[misses] else '0/24' if p['status']=='complete' else '待' for p in pair)
        cells[7]='三轮速度/NLL完成' if all(p['status']=='complete' for p in pair) else 'NLL完成；速度待补'
        lines[i]='| '+' | '.join(cells)+' |';changed+=1
    note='当前数值直接来自各run冻结点CSV；空速度不借用校准结果。配对误差与同候选协议见[12组Llama诊断](night151/LLAMA_DIAGNOSTICS.md)。'
    if note not in lines:lines.insert(2,note+'\n')
    text=text[:block.start()]+'\n'.join(lines)+'\n'+text[block.end():]
    path.write_text(text)
    print('Updated',changed,'matched comparison rows.')


if __name__=='__main__':main()
