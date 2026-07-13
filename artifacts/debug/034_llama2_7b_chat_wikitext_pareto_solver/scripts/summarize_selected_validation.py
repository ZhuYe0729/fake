#!/usr/bin/env python3
from __future__ import annotations
import csv,json,statistics
from pathlib import Path
def main():
 root=Path(__file__).resolve().parents[1];rows=[]
 for scenario in ('prefill_only','prefill_decode'):
  selected=json.load(open(root/'validation'/scenario/'selection.json'))
  for p in selected:
   point=str(p['point_index']);nll=list(csv.DictReader(open(root/'validation'/scenario/f'nll_point_{point}.csv')))[0];runs=root/'validation'/scenario/'speed'/f'point_{point}'/'runs'
   if scenario=='prefill_only':
    vals=[json.load(open(runs/f'measured_{i}.json'))['elapsed_ms'] for i in range(5)];speed={'e2e_median_ms':statistics.median(vals),'ttft_median_ms':statistics.median(vals),'tpot_ms':0.}
   else:
    runs=root/'validation'/scenario/'speed_mem08'/f'point_{point}'/'runs'
    one=[json.load(open(runs/f'measured_{i}_o1.json'))['generate_s']*1000 for i in range(10)];full=[json.load(open(runs/f'measured_{i}_o80.json'))['generate_s']*1000 for i in range(10)];speed={'e2e_median_ms':statistics.median(full),'ttft_median_ms':statistics.median(one),'tpot_ms':(statistics.median(full)-statistics.median(one))/79}
   rows.append({'scenario':scenario,'point_index':point,'predicted_quality_cost':p['predicted_quality_cost'],'measured_wikitext_delta_nll':nll['target_delta_nll'],'raw_predicted_linear_ms':p['raw_predicted_linear_ms'],**speed})
 out=root/'validation'/'summary.csv'
 with out.open('w',newline='') as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
 print(out)
if __name__=='__main__':main()
