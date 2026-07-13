#!/usr/bin/env python3
"""Build measured selected-point curves after all required speed runs finish."""
from __future__ import annotations
import csv,json,statistics
from pathlib import Path
def load_csv(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))
def main():
 root=Path(__file__).resolve().parents[1];all_rows=[]
 for scenario in ('prefill_only','prefill_decode'):
  selected=json.load(open(root/'validation'/scenario/'selection.json'));rows=[]
  for point in selected:
   index=str(point['point_index']);nll=load_csv(root/'validation'/scenario/f'nll_point_{index}.csv')[0]
   speed_root=root/'validation'/scenario/('speed' if scenario=='prefill_only' else 'speed_mem08')/f'point_{index}'/'runs'
   if scenario=='prefill_only':
    files=[speed_root/f'measured_{i}.json' for i in range(5)]
    if not all(x.exists() for x in files):raise RuntimeError(f'missing prefill runs point={index}')
    values=[json.load(open(x))['elapsed_ms'] for x in files];e2e=ttft=statistics.median(values);tpot=0.
   else:
    one=[speed_root/f'measured_{i}_o1.json' for i in range(10)];full=[speed_root/f'measured_{i}_o80.json' for i in range(10)]
    if not all(x.exists() for x in one+full):raise RuntimeError(f'missing decode runs point={index}')
    ttft=1000*statistics.median(json.load(open(x))['generate_s'] for x in one);e2e=1000*statistics.median(json.load(open(x))['generate_s'] for x in full);tpot=(e2e-ttft)/79
   rows.append({'scenario':scenario,'point_index':index,'selection_reason':point['selection_reason'],'predicted_quality_cost':point['predicted_quality_cost'],'measured_wikitext_delta_nll':nll['target_delta_nll'],'raw_predicted_linear_ms':point['raw_predicted_linear_ms'],'measured_e2e_median_ms':e2e,'measured_ttft_median_ms':ttft,'measured_tpot_ms':tpot,'speed_gpu_memory_utilization':.9 if scenario=='prefill_only' else .8})
  rows.sort(key=lambda r:float(r['measured_wikitext_delta_nll']));best=float('inf')
  for r in rows:
   r['measured_pareto_kept']=float(r['measured_e2e_median_ms'])<best
   if r['measured_pareto_kept']:best=float(r['measured_e2e_median_ms'])
  all_rows.extend(rows)
 out=root/'validation'/'measured_selected_frontier.csv'
 with out.open('w',newline='') as f:w=csv.DictWriter(f,all_rows[0].keys());w.writeheader();w.writerows(all_rows)
 print(out)
if __name__=='__main__':main()
