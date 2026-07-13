#!/usr/bin/env python3
"""Select endpoints, knee, and two intermediate points from predicted curves."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
 for scenario in ('prefill_only','prefill_decode'):
  rows=read(a.root/scenario/'pareto/pareto_points.csv');n=len(rows);indices={0,n-1,round((n-1)*.25),round((n-1)*.75)}
  # Knee maximizes equal normalized quality/speed progress.
  qmax=float(rows[-1]['predicted_quality_cost']);t0=float(rows[0]['raw_predicted_linear_ms']);t1=float(rows[-1]['raw_predicted_linear_ms']);knee=max(range(n),key=lambda i:min(float(rows[i]['predicted_quality_cost'])/max(qmax,1e-12),(t0-float(rows[i]['raw_predicted_linear_ms']))/max(t0-t1,1e-12)))
  indices.add(knee)
  if len(indices)<5:indices.add(round((n-1)*.5))
  selected=[]
  for i in sorted(indices):
   row=dict(rows[i]);row['selection_reason']='knee' if i==knee else 'endpoint' if i in {0,n-1} else 'intermediate';selected.append(row)
  out=a.root/'validation'/scenario;out.mkdir(parents=True,exist_ok=True);(out/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
  with (out/'selection.csv').open('w',newline='') as f:w=csv.DictWriter(f,selected[0].keys());w.writeheader();w.writerows(selected)
  print(scenario,len(selected))
if __name__=='__main__':main()
