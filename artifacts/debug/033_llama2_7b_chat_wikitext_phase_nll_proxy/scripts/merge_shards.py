#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--scenario',required=True);p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--blocks',type=int,default=100);a=p.parse_args();rows=[]
 for x in sorted((a.output_root/'nll/shards'/a.scenario).glob('p*.csv')):
  with x.open(newline='') as f:rows.extend(csv.DictReader(f))
 rows=[r for r in rows if int(r['sample_count'])==a.blocks]
 if len({r['policy_id'] for r in rows})!=72:raise RuntimeError(f'expected 72 {a.blocks}-block shards, got {len(rows)}')
 rows.sort(key=lambda r:r['policy_id']);out=a.output_root/'nll'/f'{a.scenario}.csv';out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='') as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
 print(out)
if __name__=='__main__':main()
