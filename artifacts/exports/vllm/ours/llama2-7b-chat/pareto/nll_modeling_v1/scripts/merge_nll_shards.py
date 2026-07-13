#!/usr/bin/env python3
"""Merge one-policy NLL worker shards after a parallel calibration run."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
def read(p):
    with p.open(newline="") as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scenario",choices=("prefill_only","prefill_decode"),required=True);p.add_argument("--output-root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    rows=[]
    for shard in sorted((a.output_root/"nll"/"shards"/a.scenario).glob("p*.csv")): rows.extend(read(shard))
    ids={r["policy_id"] for r in rows}
    if len(ids)!=30: raise RuntimeError(f"need 30 distinct shards, found {len(ids)}")
    rows=sorted(rows,key=lambda r:r["policy_id"]); out=a.output_root/"nll"/f"{a.scenario}.csv"
    with out.open("w",newline="") as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    print(f"merged {len(rows)} policies into {out}")
if __name__=="__main__":main()
