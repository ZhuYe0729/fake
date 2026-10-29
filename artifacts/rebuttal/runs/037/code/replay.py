"""Run unchanged source numeric code with an isolated output root."""
import argparse
import importlib.util
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-run',choices=['034','035'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--policy',required=True);p.add_argument('--kind',choices=['nll','generation'],required=True)
    p.add_argument('--dataset');args=p.parse_args()
    source=Path(__file__).resolve().parents[2]/args.source_run
    sys.path.insert(0,str(source/'code'))
    name='score_nll.py'if args.kind=='nll'else'generate_tasks.py'
    spec=importlib.util.spec_from_file_location('exact_replayed_numeric_module',source/'code'/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.RUN=args.root
    sys.argv=[str(source/'code'/name),'--checkpoint',args.checkpoint,'--policy',args.policy]
    if args.kind=='nll':sys.argv+=['--split','confirmation','--scenarios','prefill_decode']
    else:sys.argv+=['--dataset',args.dataset,'--begin','0','--end','128']
    module.main()


if __name__=='__main__':main()
