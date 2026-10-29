"""Reuse frozen source engine code, writing measurements only into run022."""
import argparse
import sys
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-run',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--policy',required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--outputs',nargs='+',required=True)
    args=parser.parse_args()
    source=RUN.parent/args.source_run
    sys.path.insert(0,str(source/'code'))
    import benchmark
    benchmark.RUN=RUN/'models'/args.source_run
    sys.argv=[str(__file__),'--checkpoint',args.checkpoint,'--policy',args.policy,
        '--mode','arena_graph','--label',args.label,'--outputs',*args.outputs,'--iterations','8']
    benchmark.main()


if __name__=='__main__':main()
