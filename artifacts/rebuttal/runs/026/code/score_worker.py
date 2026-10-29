"""Read frozen source checkpoints; redirect all score/capture output to run026."""
import argparse
import json
import sys
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
SOURCE=RUN.parent/'018'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',required=True)
    parser.add_argument('--split',required=True);args=parser.parse_args()
    sys.path.insert(0,str(SOURCE/'code'))
    import score_nll
    spec=json.loads((SOURCE/'configs/model.json').read_text())
    checkpoint=Path(spec['temp'])/'checkpoints'/args.policy
    assert (checkpoint/'model.safetensors').exists()
    score_nll.RUN=RUN
    sys.argv=[str(__file__),'--checkpoint',str(checkpoint),'--policy',args.policy,
        '--split',args.split,'--scenarios','prefill_decode']
    score_nll.main()


if __name__=='__main__':main()
