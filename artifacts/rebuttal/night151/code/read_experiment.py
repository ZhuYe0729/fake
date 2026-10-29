"""Read-only, bounded text access to the current rebuttal experiments."""
import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'runs'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',choices=['016','017','018','019','020','021'])
    parser.add_argument('relative_path')
    parser.add_argument('--tail',type=int)
    args=parser.parse_args()
    base=(ROOT/args.run).resolve()
    path=(base/args.relative_path).resolve()
    assert path.is_relative_to(base), 'Only files within this experiment are readable.'
    assert path.suffix in ['.json','.csv','.md','.py','.log','.txt']
    assert path.is_file()
    if args.tail:
        from collections import deque
        with path.open(errors='replace') as handle:
            text=''.join(deque(handle,maxlen=min(args.tail,100)))
        print(text[-50000:])
    else:
        with path.open('rb') as handle:data=handle.read(250001)
        assert len(data)<=250000, 'Use --tail for a large log; choose a compact report for results.'
        print(data.decode('utf-8'))


if __name__=='__main__':main()
