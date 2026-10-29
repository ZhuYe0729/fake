"""Replay allocation into a temporary tree, checking the frozen policies."""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path

from common import RUN, write_json
import design_policies


def main():
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='qwen3_workload_allocation_') as root:
        parent=Path(root);target=parent/'003'
        (target/'configs').mkdir(parents=True);(target/'results').mkdir()
        for name in ['001','002']:(parent/name).symlink_to(RUN.parent/name)
        for name in ['data','reference']:(target/name).symlink_to(RUN/name)
        (target/'configs/design.json').symlink_to(RUN/'configs/design.json')
        (target/'results/action_profile.csv').symlink_to(RUN/'results/action_profile.csv')
        design_policies.RUN=target;sys.argv=[sys.argv[0]]
        with contextlib.redirect_stdout(io.StringIO()):design_policies.main()
        actual=json.loads((target/'configs/evaluation_points.json').read_text())
        expected=json.loads((RUN/'configs/evaluation_points.json').read_text())
        assert actual==expected
        for pid in expected['policies']:
            assert (target/f'configs/policies/{pid}.json').read_bytes()==(RUN/f'configs/policies/{pid}.json').read_bytes()
        seconds=time.perf_counter()-started
    write_json(RUN/'results/allocation_replay_audit.json',{'passed':True,'policies':len(expected['policies']),
        'points':len(expected['points']),'replay_seconds':seconds,
        'scope':'CPU search replay with frozen costs and quality models; no test evaluation used',
        'evaluation_points_sha256':hashlib.sha256((RUN/'configs/evaluation_points.json').read_bytes()).hexdigest()})
    print(json.dumps({'passed':True,'seconds':seconds}))


if __name__=='__main__':main()
