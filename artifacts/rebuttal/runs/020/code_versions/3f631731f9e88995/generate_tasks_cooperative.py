"""Coordinate identical generation jobs without touching live partial outputs."""
import fcntl
import hashlib
import importlib.util
import sys
import time
from pathlib import Path

from common import RUN,write_json


def main():
    pid=sys.argv[sys.argv.index('--policy')+1]
    dataset=sys.argv[sys.argv.index('--dataset')+1]
    target=RUN/f'results/generation/{pid}/{dataset}'
    lock=RUN/f'results/generation_locks/{pid}_{dataset}.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        if (target/'metrics_0000_end.json').exists():return
        assert '--begin' not in sys.argv and '--end' not in sys.argv
        stage=RUN/f'results/generation_staging/{pid}_{dataset}_{time.time_ns()}'
        stage.mkdir(parents=True)
        for name in ['data','configs']:(stage/name).symlink_to(RUN/name,target_is_directory=True)
        original=RUN/'code/generate_tasks_impl.py'
        spec=importlib.util.spec_from_file_location('unchanged_generation_body',original)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.RUN=stage
        module.main()
        produced=stage/f'results/generation/{pid}/{dataset}'
        assert (produced/'metrics_0000_end.json').exists()
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            target.rename(target.with_name(dataset+'_previous_'+str(time.time_ns())))
        produced.rename(target)
        write_json(RUN/f'results/generation_relocations/{pid}_{dataset}.json',{
            'original_generator_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
            'staging':str(produced),'published':str(target),
            'metrics_sha256':hashlib.sha256((target/'metrics_0000_end.json').read_bytes()).hexdigest(),
            'scope':'Same generator/arguments/data/weights; isolated output until completion, then atomic directory move. No partial outputs lost.'})


if __name__=='__main__':main()
