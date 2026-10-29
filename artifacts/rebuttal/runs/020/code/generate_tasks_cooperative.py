"""Coordinate identical generation jobs without touching live partial outputs."""
import fcntl
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

from common import RUN,write_json


def reuse_completed(target):
    metrics=target/'metrics_0000_end.json'
    if not metrics.exists():return False
    answers=target/'answers_0000_end.jsonl'
    if not answers.exists():
        # The already-running old queue checks metrics before partial answers.
        # Publication between those two checks can mislabel completed answers.
        expected=json.loads(metrics.read_text())['samples']
        matches=[p for p in target.glob('answers_failed_*.jsonl')
            if sum(1 for _ in p.open())==expected]
        assert len(matches)==1,(target,matches)
        previous=matches[0]
        digest=hashlib.sha256(previous.read_bytes()).hexdigest()
        previous.rename(answers)
        write_json(RUN/f'results/generation_relocations/answer_name_restore_{time.time_ns()}.json',{
            'previous':str(previous),'restored':str(answers),'sha256':digest,
            'reason':'Completed publication raced the old queue partial-name check; bytes preserved.'})
    return True


def main():
    pid=sys.argv[sys.argv.index('--policy')+1]
    dataset=sys.argv[sys.argv.index('--dataset')+1]
    target=RUN/f'results/generation/{pid}/{dataset}'
    lock=RUN/f'results/generation_locks/{pid}_{dataset}.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        if reuse_completed(target):return
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
