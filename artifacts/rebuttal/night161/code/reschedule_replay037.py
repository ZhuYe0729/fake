"""Restart only the own waiting replay controller after removing an irrelevant gate."""
import fcntl
import json
import os
import signal
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT.parent/'runs/037'


def main():
    state=json.loads((RUN/'results/worker.json').read_text())
    assert state['stage']=='original_sweep_audits'
    assert not(RUN.parent/'036/results/sweep_final_audit.json').exists()
    assert not list((RUN/'results/completed').glob('*.json'))
    lock=RUN/'results/driver.lock';stat=lock.stat();holders=[]
    for line in Path('/proc/locks').read_text().splitlines():
        fields=line.split()
        if len(fields)<6 or fields[1]!='FLOCK':continue
        major,minor,inode=fields[5].split(':')
        if(int(major,16),int(minor,16),int(inode))==(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino):holders.append(int(fields[4]))
    assert len(holders)==1,holders
    pid=holders[0];args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');cwd=Path(f'/proc/{pid}/cwd').resolve()
    assert pid!=os.getpid()and any((cwd/os.fsdecode(a)).resolve()==RUN/'code/driver.py'for a in args[1:]if a and not a.startswith(b'-')),args
    children=Path(f'/proc/{pid}/task/{pid}/children').read_text().strip();assert not children,children
    os.kill(pid,signal.SIGTERM)
    with lock.open('a')as own:
        deadline=time.monotonic()+10
        while True:
            try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:assert time.monotonic()<deadline;time.sleep(.1)
        record={'stopped_own_waiting_pid':pid,'children':[],'changed_at_unix':time.time(),
            'reason':'Only034/035 exact baseline caches are replay inputs. Optional036 speed is not a numerical prerequisite; quality/generation protocols unchanged.',
            'source_audits_required':['034','035'],'source_weights_changed':False,'gpu_admission_file':str(ROOT/'gpu_admission.json')}
        (RUN/'reference/rescheduling.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
