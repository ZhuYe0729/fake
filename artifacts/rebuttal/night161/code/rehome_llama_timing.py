"""Restart only an explicitly scoped own idle campaign on an available card."""
import fcntl
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT.parent/'runs'/os.environ.get('DASO_REHOME_RUN','035');LOCKS=ROOT.parent/'night151'
assert RUN.name in ['035','036']
read=lambda p:json.loads(p.read_text())


def main():
    sys.path.insert(0,str(RUN/'code'))
    from telemetry import nvml
    print('Actual experiment NVML memory MiB',[(gpu,round(nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(gpu)).used/1024**2))
        for gpu in range(nvml.nvmlDeviceGetCount())],flush=True)
    with(LOCKS/'measurement.lock').open('a')as phase:
        try:fcntl.flock(phase,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('Measurement active; controller unchanged.',flush=True);return
        assert not(RUN/'results/sweep_formal_complete.json').exists()
        state=read(RUN/'results/sweep_worker.json');assert state['stage']in['formal_speed','stable_card_recovery']
        if state['stage']=='stable_card_recovery':assert list((RUN/'results').glob('sweep_failure_*.json'))
        previous=read(RUN/'configs/sweep_formal_device.json');old_gpu=previous['physical_gpu']
        used=nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(old_gpu)).used
        failed_idle=False
        with(RUN/'results/sweep_driver.lock').open('a')as probe:
            try:
                fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                failed_idle=bool(list((RUN/'results').glob('sweep_failure_*.json')))
            except BlockingIOError:pass
        if used<512*1024**2 and not failed_idle:print('Frozen GPU became free; leave original campaign.',flush=True);return
        candidates=[int(os.environ['DASO_REHOME_GPU'])]if'DASO_REHOME_GPU'in os.environ else[*range(1,nvml.nvmlDeviceGetCount()),0]
        for gpu in candidates:
            assert 0<=gpu<nvml.nvmlDeviceGetCount()
            if gpu==old_gpu and not failed_idle:continue
            with(LOCKS/f'gpu_locks/gpu{gpu}.lock').open('a')as claim:
                try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                handle=nvml.nvmlDeviceGetHandleByIndex(gpu)
                if nvml.nvmlDeviceGetMemoryInfo(handle).used>=512*1024**2:continue
                lock=RUN/'results/sweep_driver.lock';stat=lock.stat();holders=[]
                for line in Path('/proc/locks').read_text().splitlines():
                    fields=line.split()
                    if len(fields)<6 or fields[1]!='FLOCK':continue
                    major,minor,inode=fields[5].split(':')
                    if(int(major,16),int(minor,16),int(inode))==(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino):holders.append(int(fields[4]))
                assert len(holders)<=1,holders
                pid=holders[0]if holders else None
                if pid is not None:
                    args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');cwd=Path(f'/proc/{pid}/cwd').resolve()
                    assert pid!=os.getpid()and any((cwd/os.fsdecode(a)).resolve()==RUN/'code/sweep_driver.py'for a in args[1:]if a and not a.startswith(b'-')),args
                    os.kill(pid,signal.SIGTERM)
                else:assert list((RUN/'results').glob('sweep_failure_*.json')),'No own failed controller record'
                with lock.open('a')as own:
                    deadline=time.monotonic()+10
                    while True:
                        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                        except BlockingIOError:
                            assert time.monotonic()<deadline;time.sleep(.1)
                    archive=RUN/f'results/interrupted_formal_gpu{old_gpu}_{time.time_ns()}'
                    archive.mkdir();files=list((RUN/'results/measurements').glob('sweep_formal*.json'))
                    for path in files:shutil.move(path,archive/path.name)
                    shutil.copy2(RUN/'configs/sweep_formal_device.json',archive/'original_device.json')
                    shutil.copy2(RUN/'configs/sweep_formal_rounds.json',archive/'original_rounds.json')
                    record={'physical_gpu':gpu,'name':nvml.nvmlDeviceGetName(handle),'cpu_affinity':list(range(gpu*8,gpu*8+8)),
                        'numa_node':0 if gpu<4 else 1,'frozen_at_unix':time.time(),'rule':'Restart all3 matched rounds on free card; previous partial timings retained, no cross-card merge.',
                        'reason':'Own failed idle campaign reset after OOM with other memory present; all partial timing archived.'if failed_idle else'Frozen GPU remains occupied above512MiB idle gate; exact own idle controller stopped under exclusive measurement lock.',
                        'previous_gpu':old_gpu,'previous_gpu_used_bytes':used,'own_idle_controller_pid':pid,'preserved_partial_records':len(files),
                        'archive':str(archive),'quality_and_policy_unchanged':True}
                    (RUN/'configs/sweep_formal_device.json').write_text(json.dumps(record,indent=2)+'\n')
                    (archive/'migration.json').write_text(json.dumps(record,indent=2)+'\n')
                    print(json.dumps(record),flush=True);return
        print('No free card; original controller unchanged.',flush=True)


if __name__=='__main__':main()
