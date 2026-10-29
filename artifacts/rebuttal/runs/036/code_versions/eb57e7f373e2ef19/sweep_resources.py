"""Short quality leases, exclusive matched timing, no unrelated process changes."""
import contextlib
import fcntl
import json
import time

from launch import RUN,write
from telemetry import nvml

ROOT=RUN.parents[1]/'night151'


def formal_waiting():
    ready=[]
    for rid in ['034', '035', '036']:
        for name in ['sweep_worker.json','worker.json']:
            path=RUN.parent/rid/'results'/name
            if not path.exists():continue
            state=json.loads(path.read_text())
            if state.get('stage')not in ['formal_speed','waiting_formal_card']:continue
            device=RUN.parent/rid/'configs/sweep_formal_device.json'
            if device.exists():
                gpu=json.loads(device.read_text())['physical_gpu']
                if nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(gpu)).used>=512*1024**2:break
            ready.append(rid);break
    return ready

@contextlib.contextmanager
def lease(exclusive=False):
    while True:
        priority=formal_waiting()
        if (not exclusive and priority) or (exclusive and priority and min(priority)<RUN.name):
            time.sleep(5);continue
        phase=(ROOT/'measurement.lock').open('a')
        try:fcntl.flock(phase,(fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)|fcntl.LOCK_NB)
        except BlockingIOError:
            phase.close();time.sleep(5);continue
        device=RUN/'configs/sweep_formal_device.json'
        cards=[json.loads(device.read_text())['physical_gpu']] if exclusive and device.exists() else (
            range(nvml.nvmlDeviceGetCount()) if exclusive else range(1,nvml.nvmlDeviceGetCount()))
        claim=None
        for gpu in cards:
            handle=nvml.nvmlDeviceGetHandleByIndex(gpu)
            if nvml.nvmlDeviceGetMemoryInfo(handle).used>=512*1024**2:continue
            lock=(ROOT/f'gpu_locks/gpu{gpu}.lock').open('a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close();continue
            if nvml.nvmlDeviceGetMemoryInfo(handle).used>=512*1024**2:
                lock.close();continue
            claim=lock
            if exclusive and not device.exists():
                write(device,{'physical_gpu':gpu,'name':nvml.nvmlDeviceGetName(handle),
                    'cpu_affinity':list(range(gpu*8,gpu*8+8)),'numa_node':0 if gpu<4 else 1,
                    'frozen_at_unix':time.time(),'rule':'First free card before first formal point; fixed for all rounds.'})
            break
        if claim:
            try:yield gpu
            finally:claim.close();phase.close()
            return
        phase.close();time.sleep(5)
