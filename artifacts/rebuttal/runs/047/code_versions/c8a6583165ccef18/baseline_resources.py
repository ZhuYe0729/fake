"""Yield quality to earlier formal campaigns, retaining per-card admission."""
import contextlib
import fcntl
import json
import os
import time

from launch import RUN,write
from telemetry import nvml

ROOT=RUN.parents[1]/'night151'


def waiting():
    held=set()
    with open('/proc/locks')as handles:
        for line in handles:
            fields=line.split()
            if len(fields)>=6 and fields[1]=='FLOCK':
                major,minor,inode=fields[5].split(':')
                held.add((int(major,16),int(minor,16),int(inode)))
    result=[]
    for rid in ['036','038','039','040','041','042','043']:
        for name in ['sweep_worker.json','worker.json']:
            path=RUN.parent/rid/'results'/name
            if not path.exists():continue
            state=json.loads(path.read_text())
            if state.get('stage')not in ['formal_speed','waiting_formal_card']:continue
            own=RUN.parent/rid/'results'/('sweep_driver.lock'if rid in ['034','035','036','038','039']else'driver.lock')
            if not own.exists():continue
            stat=own.stat()
            if(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino)not in held:continue
            device=RUN.parent/rid/'configs'/('baseline_formal_device.json'if rid in ['042','043']else'sweep_formal_device.json')
            if device.exists():
                gpu=json.loads(device.read_text())['physical_gpu']
                if nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(gpu)).used>=512*1024**2:break
            result.append(rid);break
    return result

@contextlib.contextmanager
def lease(exclusive=False):
    while True:
        priority=waiting()
        if priority and(not exclusive or min(priority)<RUN.name):time.sleep(5);continue
        phase=(ROOT/'measurement.lock').open('a')
        try:fcntl.flock(phase,(fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)|fcntl.LOCK_NB)
        except BlockingIOError:phase.close();time.sleep(5);continue
        path=RUN/'configs/baseline_formal_device.json'
        cards=[json.loads(path.read_text())['physical_gpu']]if exclusive and path.exists()else json.loads((ROOT.parent/'night161/gpu_admission.json').read_text())['cards']
        claim=None
        for gpu in cards:
            handle=nvml.nvmlDeviceGetHandleByIndex(gpu)
            if nvml.nvmlDeviceGetMemoryInfo(handle).used>=512*1024**2:continue
            lock=(ROOT/f'gpu_locks/gpu{gpu}.lock').open('a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lock.close();continue
            if nvml.nvmlDeviceGetMemoryInfo(handle).used>=512*1024**2:lock.close();continue
            claim=lock
            if exclusive and not path.exists():write(path,{'physical_gpu':gpu,'name':nvml.nvmlDeviceGetName(handle),
                'cpu_affinity':list(range(gpu*8,gpu*8+8)),'numa_node':0 if gpu<4 else 1,'frozen_at_unix':time.time()})
            break
        if claim:
            try:yield gpu
            finally:claim.close();phase.close()
            return
        phase.close();time.sleep(5)
