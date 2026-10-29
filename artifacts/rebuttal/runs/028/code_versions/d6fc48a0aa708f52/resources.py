"""Short GPU leases; yield to the parent run020 formal timing campaign."""
import contextlib
import fcntl
import json
import time

from common import RUN
from telemetry import nvml

ROOT = RUN.parents[1] / 'night151'


def parent_priority():
    parent = RUN.parent / '020/results'
    if (parent / 'formal_complete.json').exists():
        return False
    state = json.loads((parent / 'worker.json').read_text())
    return (parent / 'full_quality_complete.json').exists() or state['stage'] in ['formal_speed', 'report', 'audit']


def busy(gpu):
    return nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(gpu)).used >= 512 * 1024**2


@contextlib.contextmanager
def lease(exclusive=False):
    while True:
        if parent_priority():
            time.sleep(5)
            continue
        phase = (ROOT / 'measurement.lock').open('a')
        try:
            fcntl.flock(phase, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            phase.close()
            time.sleep(5)
            continue
        claim = None
        for gpu in ([0] if exclusive else range(1, nvml.nvmlDeviceGetCount())):
            if busy(gpu):
                continue
            handle = (ROOT / f'gpu_locks/gpu{gpu}.lock').open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            if busy(gpu):
                handle.close()
                continue
            claim = handle
            break
        if claim is not None:
            try:
                yield gpu
            finally:
                claim.close()
                phase.close()
            return
        phase.close()
        time.sleep(5)
