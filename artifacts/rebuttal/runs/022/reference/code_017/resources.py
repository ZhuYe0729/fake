"""Shared/exclusive measurement phases and per-GPU claims for the two new runs."""
import contextlib
import fcntl
import json
import time
from pathlib import Path

from common import RUN
from telemetry import nvml

ROOT = RUN.parents[1] / 'night151'


@contextlib.contextmanager
def phase(exclusive):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'measurement.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def busy(gpu):
    return nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(gpu)).used >= 512*1024**2


@contextlib.contextmanager
def quality_gpus(maximum=4, minimum=2):
    directory = ROOT / 'gpu_locks'
    directory.mkdir(parents=True, exist_ok=True)
    claims = []
    with phase(False):
        try:
            while len(claims) < minimum:
                reserved = set()
                old = RUN.parent / '016/results'
                if not (old / 'full_quality_complete.json').exists():
                    p = old / 'full_quality_progress.json'
                    if not p.exists():
                        time.sleep(10)
                        continue
                    reserved.update(json.loads(p.read_text()).get('gpus', []))
                for gpu in range(nvml.nvmlDeviceGetCount()):
                    if gpu in reserved or busy(gpu):
                        continue
                    handle = (directory / f'gpu{gpu}.lock').open('a')
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close()
                        continue
                    claims.append((gpu, handle))
                    if len(claims) == maximum:
                        break
                if len(claims) < minimum:
                    for _, handle in claims:
                        handle.close()
                    claims.clear()
                    time.sleep(10)
            yield [gpu for gpu, _ in claims]
        finally:
            for _, handle in claims:
                handle.close()


@contextlib.contextmanager
def speed_gpu():
    while busy(0) or not (RUN.parent / '016/results/full_quality_complete.json').exists():
        time.sleep(10)
    with phase(True):
        with (ROOT / 'gpu_locks/gpu0.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            while busy(0):
                time.sleep(10)
            yield 0
