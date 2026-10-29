"""Move only this run's idle timing controller to a free GPU."""
import fcntl
import json
import os
import shutil
import signal
import time

from common import RUN, write_json
from resources import ROOT
from telemetry import nvml


def prepare():
    if (RUN / 'results/formal_complete.json').exists():
        print('Original campaign already complete; no migration.', flush=True)
        return False
    with (ROOT / 'measurement.lock').open('r') as phase:
        try:
            fcntl.flock(phase, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('A measurement is running; leave all processes alone.', flush=True)
            return False
        used = nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(0)).used
        if used < 512 * 1024**2:
            print('GPU0 became free; original controller can proceed.', flush=True)
            return False
        for gpu in range(1, nvml.nvmlDeviceGetCount()):
            with (ROOT / f'gpu_locks/gpu{gpu}.lock').open('r') as claim:
                try:
                    fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                handle = nvml.nvmlDeviceGetHandleByIndex(gpu)
                if nvml.nvmlDeviceGetMemoryInfo(handle).used >= 512 * 1024**2:
                    continue
                lock = RUN / 'results/driver.lock'
                stat = lock.stat()
                holders = []
                for line in open('/proc/locks'):
                    fields = line.split()
                    if len(fields) < 6 or fields[1] != 'FLOCK':
                        continue
                    major, minor, inode = fields[5].split(':')
                    if (int(major,16),int(minor,16),int(inode)) == (os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino):
                        holders.append(int(fields[4]))
                assert len(holders) == 1, holders
                pid = holders[0]
                from pathlib import Path
                args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                cwd = Path(f'/proc/{pid}/cwd').resolve()
                assert pid != os.getpid() and any((cwd / os.fsdecode(arg)).resolve() == RUN / 'code/driver.py'
                    for arg in args[1:] if arg and not arg.startswith(b'-')), args
                state = json.loads((RUN / 'results/worker.json').read_text())
                assert state['stage'] == 'formal_speed' and not (RUN / 'results/formal_complete.json').exists()
                # The exclusive measurement lock proves no GPU job is running
                # through this controller. Stop only its exact lock holder.
                os.kill(pid, signal.SIGTERM)
                with lock.open('r') as driver_lock:
                    deadline = time.monotonic() + 10
                    while True:
                        try:
                            fcntl.flock(driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            assert time.monotonic() < deadline, 'Own controller did not release its lock'
                            time.sleep(.1)
                archive = RUN / f'results/gpu0_interrupted_{time.time_ns()}'
                archive.mkdir()
                (RUN / 'results/measurements').rename(archive / 'measurements')
                shutil.copy2(RUN / 'configs/formal_rounds.json', archive / 'formal_rounds.json')
                device = {'physical_gpu':gpu,'gpu_name':nvml.nvmlDeviceGetName(handle),
                    'cpu_affinity':list(range(gpu*8,gpu*8+8)), 'numa_node':0 if gpu<4 else 1,
                    'reason':'GPU0 stayed busy before remaining measurements; rerun all three matched rounds on a free card.',
                    'gpu0_used_bytes':used,'stopped_own_idle_controller_pid':pid,
                    'archived_gpu0_records':str(archive.relative_to(RUN)),
                    'changed_at_unix':time.time(), 'configuration_and_quality_results_unchanged':True}
                write_json(RUN / 'configs/formal_device.json', device)
                write_json(archive / 'migration.json', device)
                print('Migrated own idle timing campaign:',device,flush=True)
                return True
    print('No free GPU; original controller remains untouched.',flush=True)
    return False
