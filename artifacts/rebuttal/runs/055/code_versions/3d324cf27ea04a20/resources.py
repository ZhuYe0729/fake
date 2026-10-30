"""Read-only NVML resource accounting, with the driver reserve separated."""
from pathlib import Path
import json
import pynvml as nvml

nvml.nvmlInit()


def snapshot():
    rows = []
    for index in range(nvml.nvmlDeviceGetCount()):
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        memory = nvml.nvmlDeviceGetMemoryInfo(handle, version=nvml.nvmlMemory_v2)
        processes = {}
        for kind, function in [('compute', nvml.nvmlDeviceGetComputeRunningProcesses),
                               ('graphics', nvml.nvmlDeviceGetGraphicsRunningProcesses)]:
            for process in function(handle):
                processes[process.pid] = {'pid': process.pid, 'kind': kind,
                    'used_bytes': int(process.usedGpuMemory or 0)}
        rows.append({'index': index, 'allocated_bytes': memory.total - memory.free - memory.reserved,
            'reserved_bytes': memory.reserved, 'free_bytes': memory.free,
            'utilization_percent': nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            'processes': list(processes.values())})
    return rows


def idle(rows):
    baseline = json.loads((Path(__file__).resolve().parents[1] / 'configs/nvml_baseline.json').read_text())
    offsets = {row['index']: row['idle_allocated_bytes'] for row in baseline['gpus']}
    return all(row['allocated_bytes'] - offsets[row['index']] < 128 * 1024**2 and row['utilization_percent'] <= 5
               and not row['processes'] for row in rows)


def descendants(pid):
    found, pending = {pid}, [pid]
    while pending:
        current = pending.pop()
        path = Path(f'/proc/{current}/task/{current}/children')
        if not path.exists():
            continue
        for child in map(int, path.read_text().split()):
            if child not in found:
                found.add(child); pending.append(child)
    return found
