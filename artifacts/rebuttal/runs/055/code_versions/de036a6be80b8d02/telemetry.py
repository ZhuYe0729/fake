"""Read-only GPU and CPU placement metadata for run055."""
import ctypes
import os
from pathlib import Path

import pynvml as nvml

PHYSICAL_GPU = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
nvml.nvmlInit()
HANDLE = nvml.nvmlDeviceGetHandleByIndex(PHYSICAL_GPU)
LIBC = ctypes.CDLL(None)


def gpu():
    return {'physical_gpu_index': PHYSICAL_GPU, 'host_cpu': LIBC.sched_getcpu(),
            'sm_mhz': nvml.nvmlDeviceGetClockInfo(HANDLE, nvml.NVML_CLOCK_SM),
            'memory_mhz': nvml.nvmlDeviceGetClockInfo(HANDLE, nvml.NVML_CLOCK_MEM),
            'used_bytes': nvml.nvmlDeviceGetMemoryInfo(HANDLE).used,
            'gpu_utilization_percent': nvml.nvmlDeviceGetUtilizationRates(HANDLE).gpu,
            'power_w': nvml.nvmlDeviceGetPowerUsage(HANDLE) / 1000,
            'temperature_c': nvml.nvmlDeviceGetTemperature(HANDLE, nvml.NVML_TEMPERATURE_GPU),
            'other_gpus_sampled': [{'index': index,
                'used_bytes': nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(index)).used,
                'gpu_utilization_percent': nvml.nvmlDeviceGetUtilizationRates(nvml.nvmlDeviceGetHandleByIndex(index)).gpu}
                for index in range(nvml.nvmlDeviceGetCount()) if index != PHYSICAL_GPU]}


def placement():
    return {'pid': os.getpid(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'numa_maps': Path('/proc/self/numa_maps').read_text(),
            'process_status': Path('/proc/self/status').read_text()}
