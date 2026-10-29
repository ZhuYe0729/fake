"""Read-only NVML and Linux placement metadata, outside timed regions."""
import importlib.util
import ctypes
import os
import sys
from pathlib import Path

SPEC=importlib.util.spec_from_file_location('rebuttal_nvml','/root/wja/project/my/cospaq/test/vllm/vllm/vllm/third_party/pynvml.py')
nvml=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=nvml;SPEC.loader.exec_module(nvml)
PHYSICAL_GPU=int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
nvml.nvmlInit();HANDLE=nvml.nvmlDeviceGetHandleByIndex(PHYSICAL_GPU)
LIBC=ctypes.CDLL(None)


def gpu():
    cpu=LIBC.sched_getcpu()
    return {'physical_gpu_index':PHYSICAL_GPU,'sm_mhz':nvml.nvmlDeviceGetClockInfo(HANDLE,nvml.NVML_CLOCK_SM),
        'host_cpu':cpu,'host_cpu_khz':int(Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq').read_text()),
        'memory_mhz':nvml.nvmlDeviceGetClockInfo(HANDLE,nvml.NVML_CLOCK_MEM),
        'power_w':nvml.nvmlDeviceGetPowerUsage(HANDLE)/1000,
        'temperature_c':nvml.nvmlDeviceGetTemperature(HANDLE,nvml.NVML_TEMPERATURE_GPU),
        'pstate':nvml.nvmlDeviceGetPerformanceState(HANDLE),
        'pcie_generation':nvml.nvmlDeviceGetCurrPcieLinkGeneration(HANDLE),
        'pcie_width':nvml.nvmlDeviceGetCurrPcieLinkWidth(HANDLE)}


def placement():
    raw=Path('/proc/self/numa_maps').read_text();nodes={};anon={}
    for line in raw.splitlines():
        fields=line.split();page=next((int(x.split('=')[1]) for x in fields if x.startswith('kernelpagesize_kB=')),4)*1024
        for field in fields:
            if field.startswith('N') and '=' in field and field[1:field.index('=')].isdigit():
                node,count=field.split('=');nodes[node]=nodes.get(node,0)+int(count)*page
                if ' anon=' in line or ' heap ' in line:anon[node]=anon.get(node,0)+int(count)*page
    return {'pid':os.getpid(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'resident_bytes_by_node':nodes,'anonymous_bytes_by_node':anon,'numa_maps':raw,
        'process_status':Path('/proc/self/status').read_text()}
