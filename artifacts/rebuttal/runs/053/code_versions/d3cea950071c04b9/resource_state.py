"""Use allocated memory, excluding NVML's separately reported system reserve."""
from telemetry import nvml

def memory(handle):
 value=nvml.nvmlDeviceGetMemoryInfo(handle,version=nvml.nvmlMemory_v2)
 used=value.total-value.free-value.reserved
 assert 0<=used<=value.total
 return {'used_bytes':used,'reserved_bytes':value.reserved,'total_bytes':value.total,
  'free_bytes':value.free,'nvml_v2_reported_used_bytes':value.used}
