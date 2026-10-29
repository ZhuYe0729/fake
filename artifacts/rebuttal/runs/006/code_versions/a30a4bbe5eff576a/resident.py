"""Experiment-local dual-resident control; no shared runtime source changes."""
import torch


def install(phase):
    phase._RELEASE_PREFILL_ENABLED=False
    for layer in phase._LAYERS:
        if layer.ph_prefill_method==layer.ph_decode_method:
            phase._reuse_prefill_for_decode(layer)
        else:
            phase._materialize_decode_from_cpu(layer,non_blocking=False)
            layer.ph_decode_event=torch.cuda.Event()
            layer.ph_decode_event.record(torch.cuda.current_stream())
    torch.cuda.synchronize()

    def restore():
        phase._PHASE='prefill';phase._SCHEDULED.clear();phase._PREFILL_READY_EVENT=None
        for layer in phase._LAYERS:
            layer.ph_decode_scheduled=False;layer.ph_decode_waited=True
            layer.ph_prefill_waited=True;layer.ph_prefill_released=False

    def schedule(layer):
        if not phase._ENABLED or layer.ph_decode_scheduled:return
        layer.ph_decode_scheduled=True;phase._SCHEDULED.add(id(layer));phase._maybe_enter_decode()

    phase.prepare_next_prefill=restore
    phase.wait_for_prefill_ready=lambda:None
    phase._schedule_decode_copy=schedule
    restore()
    # Keep host tensors here to isolate GPU transfer/scheduling from host-storage
    # policy. They are counted explicitly rather than hidden by deleting caches.
