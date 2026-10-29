"""Stable per-layer GPU addresses with asynchronous phase reuse, not dual residency."""
import torch
import time


class ArenaRuntime:
    def __init__(self, phase):
        self.phase = phase
        self.active = False
        self.capturing_decode = False
        self.scheduled = set()
        self.ready = False
        self.prefill_ready = None
        self.decode_ready = torch.cuda.Event()
        self.copy_stream = torch.cuda.Stream()
        self.arena_bytes = 0
        self.transfer_bytes = {'prefill': 0, 'decode': 0}
        self.generations = 0
        self.capture_model_ms = 0.
        self.install()

    @staticmethod
    def layout(tensors, arena=None):
        size = 0
        views = {}
        for name, tensor in tensors.items():
            offset = (size + 255) // 256 * 256
            size = offset + tensor.numel() * tensor.element_size()
            if arena is not None:
                views[name] = arena[offset:size].view(tensor.dtype).view(tensor.shape)
        return size, views

    def initialize_layer(self, layer):
        if layer.ph_prefill_method == layer.ph_decode_method:
            self.phase._reuse_prefill_for_decode(layer)
            return
        pref = {n: getattr(layer, 'ph_prefill_' + n) for n in layer.ph_prefill_adapter.names}
        dec = layer.ph_decode_cpu
        size = max(self.layout(pref)[0], self.layout(dec)[0])
        arena = torch.empty(size, dtype=torch.uint8, device=next(iter(pref.values())).device)
        self.arena_bytes += size
        layer.ph_stable_arena = arena
        for name, view in self.layout(pref, arena)[1].items():
            parameter = getattr(layer, 'ph_prefill_' + name)
            view.copy_(parameter.data)
            parameter.data = view
        for name, view in self.layout(dec, arena)[1].items():
            setattr(layer, 'ph_decode_' + name + '_gpu', view)
        layer.ph_decode_adapter.finalize_decode(layer)
        for name in layer.ph_decode_adapter.names:
            assert getattr(layer, 'ph_decode_' + name + '_gpu').untyped_storage().data_ptr() == arena.data_ptr()
        layer.ph_arena_prefill_done = torch.cuda.Event()
        self.transfer_bytes['prefill'] += sum(t.numel() * t.element_size() for t in layer.ph_prefill_cpu.values())
        self.transfer_bytes['decode'] += sum(t.numel() * t.element_size() for t in dec.values())

    def copy_phase(self, phase):
        for layer in self.phase._LAYERS:
            if layer.ph_prefill_method == layer.ph_decode_method:
                continue
            adapter = getattr(layer, 'ph_' + phase + '_adapter')
            for name in adapter.names:
                suffix = '_gpu' if phase == 'decode' else ''
                destination = getattr(layer, 'ph_' + phase + '_' + name + suffix)
                source = getattr(layer, 'ph_' + phase + '_cpu')[name]
                destination.copy_(source.reshape(destination.shape), non_blocking=True)

    def schedule(self, layer):
        if id(layer) in self.scheduled:
            return
        if layer.ph_prefill_method != layer.ph_decode_method:
            layer.ph_arena_prefill_done.record(torch.cuda.current_stream())
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(layer.ph_arena_prefill_done)
                for name in layer.ph_decode_adapter.names:
                    destination = getattr(layer, 'ph_decode_' + name + '_gpu')
                    destination.copy_(layer.ph_decode_cpu[name].reshape(destination.shape), non_blocking=True)
        self.scheduled.add(id(layer))
        if len(self.scheduled) == len(self.phase._LAYERS):
            self.decode_ready.record(self.copy_stream)
            self.phase._PHASE = 'decode'
            self.ready = False

    def wait_decode(self):
        if self.active and self.phase._PHASE == 'decode' and not self.ready:
            torch.cuda.current_stream().wait_event(self.decode_ready)
            self.ready = True

    def restore(self):
        # Called only after a generation has returned, matching the old protocol.
        torch.cuda.synchronize()
        with torch.cuda.stream(self.copy_stream):
            self.copy_phase('prefill')
            self.prefill_ready = torch.cuda.Event()
            self.prefill_ready.record(self.copy_stream)
        self.phase._PHASE = 'prefill'
        self.scheduled.clear()
        self.ready = False
        self.generations += 1

    def wait_prefill(self):
        if self.prefill_ready is not None:
            torch.cuda.current_stream().wait_event(self.prefill_ready)

    def activate(self):
        self.active = True
        self.phase._ENABLED = True
        self.phase._PHASE = 'prefill'

    def addresses(self):
        return {layer.prefix: {
            'prefill': {n: getattr(layer, 'ph_prefill_' + n).data_ptr() for n in layer.ph_prefill_adapter.names},
            'decode': {n: getattr(layer, 'ph_decode_' + n + '_gpu').data_ptr() for n in layer.ph_decode_adapter.names}}
            for layer in self.phase._LAYERS}

    def install(self):
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        runtime = self
        original_finalize = self.phase.PhaseHeteroMyTestLinearMethod.process_weights_after_loading
        original_dummy = GPUModelRunner._dummy_run
        original_execute = GPUModelRunner.execute_model
        original_capture = GPUModelRunner.capture_model

        def finalize(method, layer):
            original_finalize(method, layer)
            runtime.initialize_layer(layer)

        def apply(method, layer, x, bias=None):
            if runtime.capturing_decode or (runtime.active and runtime.phase._PHASE == 'decode'):
                return method.decode_apply(layer, x, bias)
            out = method.prefill_apply(layer, x, bias)
            if runtime.active:
                runtime.schedule(layer)
            return out

        def dummy(runner, *a, **kw):
            # Native decode capture must see decode representations in the arena.
            # This happens outside the inner torch.cuda.graph capture scope.
            if not kw.get('uniform_decode', False):
                return original_dummy(runner, *a, **kw)
            assert not runtime.active
            runtime.copy_phase('decode')
            torch.cuda.synchronize()
            runtime.capturing_decode = True
            try:
                return original_dummy(runner, *a, **kw)
            finally:
                torch.cuda.synchronize()
                runtime.capturing_decode = False
                runtime.copy_phase('prefill')
                torch.cuda.synchronize()

        def execute(runner, *a, **kw):
            runtime.wait_decode()
            return original_execute(runner, *a, **kw)

        def capture(runner, *a, **kw):
            started = time.perf_counter()
            try:
                return original_capture(runner, *a, **kw)
            finally:
                runtime.capture_model_ms += (time.perf_counter() - started) * 1000

        self.phase.PhaseHeteroMyTestLinearMethod.process_weights_after_loading = finalize
        self.phase.PhaseHeteroMyTestLinearMethod.apply = apply
        self.phase.prepare_next_prefill = self.restore
        self.phase.wait_for_prefill_ready = self.wait_prefill
        GPUModelRunner._dummy_run = dummy
        GPUModelRunner.execute_model = execute
        GPUModelRunner.capture_model = capture
