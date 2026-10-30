"""Experiment-local graph-safe FP4 bindings with unchanged dynamic scale values."""
import importlib.util
from pathlib import Path

from common import RUN


def install():
    import cutlass_wrapper.nvfp4 as dense
    import cutlass_wrapper.sparse_nvfp4 as sparse
    directory = RUN / 'extensions/graph_bindings'
    for module, stem in [(dense, 'nvfp4'), (sparse, 'sparse_nvfp4')]:
        original = module.load

        def load_graph(*a, _original=original, _stem=stem, **kw):
            name = kw['name'] + '_graph15'
            build = directory / _stem
            binary = build / (name + '.so')
            if binary.exists():
                spec = importlib.util.spec_from_file_location(name, binary)
                extension = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(extension)
                return extension
            build.mkdir(parents=True, exist_ok=True)
            kw['name'] = name
            kw['sources'] = [str(RUN / 'code/csrc' / Path(p).name) for p in kw['sources']]
            kw['build_directory'] = str(build)
            return _original(*a, **kw)

        module.load = load_graph
