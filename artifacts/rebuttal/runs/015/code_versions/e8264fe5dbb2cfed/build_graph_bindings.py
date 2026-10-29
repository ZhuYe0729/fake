"""Compile isolated binding changes once, outside all measurements."""
from graph_bindings import install

install()
import cutlass_wrapper.nvfp4 as dense
import cutlass_wrapper.sparse_nvfp4 as sparse

dense._load_ext()
print('dense graph binding ready', flush=True)
sparse._load_ext()
print('sparse graph binding ready', flush=True)
