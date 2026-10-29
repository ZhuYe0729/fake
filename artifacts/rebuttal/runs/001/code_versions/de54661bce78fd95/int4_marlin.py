"""Use installed vLLM's compiled Marlin INT4 with the repository's quantizer."""
import torch
from torch import nn

from common import WRAPPER
from standalone_marlin_int4.marlin_int4.linear import _quantize_weight_to_marlin_format, _make_workspace


@torch.no_grad()
def pack(weight):
    from vllm import _custom_ops as ops
    canonical, scale, reference = _quantize_weight_to_marlin_format(weight, 128)
    n,k=weight.shape
    packed=ops.gptq_marlin_repack(canonical.view(torch.int32).T.contiguous(),
        torch.empty(0,dtype=torch.int32,device=weight.device),k,n,4)
    return packed,scale,reference


def forward(x,weight,scale,workspace,n,k):
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    shape=x.shape[:-1]
    flat=x.to(torch.float16).reshape(-1,k).contiguous()
    result=ops.gptq_marlin_gemm(flat,None,weight,None,scale,None,None,None,None,
        workspace,scalar_types.uint4b8,flat.shape[0],n,k,True,False,True,False)
    return result.to(x.dtype).reshape(*shape,n)


class MarlinInt4(nn.Module):
    def __init__(self,weight):
        super().__init__()
        self.n,self.k=weight.shape
        packed,scale,_=pack(weight)
        self.register_buffer('weight',packed)
        self.register_buffer('scale',scale)
        self.register_buffer('workspace',_make_workspace(weight.device,self.n),persistent=False)

    def forward(self,x):
        return forward(x,self.weight,self.scale,self.workspace,self.n,self.k)
