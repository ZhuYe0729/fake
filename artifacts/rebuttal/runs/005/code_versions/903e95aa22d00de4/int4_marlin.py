"""Use installed vLLM's compiled Marlin INT4 with the repository's quantizer."""
import torch
from torch import nn

from standalone_marlin_int4.marlin_int4.linear import _quantize_weight_to_marlin_format, _make_workspace, _scale_perms


@torch.no_grad()
def pack(weight):
    from vllm import _custom_ops as ops
    canonical, scale, _ = _quantize_weight_to_marlin_format(weight, 128)
    n,k=weight.shape
    scale=scale.to(torch.bfloat16)
    permutation,_=_scale_perms()
    inverse=torch.argsort(torch.tensor(permutation,device=weight.device))
    logical=scale.reshape(-1,len(permutation))[:,inverse].reshape(k//128,n)
    codes=torch.empty((n,k),device=weight.device,dtype=torch.int8)
    codes[:,0::2]=(canonical&15).to(torch.int8)-8
    codes[:,1::2]=(canonical>>4).to(torch.int8)-8
    reference=codes.to(torch.bfloat16)*logical.repeat_interleave(128,dim=0).T
    packed=ops.gptq_marlin_repack(canonical.view(torch.int32).T.contiguous(),
        torch.empty(0,dtype=torch.int32,device=weight.device),k,n,4)
    return packed,scale,reference


def forward(x,weight,scale,workspace,n,k):
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    shape=x.shape[:-1]
    flat=x.reshape(-1,k).contiguous()
    result=ops.gptq_marlin_gemm(flat,None,weight,None,scale,None,None,None,None,
        workspace,scalar_types.uint4b8,flat.shape[0],n,k,True,False,True,False)
    return result.reshape(*shape,n)


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
