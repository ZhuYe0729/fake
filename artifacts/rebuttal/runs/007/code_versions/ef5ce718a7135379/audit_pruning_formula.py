"""Independent check against the pinned official Wanda norm accumulator."""
import hashlib
import importlib.util
import json

import torch

from common import RUN,write_json


def main():
    source=RUN/'reference/wanda/lib/layerwrapper.py';spec=importlib.util.spec_from_file_location('official_wanda_wrapper',source)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    torch.manual_seed(144959);layer=torch.nn.Linear(256,32,bias=False);wrapped=official.WrappedGPT(layer)
    samples=[torch.randn(1,16,256) for _ in range(128)]
    for x in samples:wrapped.add_batch(x,None)
    local=sum(x.reshape(-1,256).square().sum(0)/128 for x in samples)
    relative=float(((wrapped.scaler_row-local).abs()/local.clamp_min(1e-12)).max())
    assert relative<5e-6,relative
    checks=[]
    for method in ['magnitude','wanda']:
        score=layer.weight.detach().abs().clone()
        if method=='wanda':score=score*wrapped.scaler_row.sqrt()[None]
        reference=torch.zeros_like(score,dtype=torch.bool)
        for i in range(0,score.shape[1],4):reference.scatter_(1,i+score[:,i:i+4].float().topk(2,dim=1,largest=False).indices,True)
        grouped=score.reshape(32,64,4);actual=torch.zeros_like(grouped,dtype=torch.bool)
        actual.scatter_(-1,grouped.topk(2,dim=-1,largest=False).indices,True)
        assert torch.equal(reference,actual.reshape_as(reference))
        checks.append({'algorithm':method,'mask_matches_official_loop':True,'zeros_per_4':2})
    metadata=[]
    for method in ['magnitude','wanda']:
        r=json.loads((RUN/f'results/canonical_{method}.json').read_text());assert len(r['modules'])==252
        assert all(m['max_nonzero_per_4']<=2 and m['mask_zero_fraction']==.5 for m in r['modules'])
        metadata.append({'algorithm':method,'modules':252,'pattern_audit':True})
    write_json(RUN/'results/pruning_formula_audit.json',{'passed':True,'official_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'max_relative_norm_accumulator_difference':relative,'mask_checks':checks,'full_model_pattern_checks':metadata,
        'scope':'Official norm accumulator and pruning-loop equivalence on fixed random inputs; full model structured-mask invariants, not an equivalence claim to official C4/Qwen-unavailable accuracy.'})
    print('pruning formula and all 504 module patterns passed')


if __name__=='__main__':main()
