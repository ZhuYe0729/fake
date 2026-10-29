"""Check unpruned canonical tensors against the restored/raw target checkpoint."""
import json

import torch
from safetensors import safe_open

from common import MODEL, RUN, SPEC, write_json


def main():
    mapping = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    config = json.loads((MODEL / 'config.json').read_text())
    checks = []
    names = [f'model.layers.{i}.{norm}.weight' for i in range(config['num_hidden_layers'])
             for norm in ['input_layernorm','post_attention_layernorm']]
    names += ['model.norm.weight','model.embed_tokens.weight','lm_head.weight']
    for method in ['sparse_bf16','sparse_nvfp4']:
        path = __import__('pathlib').Path(SPEC['canonical_root']) / method / 'model.pt'
        state = torch.load(path,map_location='cpu',mmap=True,weights_only=True)['state_dict']
        for name in names:
            with safe_open(MODEL / mapping[name],framework='pt',device='cpu') as source:
                raw = source.get_tensor(name)
                retained = state[name]
                assert raw.shape == retained.shape,(method,name)
                if raw.ndim == 2:
                    indices = torch.linspace(0,len(raw)-1,64).long()
                    raw,retained = raw[indices],retained[indices]
                equal = torch.equal(raw.to(torch.bfloat16),retained.to(torch.bfloat16))
                assert equal,(method,name,'canonical non-pruned tensor differs from target model')
                checks.append({'method':method,'tensor':name,'equal_effective_bf16':equal,
                               'scope':'all norm values,64 fixed embedding/head rows'})
    write_json(RUN / 'results/source_identity_gate.json',{'passed':True,'checks':checks,
        'scope':'Model identity guard for old canonical states; functional BF16/NVFP4/mixed old-label gates additionally required before reuse.'})
    print('Canonical source identity passed:',SPEC['model_id'],len(checks),'tensor checks',flush=True)


if __name__ == '__main__':
    main()
