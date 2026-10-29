"""Freeze simple structured-pruning controls and archive official references."""
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def main():
    for d in ['configs/policies','data','reference/wanda','results','logs','code/compat']:(RUN/d).mkdir(parents=True,exist_ok=True)
    source=RUN.parent/'003'
    for name in ['common.py','eightbit.py','runtime_extensions.py','export_model.py','int4_marlin.py']:shutil.copy2(source/'code'/name,RUN/'code'/name)
    shutil.copytree(source/'code/compat',RUN/'code/compat',dirs_exist_ok=True)
    for name in ['calibration_128_2112.pt','wikitext_test_tokens.pt']:
        if not (RUN/'data'/name).exists():(RUN/'data'/name).symlink_to(RUN.parent/'001/data'/name)
    if not (RUN/'data/activation_scales.pt').exists():(RUN/'data/activation_scales.pt').symlink_to(source/'data/activation_scales.pt')
    policy=json.loads((source/'configs/policies/uniform_sparse_bf16.json').read_text())
    for method in ['magnitude','wanda','sparsegpt']:
        policy['policy_id']=method+'_2of4';(RUN/f'configs/policies/{method}.json').write_text(json.dumps(policy,indent=2)+'\n')
    commit=subprocess.check_output(['git','ls-remote','https://github.com/locuslab/wanda.git','refs/heads/main'],text=True).split()[0];records={}
    for name in ['lib/prune.py','lib/layerwrapper.py','LICENSE']:
        data=urllib.request.urlopen(f'https://raw.githubusercontent.com/locuslab/wanda/{commit}/{name}').read()
        path=RUN/'reference/wanda'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        records[name]={'sha256':hashlib.sha256(data).hexdigest(),'url':f'https://github.com/locuslab/wanda/blob/{commit}/{name}'}
    (RUN/'reference/wanda/provenance.json').write_text(json.dumps({'commit':commit,'files':records},indent=2)+'\n')
    (RUN/'configs/design.json').write_text(json.dumps({'methods':['magnitude','wanda','sparsegpt'],
        'pattern':'2 zeros per consecutive 4 input weights, each output row','kernel':'same real sparse_bf16 cuSPARSELt, prune=False at packing',
        'calibration':'same original 128 WikiText-train blocks, first 2048 tokens; magnitude uses no calibration',
        'wanda':'official abs(weight)*sqrt(input sum squares per sample), sequential propagation between pruned decoder layers',
        'sparsegpt':'reuse immutable run001 canonical SparseGPT BF16 weights; report inherited preparation cost',
        'quality':'64 fixed nonoverlapping WikiText test windows, prefill and actual B8/O64 teacher forcing',
        'speed':'same GPU5, B8/S2048/O64, 2 warmups+5 iterations, 6GiB KV; homogeneous batches',
        'scope':'simple pruning controls within the same structured execution format; no uniform-pruning accuracy claim for heterogeneous DASO'},indent=2)+'\n')


if __name__=='__main__':main()
