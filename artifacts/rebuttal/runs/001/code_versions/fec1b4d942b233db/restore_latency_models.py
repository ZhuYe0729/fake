"""Rebuild missing original Roofline artifacts from preserved profiling CSVs."""
from __future__ import annotations

import hashlib
import importlib
import shutil
import time

import pandas as pd

from common import RUN, WRAPPER, write_json
from daso.modeling.latency import KernelLatencyBackend


def main():
    started=time.perf_counter()
    source=WRAPPER/'artifacts/modeling'
    output=RUN/'results/latency_models'
    provenance=[]
    for kernel in ['dense_bf16','dense_nvfp4','sparse_bf16','sparse_nvfp4','marlin_nvfp4']:
        destination=output/kernel
        destination.mkdir(parents=True,exist_ok=True)
        inputs=[]
        for suffix in ['aggregated','skip_rows']:
            path=source/kernel/f'{kernel}_{suffix}.csv'
            if path.exists():
                shutil.copy2(path,destination/path.name)
                inputs.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        data=pd.read_csv(destination/f'{kernel}_aggregated.csv')
        skip=destination/f'{kernel}_skip_rows.csv'
        skips=pd.read_csv(skip) if skip.exists() else None
        module=importlib.import_module(f'modeling.{kernel}.{kernel}_model')
        if kernel in ['dense_bf16','sparse_bf16']:
            model=module.train_model(data,**({'skip_rows':skips} if skips is not None else {}))
            module.save_model(model,destination)
        else:
            from modeling.common.kernel_latency_model import train_model,save_model
            model=train_model(module.CONFIG,data,skip_rows=skips)
            save_model(model,destination)
        provenance.append({'kernel':kernel,'rows':len(data),'inputs':inputs})
        print(kernel,len(data),'restored',flush=True)
    write_json(output/'provenance.json',{'source':'unchanged original Roofline/residual train_model, default hyperparameters',
        'reason':'default model JSON/NPZ absent; retained original aggregate and skip CSVs',
        'elapsed_seconds':time.perf_counter()-started,'models':provenance})
    backend=KernelLatencyBackend(WRAPPER,output)
    rows=[]
    for kind,(n,k) in {'qkv_proj':(6144,4096),'o_proj':(4096,4096),
                       'gate_up_proj':(24576,4096),'down_proj':(4096,12288)}.items():
        for phase,m in [('prefill',16384),('decode',8)]:
            actions=backend.predict(m,n,k)
            # Runtime pads sparse FP4 activations to M%32==0, including decode.
            if m%32:actions['sparse_nvfp4']=backend.predict(32,n,k)['sparse_nvfp4']
            row={'kind':kind,'phase':phase,'m':m,'actions':{key:vars(value) for key,value in actions.items()}}
            rows.append(row)
            print(row,flush=True)
    write_json(RUN/'results/daso_predictor_restored_audit.json',rows)
    assert all(a['supported'] and a['latency_ms'] is not None for row in rows for a in row['actions'].values())


if __name__=='__main__':main()
