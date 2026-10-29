"""Persist the exact exported tensor footprint before releasing temporary binaries."""
import json
from pathlib import Path


def inspect_checkpoint(checkpoint, destination):
    checkpoint,destination=Path(checkpoint),Path(destination)
    binary=checkpoint/'model.safetensors'
    if not binary.exists():return
    with binary.open('rb') as handle:
        header=json.loads(handle.read(int.from_bytes(handle.read(8),'little')))
    groups={key:0 for key in ['prefill','decode','shared']}
    tensors=[]
    for name,value in header.items():
        if name=='__metadata__':continue
        size=value['data_offsets'][1]-value['data_offsets'][0]
        group='prefill' if '.ph_prefill_' in name else ('decode' if '.ph_decode_' in name else 'shared')
        groups[group]+=size
        tensors.append({'name':name,'shape':value['shape'],'dtype':value['dtype'],'bytes':size,'group':group})
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'tensor_footprint.json').write_text(json.dumps({'checkpoint':str(checkpoint),
        'file_bytes':binary.stat().st_size,'tensor_bytes':groups,'tensors':tensors,
        'scope':'serialized export tensors; runtime GPU peaks separately measured, not inferred from file size'},indent=2)+'\n')
