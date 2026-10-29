"""Measure pinned H2D on fixed CPU/memory nodes and the same GPU0."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True);args=parser.parse_args()
    run=Path(__file__).resolve().parents[1];design=json.loads((run/'configs/design.json').read_text());spec=design['copy_probe']
    import torch
    from telemetry import gpu,placement
    rows=[]
    for mib in spec['mib']:
        source=torch.empty(mib*1024**2,dtype=torch.uint8,pin_memory=True);source.fill_(37)
        target=torch.empty_like(source,device='cuda');assert source.is_pinned()
        for i in range(spec['warmups']+spec['iterations']):
            before=gpu();torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            clock=time.perf_counter();start.record();target.copy_(source,non_blocking=True);end.record();end.synchronize()
            wall=(time.perf_counter()-clock)*1000;device=start.elapsed_time(end)
            rows.append({'mib':mib,'iteration':i,'warmup':i<spec['warmups'],'gpu_ms':device,'wall_ms':wall,
                'gib_per_second':mib/1024/(device/1000),'before':before,'after':gpu()})
        assert torch.equal(target.cpu(),source)
        del source,target
    output={'index':args.index,'node':spec['order'][args.index],'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
        'placement':placement(),'samples':rows,'data_equal':True}
    path=run/f'results/copy/{args.index}.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'index':args.index,'node':output['node'],'gib_per_second':{m:statistics.median(r['gib_per_second'] for r in rows if r['mib']==m and not r['warmup']) for m in spec['mib']}}))


if __name__=='__main__':main()
