"""Keep the assigned card occupied between runs; release before inference."""
import json
import os
import time

import torch

from common import RUN, write_json


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    root=RUN/'results/gpu0_reservation';root.mkdir(exist_ok=True)
    size=12*1024**3
    buffer=torch.empty(size,device='cuda',dtype=torch.uint8)
    write_json(root/'state.json',{'status':'holding','pid':os.getpid(),'gpu':0,'reserved_bytes':size})
    last=None
    while True:
        request=root/'request.json'
        if not request.exists():time.sleep(.05);continue
        value=json.loads(request.read_text())
        if value['nonce']==last:time.sleep(.05);continue
        operation=value['operation']
        if operation in ['release','stop']:
            buffer=None;torch.cuda.empty_cache()
        elif operation=='hold' and buffer is None:
            if torch.cuda.mem_get_info()[0]<size+512*1024**2:time.sleep(.05);continue
            buffer=torch.empty(size,device='cuda',dtype=torch.uint8)
        ack={**value,'reserved_bytes':0 if buffer is None else size,'pid':os.getpid(),'time_ns':time.time_ns()}
        write_json(root/'ack.json',ack);write_json(root/'state.json',{'status':operation,**ack})
        with (root/'events.jsonl').open('a') as f:f.write(json.dumps(ack)+'\n')
        last=value['nonce']
        if operation=='stop':break


if __name__=='__main__':main()
