"""Teacher forcing that ignores vLLM's discarded partial-prefill samples."""
import json
from pathlib import Path

import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from mixed_runtime import STATE


class Request:
    def __init__(self,extra):
        self.targets=extra['teacher_force_target_ids'];self.path=Path(extra['teacher_force_capture_path'])
        self.key=extra['teacher_force_request_key']

    def __call__(self,output_ids,logits):
        if self.key not in STATE.get('ready_requests',set()):
            STATE['teacher_skipped_partial_samples']+=1
            return logits
        step=len(output_ids)
        if step>=len(self.targets):return logits
        target=self.targets[step];logprob=float(torch.log_softmax(logits.float(),dim=-1)[target])
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a') as handle:handle.write(json.dumps({'step':step,'target_id':target,'logprob':logprob,'output_length':len(output_ids)})+'\n')
        value=logits[target].clone();logits.fill_(float('-inf'));logits[target]=value
        return logits


class TeacherForceBatchProcessor(AdapterLogitsProcessor):
    def is_argmax_invariant(self):return False

    def new_req_logits_processor(self,params):
        extra=params.extra_args
        return Request(extra) if extra and 'teacher_force_target_ids' in extra else None
