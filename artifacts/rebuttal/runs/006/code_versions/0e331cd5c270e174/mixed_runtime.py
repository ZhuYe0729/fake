"""Route real scheduler token rows to independent resident phase kernels."""
from collections import Counter

import numpy as np
import torch

STATE={'active':False,'prefill':None,'decode':None,'steps':[],'linear_counts':Counter()}


def indices(positions,prompt_lengths,scheduled):
    limits=np.repeat(np.asarray(prompt_lengths),np.asarray(scheduled))
    positions=np.asarray(positions)
    assert len(positions)==len(limits)
    mask=positions<limits
    return np.flatnonzero(mask),np.flatnonzero(~mask)


def hook_inputs():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    original=GPUModelRunner._prepare_inputs

    def prepare(self,scheduler_output):
        result=original(self,scheduler_output)
        if not STATE['active']:return result
        total=scheduler_output.total_num_scheduled_tokens;n=self.input_batch.num_reqs
        scheduled=[scheduler_output.num_scheduled_tokens[i] for i in self.input_batch.req_ids]
        pre,dec=indices(self.positions.np[:total],self.input_batch.num_prompt_tokens[:n],scheduled)
        STATE.update(prefill=torch.from_numpy(pre).to(self.device),decode=torch.from_numpy(dec).to(self.device),
            pre_count=len(pre),dec_count=len(dec),total=total)
        STATE['steps'].append({'scheduled_tokens':total,'prefill_tokens':len(pre),'decode_tokens':len(dec),'requests':n,'mixed':bool(len(pre) and len(dec))})
        if STATE.get('track_teacher'):
            # vLLM also samples (and discards) logits for incomplete prompt
            # chunks. Such rows must not consume a teacher-forcing target.
            STATE['ready_requests']={rid for i,rid in enumerate(self.input_batch.req_ids)
                if self.input_batch.num_computed_tokens_cpu[i]+scheduled[i]>=self.input_batch.num_prompt_tokens[i]}
        return result

    GPUModelRunner._prepare_inputs=prepare


def install(phase):
    from resident import install as resident
    resident(phase)
    original=phase.PhaseHeteroMyTestLinearMethod.apply

    def apply(self,layer,x,bias=None):
        if not STATE['active']:return original(self,layer,x,bias)
        if layer.ph_prefill_method==layer.ph_decode_method:
            STATE['linear_counts']['same_method']+=1
            return self.prefill_apply(layer,x,bias)
        flat=x.reshape(-1,x.shape[-1]);assert len(flat)==STATE['total'],(len(flat),STATE['total'])
        if not STATE['dec_count']:
            STATE['linear_counts']['homogeneous_prefill']+=1
            return self.prefill_apply(layer,x,bias)
        if not STATE['pre_count']:
            STATE['linear_counts']['homogeneous_decode']+=1
            return self.decode_apply(layer,x,bias)
        STATE['linear_counts']['mixed']+=1
        pref=STATE['prefill'];dec=STATE['decode']
        pre_out=self.prefill_apply(layer,flat.index_select(0,pref),bias)
        dec_out=self.decode_apply(layer,flat.index_select(0,dec),bias)
        out=torch.empty((len(flat),layer.output_size_per_partition),dtype=x.dtype,device=x.device)
        out.index_copy_(0,pref,pre_out);out.index_copy_(0,dec,dec_out)
        return out.reshape(*x.shape[:-1],layer.output_size_per_partition)

    phase.PhaseHeteroMyTestLinearMethod.apply=apply
    STATE['active']=True
    return apply


def reset():
    STATE['steps'].clear();STATE['linear_counts'].clear()
    STATE['teacher_skipped_partial_samples']=0


def summary():
    return {'steps':list(STATE['steps']),'linear_counts':dict(STATE['linear_counts']),
        'teacher_skipped_partial_samples':STATE.get('teacher_skipped_partial_samples',0),
        'mixed_forwards':sum(x['mixed'] for x in STATE['steps']),
        'total_prefill_rows':sum(x['prefill_tokens'] for x in STATE['steps']),
        'total_decode_rows':sum(x['decode_tokens'] for x in STATE['steps'])}


def audit(phase):
    # Includes a one-token prompt, partial chunks, a request spanning its prompt
    # boundary, and interleaved request row groups. Expected rows are explicit.
    pre,dec=indices([0,1,2,3,4,5,0,1,7],[4,1,7],[6,2,1])
    assert pre.tolist()==[0,1,2,3,6] and dec.tolist()==[4,5,7,8]
    checks=[];saved=dict(STATE);torch.manual_seed(144911)
    for layer in phase._LAYERS:
        if layer.ph_prefill_method==layer.ph_decode_method:continue
        method=layer.quant_method
        x=torch.randn(17,layer.input_size_per_partition,device='cuda',dtype=torch.bfloat16)
        p=torch.tensor([0,2,3,6,7,8,11,13,16],device='cuda');d=torch.tensor([1,4,5,9,10,12,14,15],device='cuda')
        STATE.update(prefill=p,decode=d,pre_count=len(p),dec_count=len(d),total=17)
        expected=torch.empty((17,layer.output_size_per_partition),device='cuda',dtype=x.dtype)
        expected.index_copy_(0,p,method.prefill_apply(layer,x.index_select(0,p),None))
        expected.index_copy_(0,d,method.decode_apply(layer,x.index_select(0,d),None))
        actual=method.apply(layer,x,None)
        assert torch.equal(actual,expected),layer.prefix
        checks.append({'module':layer.prefix,'prefill_rows':len(p),'decode_rows':len(d),'exact_equal':True})
    STATE.clear();STATE.update(saved);reset()
    return {'passed':True,'mask_cases':1,'mixed_layer_cases':checks,
        'scope':'Exact gather/phase-adapter/scatter equality on identical row subsets, not a claim that chunking leaves global FP4 activation scale unchanged.'}
