"""Freeze the nearby workloads and existing policies before any new GPU output."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil

import torch

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/049'
SOURCE=BASE/'runs/048'
assert not (RUN/'configs/design.json').exists()
for part in ['code','configs/policies','data','logs','results','reference','tmp','extensions']:
 (RUN/part).mkdir(parents=True,exist_ok=True)
sources={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(src,dst):
 sources[str(src)]=sha(src);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
for name in ['common.py','eightbit.py','runtime_extensions.py','arena_runtime.py','graph_bindings.py',
 'export_model.py','int4_marlin.py','telemetry.py','launch.py']:
 copy(SOURCE/'code'/name,RUN/'code'/name)
for folder in ['compat','csrc']:shutil.copytree(SOURCE/'code'/folder,RUN/'code'/folder,dirs_exist_ok=True)
copy(SOURCE/'data/activation_scales.pt',RUN/'data/activation_scales.pt')
copy(SOURCE/'reference/source_model_fingerprint.json',RUN/'reference/source_model_fingerprint.json')
copy(BASE/'runs/019/data/long_quality.pt',RUN/'data/quality_windows.pt')
data=torch.load(RUN/'data/quality_windows.pt',weights_only=True)
assert tuple(data.shape)==(64,4160)
copy(SOURCE/'configs/speed_inputs.json',RUN/'reference/center_speed_inputs.json')
center=json.loads((RUN/'reference/center_speed_inputs.json').read_text())
assert len(center)==8 and all(len(x)==2048 for x in center)

primary=json.loads((BASE/'responses/reviewer2/qwen3_main_table/provenance.json').read_text())
primary={r['method']:r for r in primary['rows']}
definitions=[('bf16','BF16',None,'048','uniform_dense_bf16'),
 ('pp_daso_bal','DASO（均衡）','prefill_only','015','uniform_sq_int8'),
 ('pp_hawq_bal','HAWQ-V3（均衡）','prefill_only','015','graph_hawq_fp4_prefill_only_r0.8'),
 ('pp_shared_fast','共同速度端点','prefill_only','016','hardware_minimum_prefill_only_same_operator'),
 ('pd_daso_bal','DASO（均衡）','prefill_decode','048','w1_e0.01'),
 ('pd_daso_fast','DASO（速度优先）','prefill_decode','048','w32_e0.03'),
 ('pd_hawq_bal','HAWQ-V3（均衡）','prefill_decode','048','hawq_signed_prefill_decode_phase_pair_r0.8'),
 ('pd_hawq_fast','HAWQ-V3（速度优先）','prefill_decode','048','hardware_minimum_prefill_decode_phase_pair')]
policies={}
for pid,label,scene,rid,source_pid in definitions:
 path=BASE/f'runs/{rid}/configs/policies/{source_pid}.json';d=json.loads(path.read_text())
 sources[str(path)]=sha(path)
 signature=hashlib.sha256(json.dumps(d['method_map'],sort_keys=True).encode()).hexdigest()
 if scene:
  oldlabel=label.replace('（均衡）',' (Balanced)').replace('（速度优先）',' (Max-speed)')
  refs=['DASO (Max-speed)','HAWQ-V3 (Max-speed)'] if pid=='pp_shared_fast' else [oldlabel]
  assert all(primary[name][scene]['map_sha256']==signature for name in refs)
 d['policy_id']=pid;write(RUN/f'configs/policies/{pid}.json',d)
 policies[pid]={'display':label,'scenario':scene,'source_run':rid,'source_policy':source_pid,
  'map_sha256':signature,'composition':dict(Counter(v['prefill_method'] for v in d['method_map'].values()))}

shapes=[('center',8,2048,64,1),('batch4',4,2048,64,1),('prompt4096',8,4096,64,1),
 ('output128',8,2048,128,1),('batch16',16,2048,64,2),('prompt1024',8,1024,64,2)]
cases=[];seen=set()
for label,batch,prompt,output,tier in shapes:
 tokens=[(center[i%8]*2)[:prompt] for i in range(batch)]
 path=RUN/f'configs/{label}_speed_inputs.json';write(path,tokens)
 if label=='center':assert tokens==center
 for scene in ['prefill_only','prefill_decode']:
  if scene=='prefill_only' and (batch,prompt) in seen:continue
  if scene=='prefill_only':seen.add((batch,prompt))
  cases.append({'label':label+('_pp' if scene=='prefill_only' else '_pd'),
   'scenario':scene,'batch':batch,'prompt':prompt,'output':1 if scene=='prefill_only' else output,
   'tier':tier,'input':str(path),'input_sha256':sha(path),
   'quality_file':str(RUN/'data/quality_windows.pt'),'quality_sha256':sha(RUN/'data/quality_windows.pt')})
rows={
 'prefill_only':[{'method':'BF16','policy':'bf16'},{'method':'HAWQ-V3（均衡）','policy':'pp_hawq_bal'},
  {'method':'HAWQ-V3（速度优先）','policy':'pp_shared_fast'},{'method':'DASO（均衡）','policy':'pp_daso_bal'},
  {'method':'DASO（速度优先）','policy':'pp_shared_fast'}],
 'prefill_decode':[{'method':'BF16','policy':'bf16'},{'method':'HAWQ-V3（均衡）','policy':'pd_hawq_bal'},
  {'method':'HAWQ-V3（速度优先）','policy':'pd_hawq_fast'},{'method':'DASO（均衡）','policy':'pd_daso_bal'},
  {'method':'DASO（速度优先）','policy':'pd_daso_fast'}]}
order=list(policies)
rounds=[order, list(reversed(order)), order[3:]+order[:3]]
write(RUN/'configs/design.json',{'policies':policies,'rows':rows,'workloads':cases,'rounds':rounds,
 'warmups':5,'iterations':10,'quality_windows':64,'cards':[0,3],
 'engine':{'capture_sizes':[4,8,16],'max_model_len':4232,'max_num_seqs':16,'max_num_batched_tokens':32768,'kv_gib':6},
 'source_sha256':sources,'scope':'Frozen primary policies; tier1 seven scenarios, tier2 four scenarios; no new downstream tasks or per-load tuning.',
 'quality_scope':'64 previously studied public WikiText windows; shared prefixes across loads; no blind or new-data claim.'})
write(RUN/'configs/model.json',{'run':'049','model_id':'qwen3_8b','model_path':'/root/wja/data/models/Qwen/Qwen3-8B',
 'temp':str(RUN/'data/work'),'canonical_root':str(BASE/'night151/cache/daso_rebuttal_001_20261025/canonical')})
write(RUN/'results/preparation.json',{'prepared':True,'policies':len(policies),'tier1_scenarios':sum(c['tier']==1 for c in cases),
 'tier2_scenarios':sum(c['tier']==2 for c in cases),'center_input_equal_to_048':True,'quality_shape':list(data.shape)})
print(json.dumps(json.loads((RUN/'results/preparation.json').read_text()),indent=2))
