"""Freeze original-protocol downstream jobs and reuse immutable run049 weights."""
from pathlib import Path
import hashlib
import json
import os
import shutil

BASE=Path(__file__).resolve().parents[1]
RUN=BASE/'runs/050'
OLD=BASE/'runs/049'
assert not (RUN/'configs/design.json').exists()
for part in ['code','configs/policies','data/work/checkpoints','results','logs','reference','tmp','extensions']:
 (RUN/part).mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
sources={}
for name in ['common.py','eightbit.py','runtime_extensions.py','arena_runtime.py','graph_bindings.py','int4_marlin.py','telemetry.py']:
 p=OLD/'code'/name;sources[str(p)]=sha(p);shutil.copy2(p,RUN/'code'/name)
for folder in ['compat','csrc']:shutil.copytree(OLD/'code'/folder,RUN/'code'/folder,dirs_exist_ok=True)
for name in ['likelihood_datasets','likelihood_auxiliary_datasets']:
 source=(BASE/f'runs/015/data/{name}').resolve();(RUN/'data'/name).symlink_to(source,target_is_directory=True)
for name in ['activation_scales.pt',*[f'generation_{t}.jsonl' for t in ['cnn_dm_1000','dsum','IWSLT']]]:
 source=OLD/'data'/name if name=='activation_scales.pt' else BASE/'runs/048/data'/name
 sources[str(source)]=sha(source);shutil.copy2(source,RUN/'data'/name)
for name in ['generation_protocol.json','likelihood_task_configs.json']:
 p=BASE/'runs/001/data'/name;sources[str(p)]=sha(p);shutil.copy2(p,RUN/'reference'/name)
original=json.loads((OLD/'configs/design.json').read_text());sources[str(OLD/'configs/design.json')]=sha(OLD/'configs/design.json')
weights={}
for pid,spec in original['policies'].items():
 p=OLD/f'configs/policies/{pid}.json';shutil.copy2(p,RUN/f'configs/policies/{pid}.json');sources[str(p)]=sha(p)
 old=OLD/f'data/work/checkpoints/{pid}';new=RUN/f'data/work/checkpoints/{pid}';new.mkdir()
 for p in old.iterdir():
  if not p.is_file():continue
  if p.name=='model.safetensors':os.link(p,new/p.name)
  else:shutil.copy2(p,new/p.name)
 a=(old/'model.safetensors').stat();b=(new/'model.safetensors').stat()
 assert (a.st_dev,a.st_ino,a.st_size)==(b.st_dev,b.st_ino,b.st_size)
 config=json.loads((new/'config.json').read_text())
 assert config['quantization_config']['method_map']==json.loads((RUN/f'configs/policies/{pid}.json').read_text())['method_map']
 weights[pid]={'source':str(old/'model.safetensors'),'target':str(new/'model.safetensors'),
  'inode':a.st_ino,'device':a.st_dev,'bytes':a.st_size,'config_sha256':sha(new/'config.json'),'exact_hardlink':True}
tasks=[('winogrande',1267,'acc,none'),('arc_easy',2376,'acc,none'),('arc_challenge',1172,'acc_norm,none'),('mmlu',14042,'acc,none')]
counts={'cnn_dm_1000':1000,'dsum':1500,'IWSLT':333}
jobs=[]
pp=['bf16','pp_hawq_bal','pp_shared_fast','pp_daso_bal']
pd=['bf16','pd_hawq_fast','pd_daso_fast','pd_hawq_bal','pd_daso_bal']
for batch in [8,4]:
 for pid in pd:
  for task in ['IWSLT','cnn_dm_1000','dsum']:
   for begin in range(0,counts[task],512):
    end=min(begin+512,counts[task]);jobs.append({'id':f'gen_b{batch}_{pid}_{task}_{begin:04d}_{end:04d}',
     'kind':'generation','policy':pid,'batch':batch,'task':task,'begin':begin,'end':end})
 for pid in pp:
  for task,count,metric in tasks:
   jobs.append({'id':f'mc_b{batch}_{pid}_{task}','kind':'multiple_choice','policy':pid,'batch':batch,'task':task,'samples':count,'metric':metric})
# Establish the original-protocol full IWSLT anchor and a complete MC task first.
anchors=['gen_b8_bf16_IWSLT_0000_0333','mc_b4_bf16_arc_easy']
jobs=sorted(jobs,key=lambda j:anchors.index(j['id']) if j['id'] in anchors else 2)
protected={str(BASE/p):sha(BASE/p) for p in ['runs/049/configs/design.json','runs/049/results/tier1_complete.json',
 'responses/reviewer2/qwen3_main_table/main_table.csv','runs/048/results/final_audit.json']}
write(RUN/'configs/design.json',{'model':'Qwen3-8B','policies':original['policies'],'rows':original['rows'],
 'batches':[8,4],'mc_tasks':[{'task':t,'samples':n,'metric':m} for t,n,m in tasks],
 'generation_tasks':counts,'generation_max_new_tokens':256,'generation_input_protocol_max':3840,
 'mc_max_model_len':2048,'generation_max_model_len':4096,'jobs':jobs,'anchor_jobs':anchors,
 'preferred_cards':[3,4,6,0,1,2,5,7],'workers':4,'source_sha256':sources,'protected_sha256':protected,
 'quality_scope':'Original paper tasks/input files and generation limit256, measured separately at batches8/4. S/O speed labels do not define new task datasets; shared scores are explicitly identified.',
 'speed_stage':'deferred_until_quiet_evening_window','performance_measurements_enabled':False})
write(RUN/'reference/checkpoint_reuse.json',weights)
inputs={}
for folder in ['likelihood_datasets','likelihood_auxiliary_datasets']:
 for p in (RUN/'data'/folder).rglob('*.arrow'):
  actual=p.resolve();inputs[str(actual)]=sha(actual)
write(RUN/'reference/likelihood_inputs.json',inputs)
path=RUN/'configs/design.json';design=json.loads(path.read_text())
design['input_sha256']={p.name:sha(p) for p in (RUN/'data').glob('generation_*.jsonl')}
write(path,design)
write(RUN/'results/worker.json',{'status':'prepared','stage':'quality_only','jobs':len(jobs),'speed_enabled':False})
print(json.dumps({'jobs':len(jobs),'generation_chunks':sum(j['kind']=='generation' for j in jobs),
 'full_mc_tasks':sum(j['kind']=='multiple_choice' for j in jobs),'immutable_checkpoints':len(weights),'physical_weight_copy_bytes':0},indent=2))
