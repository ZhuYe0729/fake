"""Preserve numerical work; shorten resource admission cohorts for the same cases."""
from pathlib import Path
import hashlib
import json
import os
import shutil

BASE=Path(__file__).resolve().parents[1];OLD=BASE/'runs/052';RUN=BASE/'runs/053'
assert not RUN.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def copy(a,b):
 if Path(a).suffix in ['.safetensors','.pt','.so','.cubin','.fatbin']:os.link(a,b)
 else:shutil.copy2(a,b)
 return b
for name in ['code','configs','data','extensions']:
 shutil.copytree(OLD/name,RUN/name,copy_function=copy,ignore=shutil.ignore_patterns('__pycache__','*.lock','*.tmp','tmp*'))
for name in ['logs','results/accepted','results/round_attempts','results/leases','tmp','reference']:(RUN/name).mkdir(parents=True,exist_ok=True)
old=str(OLD.resolve());new=str(RUN.resolve())
for p in (RUN/'configs').rglob('*.json'):
 if p.name not in ['design.json','protocol_sha256.json']:p.write_text(p.read_text().replace(old,new))
d=read(OLD/'configs/design.json')
for c in d['workloads']:c['input']=c['input'].replace(old,new)
d['cohort_groups']={
 'core_pp':{'phase':'core_speed','scenario':'prefill_only','workloads':['center_pp','batch4_pp','prompt4096_pp']},
 **{f'core_pd_{name}':{'phase':'core_speed','scenario':'prefill_decode','workloads':[name]} for name in ['center_pd','batch4_pd','prompt4096_pd','output128_pd']},
 'decode256':{'phase':'long_speed','scenario':'prefill_decode','workloads':['decode256_pd']},
 'decode512':{'phase':'long_speed','scenario':'prefill_decode','workloads':['decode512_pd']}}
d['scope']='Same fixed engine/cases/maps; complete scenario/output cohorts with BF16 brackets, sustained idle admission and same-device three repeats.'
d['primary_selection']='First device with three accepted cohorts of every group of the phase; never performance-based.'
d.pop('primary_gpu',None);d.pop('validation_gpu',None)
d['watch']['required_idle_seconds']=90
d['watch']['idle_window_revision']='Before accepted 053 data: 300 -> 180 -> 90 seconds after measured idle-window availability; four samples at 30 seconds, unchanged memory/utilization and in-run resource criteria.'
write(RUN/'configs/watch.json',d['watch'])
for name in read(OLD/'configs/protocol_sha256.json'):d['source_sha256'][str((OLD/'code'/name).resolve())]=sha(OLD/'code'/name)
write(RUN/'configs/design.json',d)
weights={}
for pid,info in read(OLD/'reference/checkpoint_reuse.json').items():
 source=Path(info['target']);target=RUN/'data/work/checkpoints'/pid/'model.safetensors';s=target.stat();assert (s.st_dev,s.st_ino,s.st_size)==(source.stat().st_dev,source.stat().st_ino,source.stat().st_size)
 weights[pid]={'source':str(source),'target':str(target.resolve()),'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'config_sha256':sha(target.parent/'config.json')}
write(RUN/'reference/checkpoint_reuse.json',weights)
experiment=RUN/'code/experiment.py';s=experiment.read_text();s=s.replace(" parser.add_argument('--label',required=True)"," parser.add_argument('--label',required=True)\n parser.add_argument('--group')")
s=s.replace(' if not quality:\n  shift='," if args.group:\n  selected=design['cohort_groups'][args.group]['workloads'];cases=[c for c in cases if c['label'] in selected]\n  assert cases\n if not quality:\n  shift=")
experiment.write_text(s)
write(RUN/'reference/setup.json',{'source_run':'052','weights_hardlinked':True,'weight_copies':0,'deadline_unix':d['watch']['deadline_unix'],'core_case_rounds':117,'long_case_rounds':36})
print('053 prepared with unchanged kernels/weights and group selector only')
