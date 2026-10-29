"""Stage a quality-only parallel follow-up without creating CUDA contexts."""
from pathlib import Path
import hashlib
import json
import os
import shutil

BASE=Path(__file__).resolve().parents[1];SOURCE=BASE/'runs/053';RUN=BASE/'runs/054'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
assert not RUN.exists(),'Preserve an existing staged run'
for name in ['code','configs','reference','logs','results/measurements','results/completed','results/workers','code_versions','tmp','data/work/checkpoints']:(RUN/name).mkdir(parents=True,exist_ok=True)
manifest=json.loads((SOURCE/'configs/protocol_sha256.json').read_text())
for name,digest in manifest.items():
 assert sha(SOURCE/'code'/name)==digest
 target=RUN/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/'code'/name,target)
write(RUN/'configs/protocol_sha256.json',manifest);version=sha(RUN/'configs/protocol_sha256.json')[:16]
for name in manifest:
 target=RUN/'code_versions'/version/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(RUN/'code'/name,target)
design=json.loads((SOURCE/'configs/design.json').read_text());policies=['bf16','pd_hawq_bal','pd_hawq_fast','pd_daso_bal','pd_daso_fast']
design['policies']={p:design['policies'][p] for p in policies};design['rows']={'prefill_decode':design['rows']['prefill_decode']}
design['cohort_groups']={k:v for k,v in design['cohort_groups'].items() if v['phase']=='long_speed'}
design['workloads']=[v for v in design['workloads'] if v['tier']==2]
inputs={}
for case in design['workloads']:
 for field,digest in [('input',case['input_sha256']),('quality_file',case['quality_sha256'])]:
  source=Path(case[field]);assert sha(source)==digest
  target=RUN/'reference'/(digest[:16]+source.suffix)
  if not target.exists():shutil.copy2(source,target)
  case[field]=str(target.resolve());inputs[str(target.resolve())]=digest
design['batch_policies']=policies;design['minimum_remaining_seconds']=180;design['baseline_replication_tolerance']=1e-4
design['compiler_cache']={'triton':str((SOURCE/'extensions/triton').resolve()),'cuda':str((SOURCE/'extensions/cuda').resolve())}
design['source_sha256']={str((SOURCE/'code'/name).resolve()):digest for name,digest in manifest.items()}
design['source_sha256'][str((SOURCE/'configs/design.json').resolve())]=sha(SOURCE/'configs/design.json')
write(RUN/'configs/design.json',design);write(RUN/'reference/input_sha256.json',inputs)
reuse={}
for policy in policies:
 source=SOURCE/'data/work/checkpoints'/policy;target=RUN/'data/work/checkpoints'/policy;target.mkdir()
 for file in source.iterdir():
  if file.is_file():
   if file.name=='model.safetensors':os.link(file,target/file.name)
   else:shutil.copy2(file,target/file.name)
 stat=(target/'model.safetensors').stat();reuse[policy]={'source':str((source/'model.safetensors').resolve()),'target':str((target/'model.safetensors').resolve()),'device':stat.st_dev,'inode':stat.st_ino,'bytes':stat.st_size,'config_sha256':sha(target/'config.json')}
write(RUN/'reference/checkpoint_reuse.json',reuse)
(RUN/'README.md').write_text('# 054 — 长Decode精度并行补测\n\n仅准备，尚无GPU结果。五个冻结点、两种长度、64窗，与053数值代码完全一致。须先完成并核验053全部速度，至少剩180秒且有5张持续空闲卡才启动；原01:12:53 UTC截止。每点GPU及共同BF参考明示，可额外重复BF作跨卡检查。此处不产生性能数据。\n')
write(RUN/'results/worker.json',{'status':'prepared','new_gpu_measurements':False,'deadline_unix':design['watch']['deadline_unix']})
print(json.dumps({'run':str(RUN),'code_files':len(manifest),'policies':policies,'physical_weight_copies':0,'status':'prepared'}))
