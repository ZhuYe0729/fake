import json,hashlib,random,shutil,time
from pathlib import Path
import torch
base=Path('artifacts/rebuttal/runs');source=base/'018';run=base/'026'
assert not run.exists()
for d in ['code','configs','results','logs','data','reference/validation','reference/calibration']:(run/d).mkdir(parents=True)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
primary=set(read(source/'configs/primary_policies.json')['policies'])
pool=read(source/'configs/allocation_pool.json')
pids=sorted(pid for pid,item in pool['policies'].items() if pid in primary and 'prefill_decode' in item['selection_scenarios'])
assert 'uniform_dense_bf16' in pids
frozen=read(source/'configs/frozen_design.json');datasets={}
for split,count,seed in [('validation',32,26261027),('confirmation',64,26261029)]:
 old=frozen['datasets'][split];tokens=torch.load(old['source'],weights_only=True)
 candidates=sorted(set(range(0,tokens.numel()-2112+1,2112))-set(old['starts']))
 starts=random.Random(seed).sample(candidates,count)
 tensor=torch.stack([tokens[i:i+2112] for i in starts])
 target=run/f'data/{split}_{count}_2112.pt';torch.save(tensor,target)
 datasets[split]={'file':str(target.resolve()),'blocks':count,'starts':starts,'excluded_old_starts':old['starts'],'sha256':sha(target),'token_source':old['source'],'token_source_sha256':sha(Path(old['source'])),'seed':seed}
for pid in pids:
 shutil.copy2(source/f'results/validation/{pid}/prefill_decode.json',run/f'reference/validation/{pid}.json')
 shutil.copy2(source/f'results/measurements/calibration_{pid}.json',run/f'reference/calibration/{pid}.json')
selected=[p for p in read(source/'configs/selected_policies.json')['selected'] if p['scenario']=='prefill_decode']
design={'created_at_unix':time.time(),'source_run':'018','scenario':'prefill_decode','budgets':[.01,.04],'policies':pids,'old_validation_blocks':32,'additional_validation_blocks':32,'guard':'paired mean+2SE+.005; BF16 zero; fastest fixed original calibration E2E','pool_scope':'All source main-table primary policies eligible under the original DASO selector for P+D. Bounded incumbent study, not a full-pool reoptimization.','old_selections':selected,'source_manifest_sha256':sha(source/'results/manifest.json'),'policy_sha256':{p:sha(source/f'configs/policies/{p}.json') for p in pids},'old_validation_sha256':{p:sha(run/f'reference/validation/{p}.json') for p in pids},'calibration_sha256':{p:sha(run/f'reference/calibration/{p}.json') for p in pids},'source_code_sha256':{str(p.relative_to(source/'code')):sha(p) for p in (source/'code').rglob('*') if p.is_file()},'scope':'Prospective sample-size follow-up after observing old outcomes. No fitting/margin/precision change; new confirmation windows disjoint from source018. Old main table stays sealed. Speed/full tasks inherit exact same effective maps from the audited source campaign; no new independent downstream test claim.'}
(run/'configs/design.json').write_text(json.dumps(design,indent=2)+'\n')
(run/'configs/frozen_design.json').write_text(json.dumps({'datasets':datasets},indent=2)+'\n')
shutil.copy2(Path('/tmp/prepare_daso026.py'),run/'code/prepare_data.py')
(run/'README.md').write_text('# Validation样本量敏感性：Llama-2 P+D（R1 AK6a / R5 gkP9）\n\n原32窗口保护值使部分近阈值候选被排除，本独立实验增补32个不重叠validation窗口，保留原mean+2SE+.005规则。固定主表内所有原DASO可选PD incumbent子池，必须先复现原32窗口选择；不是完整候选池重新优化。\n\n用64个与018确认窗口不重叠的新窗口评估新选择，先冻结选择再打开确认；旧主表不变。相同map的速度及完整生成继承已审计018同一路径结果，明确不是新增独立下游样本。全部结果保留，若未改善也不回改规则。\n')
print('Prepared026',len(pids),'fixed incumbents, new32 validation and64 confirmation windows')
