"""Check delivery links, source seals, policy joins and derived claims."""
from pathlib import Path
import csv
import hashlib
import json
import re

BASE=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
rows=lambda p:list(csv.DictReader(p.open()))
analysis=read(OUT/'analysis_audit.json')
old_audit=read(BASE/'selected_evidence173/source_audit.json')
source_hashes={**old_audit['source_sha256'],**analysis['source_sha256']}
for relative,expected in source_hashes.items():
    assert sha(BASE/relative)==expected,relative
# Verify the prespecified analysis against the unchanged raw files and plan/code seals.
gate_dir=BASE/'night161/latency_validation'
seal=read(gate_dir/'preregistered.json')
gate_audit=read(gate_dir/'final_audit.json')
assert gate_audit['verified'] and gate_audit['complete']
assert sha(BASE/'night161/code/latency_gate_analysis.py')==seal['code_sha256']
assert sha(BASE.parent.parent/'dev/plans/171_preregistered_latency_gate_plan.md')==seal['plan_sha256']
for absolute,expected in gate_audit['source_sha256'].items():
    assert sha(Path(absolute))==expected,absolute
selection=read(gate_dir/'selection.json')
assert selection['confirmation_read'] is False
assert all('/matched_round0_' in x for x in selection['calibration_sha256'])
gate=rows(gate_dir/'latency_validation.csv')
assert len(gate)==36 and len({r['selected_policy'] for r in gate})==17
assert all(r['selection_feasible']=='True' and int(r['selected_misses'])==0 and int(r['original_misses'])==0 for r in gate)
assert all(int(r['selected_samples'])==16 for r in gate)
points=rows(OUT/'budget_speed.csv')
assert len(points)==6
assert all(float(r['confirmation_mean_plus_two_se'])<float(r['epsilon']) for r in points)
assert [f"{float(r['inherited_e2e_speedup_bf16']):.3f}" for r in points if float(r['epsilon'])==.04]==['2.061','1.891','1.770']
counts=rows(OUT/'budget_counts.csv')
assert sum(int(r['budget_labels']) for r in counts)==24
assert sum(int(r['unique_selected_model_policies']) for r in counts)==14
assert sum(int(r['mean_violations']) for r in counts)==0
assert sum(int(r['confirmation_guard_violations']) for r in counts)==3
plot=rows(OUT/'workload_figure_data.csv')
assert len(plot)==24 and all(float(r['median_ratio'])>1 for r in plot)
assert all(float(r['minimum_ratio'])<=float(r['median_ratio'])<=float(r['maximum_ratio']) for r in plot)
files=[BASE/'REBUTTAL_SELECTED_EVIDENCE.md',BASE/'REBUTTAL_RESPONSE_DRAFT.md',BASE/'responses/README.md',OUT/'FIGURE.md',BASE/'responses/reviewer5/latency_validation_night161/README.md']
errors=[];documents=[]
for p in files:
    lines=p.read_text().splitlines();links=re.findall(r'\]\(([^)]+)\)',p.read_text())
    for target in links:
        if target.startswith(('https://','http://','#','mailto:')):continue
        target=target.split('#',1)[0]
        if not (p.parent/target).exists():errors.append({'file':str(p),'broken_link':target})
    tables=0;data_rows=0;i=0
    while i<len(lines):
        if lines[i].startswith('|'):
            start=i;columns=len(lines[i].split('|'))-2
            while i<len(lines) and lines[i].startswith('|'):
                if len(lines[i].split('|'))-2!=columns:errors.append({'file':str(p),'bad_table_line':i+1})
                i+=1
            if start>0 and lines[start-1].strip():errors.append({'file':str(p),'missing_blank_line':start+1})
            tables+=1;data_rows+=i-start-2
        else:i+=1
    for lineno,line in enumerate(lines,1):
        if line.count('**')%2:errors.append({'file':str(p),'unbalanced_emphasis':lineno})
    documents.append({'file':str(p.relative_to(BASE)),'links':len(links),'tables':tables,'data_rows':data_rows})
assert not errors,errors
words={}
for section in re.split(r'(?m)^## ',(BASE/'REBUTTAL_RESPONSE_DRAFT.md').read_text())[1:]:
    paras=[line[2:] for line in section.splitlines() if line.startswith('> ')]
    if paras:
        words[section.splitlines()[0]]=len(re.findall(r'\b[\w]+(?:[./-][\w]+)*\b',' '.join(paras)))
assert len(words)==5
old_audit['document_sha256']=sha(BASE/'REBUTTAL_SELECTED_EVIDENCE.md')
old_audit['verification']={'verified':True,'source_files_checked':len(source_hashes),
    'documents_checked':documents,'errors':errors,'plan174_audit':'response174/final_audit.json',
    'scope':'All retained input hashes and new source seals checked; new policy-matched associations and CPU analysis explicitly separated from GPU experiments.'}
(BASE/'selected_evidence173/source_audit.json').write_text(json.dumps(old_audit,ensure_ascii=False,indent=2)+'\n')
deliveries=['REBUTTAL_SELECTED_EVIDENCE.md','REBUTTAL_RESPONSE_DRAFT.md','selected_evidence173/source_audit.json',
    'response174/build_evidence.py','response174/verify_delivery.py','response174/budget_speed.csv','response174/budget_counts.csv',
    'response174/latency_gate_summary.csv','response174/workload_figure_data.csv','response174/qwen_six_workloads.pdf',
    'response174/qwen_six_workloads.svg','response174/qwen_six_workloads.png','response174/FIGURE.md','response174/analysis_audit.json']
final={'verified':True,'source_files_checked':len(source_hashes),'raw_latency_source_files_checked':len(gate_audit['source_sha256']),
    'documents':documents,'english_words_including_optional':words,'budget_labels':24,'unique_selected_model_policies':14,
    'mean_budget_violations':0,'confirmation_guard_violations':3,'matched_budget_speed_points':6,
    'preregistered_latency_labels':36,'latency_unique_maps':17,'later_round_overruns':0,'original_overruns_same_caps':0,
    'new_gpu_measurements':0,'figure_visual_inspection':'passed','errors':errors,
    'delivery_sha256':{relative:sha(BASE/relative) for relative in deliveries}}
(OUT/'final_audit.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:final[k] for k in ['verified','source_files_checked','raw_latency_source_files_checked','documents','english_words_including_optional','errors']},ensure_ascii=False,indent=2))
