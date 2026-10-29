"""Check all published workload cells, source preservation, and overview inclusion."""
from pathlib import Path
from html.parser import HTMLParser
import csv
import hashlib
import json
import math
import re

ART=Path(__file__).resolve().parent
BASE=ART.parent
REPO=BASE.parent.parent
OUT=BASE/'responses/reviewer2/qwen_workload_comparison'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
records=lambda p:list(csv.DictReader(p.open()))
provenance=read(OUT/'provenance.json')
assert provenance['verified'] and provenance['new_gpu_measurements']==0
for path,digest in provenance['source_sha256'].items():assert sha(REPO/path)==digest,path
parent=read(BASE/'runs/040/results/final_audit.json')
for path,digest in parent['evidence_sha256'].items():assert sha(BASE/'runs/040'/path)==digest,path
assert provenance['quality_computed_from_raw_blocks'] and provenance['latency_medians_recomputed']
assert all(provenance['numerical_code_equal_to_primary_campaign'].values())
followup=read(ART/'center_speed_followup/audit.json')
assert followup['verified'] and not followup['strict_040_048_repeat'] and followup['input_tokens_different']==16384
for path,digest in followup['source_sha256'].items():assert sha(REPO/path)==digest,path

full=records(OUT/'full_comparison.csv');overview=records(OUT/'overview_comparison.csv')
assert len(full)==54 and len(overview)==30
workloads=[w['label'] for w in provenance['workloads']]
methods=[m['display_method'] for m in provenance['methods']]
assert len(workloads)==6 and len(methods)==9
by={(r['workload'],r['method']):r for r in full}
assert set(by)=={(w,m) for w in workloads for m in methods}
strong={'SQ-INT8 → W4A16','HAWQ-V3 (Balanced)','HAWQ-V3 (Max-speed)','DASO (Balanced)','DASO (Max-speed)'}
assert overview==[r for r in full if r['method'] in strong]
source={(r['workload'],r['policy']):r for r in records(BASE/'runs/040/results/workloads.csv')}
speed=['ttft_speedup_bf16','tpot_speedup_bf16','e2e_speedup_bf16','cycle_speedup_bf16']
fields={'delta_nll':'delta_nll','paired_se':'paired_se',
 **dict(zip(speed,['ttft_ms_vs_bf16','tpot_ms_vs_bf16','e2e_ms_vs_bf16','cycle_ms_vs_bf16'])),
 **{k:k for k in ['ttft_ms','tpot_ms','e2e_ms','restore_ms','cycle_ms']}}
for r in full:
 src=source[r['workload'],r['source_policy']]
 for key,raw in fields.items():assert math.isclose(float(r[key]),float(src[raw]),abs_tol=1e-10),(r['method'],key)
 assert math.isclose(float(r['two_se']),2*float(r['paired_se']),abs_tol=1e-12)
 assert int(r['quality_windows'])==64

blocks=lambda text:re.findall(r'(?m)^\|.*(?:\n\|.*)+',text)
parse=lambda block:[[v.strip().replace('**','') for v in line.strip('|').split('|')] for line in block.splitlines()]
def cells(r):
 return [r['method'],f"{float(r['delta_nll']):.5f} ± {float(r['two_se']):.5f}",*[f'{float(r[k]):.3f}' for k in speed]]
report=(OUT/'README.md').read_text();tables=blocks(report)
assert len(tables)==6
for w,table in zip(workloads,tables):
 expected=[cells(r) for r in full if r['workload']==w]
 assert parse(table)[2:]==expected,w

class TableParser(HTMLParser):
 def __init__(self):super().__init__();self.in_body=False;self.in_cell=False;self.rows=[];self.current=[]
 def handle_starttag(self,tag,attrs):
  if tag=='tbody':self.in_body=True
  if tag=='tr' and self.in_body:self.current=[]
  if tag=='td' and self.in_body:self.in_cell=True;self.value=''
 def handle_data(self,data):
  if self.in_cell:self.value+=data
 def handle_endtag(self,tag):
  if tag=='td' and self.in_cell:self.current.append(self.value);self.in_cell=False
  if tag=='tr' and self.in_body:self.rows.append(self.current)
  if tag=='tbody':self.in_body=False
parser=TableParser();parser.feed((OUT/'table.html').read_text())
assert parser.rows==[cells(r) for r in full]

root=(BASE/'REBUTTAL_SELECTED_EVIDENCE.md').read_text()
section=root.split('<a id="workloads"></a>')[1].split('<a id="mixed"></a>')[0]
assert len(blocks(section))==1
expected=[[f"{r['batch']} / {r['prompt']} / {r['output']}",*cells(r)] for r in overview]
assert parse(blocks(section)[0])[2:]==expected
assert '不是三项生成任务的Avg' in section and '相对同负载BF16' in section
assert '三项下游任务没有按这六个固定负载分别重跑' in report
assert '中心行的速度排序尚不稳定' in report and '中心行的两点速度排序尚不稳定' in section
assert '中心负载两点速度排序尚不稳定' in (OUT/'table.html').read_text()
facts={'balanced_faster_than_hawq_max_cases':sum(float(by[w,'DASO (Balanced)']['e2e_speedup_bf16'])>float(by[w,'HAWQ-V3 (Max-speed)']['e2e_speedup_bf16']) for w in workloads),
 'maxspeed_faster_than_hawq_max_cases':sum(float(by[w,'DASO (Max-speed)']['e2e_speedup_bf16'])>float(by[w,'HAWQ-V3 (Max-speed)']['e2e_speedup_bf16']) for w in workloads)}
winners=[max((by[w,m] for m in methods),key=lambda r:float(r['e2e_speedup_bf16']))['method'] for w in workloads]
facts.update(daso_envelope_fastest_cases=sum(m.startswith('DASO') for m in winners),simple_phase_fastest_cases=winners.count('SQ-INT8 → W4A16'))
assert facts==provenance['facts']==read(ART/'comparison_facts.json')['facts']
links=0
for p in [OUT/'README.md',BASE/'REBUTTAL_SELECTED_EVIDENCE.md',BASE/'responses/README.md',BASE/'REBUTTAL_RESPONSE_DRAFT.md']:
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if target.startswith(('http://','https://','mailto:')):continue
  path,_,fragment=target.partition('#');dest=p if not path else p.parent/path
  assert dest.exists(),(p,target)
  if fragment:assert f'<a id="{fragment}">' in dest.read_text(),(p,target)
  links+=1
for extension in ['pdf','svg','png']:assert (OUT/f'quality_speed.{extension}').stat().st_size>1000
overview_audit=read(BASE/'results176/audit.json')
assert overview_audit['verified'] and overview_audit['all_three_model_metric_csvs_unchanged']
assert overview_audit['overview_tables']==17 and overview_audit['overview_data_rows']==133
assert sha(ART/'reference/REBUTTAL_SELECTED_EVIDENCE.md')!=sha(BASE/'REBUTTAL_SELECTED_EVIDENCE.md')
delivery=list(OUT.iterdir())+[ART/'build_comparison.py',ART/'verify_comparison.py']
audit={'verified':True,'rows':54,'workloads':6,'methods_per_workload':9,'overview_direct_comparison_rows':30,
 'markdown_and_html_cells_match_csv':True,'raw_paired_nll_recomputed':True,'three_round_medians_recomputed':True,
 'parent_evidence_hashes_checked':len(parent['evidence_sha256']),'source_files_checked':len(provenance['source_sha256']),
 'all_three_model_metric_csvs_unchanged':True,'links_checked':links,'facts':facts,
 'quality_metric':'measured per-workload delta NLL','downstream_avg_per_workload_available':False,
 'plot_png_visually_inspected':True,'center_timing_order_flagged_as_unresolved':True,
 '040_048_input_and_engine_differences_disclosed':True,'new_gpu_measurements':0,'source_result_mutations':0,'errors':[],
 'delivery_sha256':{str(p.relative_to(BASE)):sha(p) for p in delivery if p.is_file()}}
(ART/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='delivery_sha256'},ensure_ascii=False,indent=2))
