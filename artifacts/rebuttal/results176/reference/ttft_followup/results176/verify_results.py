"""Verify the readable overview, two curated main tables, and preserved sources."""
from pathlib import Path
from html.parser import HTMLParser
import csv
import hashlib
import json
import math
import re

BASE=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
REPO=BASE.parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
records=lambda p:list(csv.DictReader(p.open()))
source=read(OUT/'overview_sources.json')
for relative,expected in source['source_sha256'].items():assert sha(REPO/relative)==expected,relative
# User-approved Qwen table and original experiment manifests must remain byte-identical.
qdir=BASE/'responses/reviewer2/qwen3_main_table'
for name,expected in read(qdir/'audit.json')['output_sha256'].items():assert sha(qdir/name)==expected,name
manifest_expected={'015':'f2373064e52549bdb23fe4748df4130c4df6d5dee90637b6929eed08df5e19b1',
 '016':'7366eec8710a81650875c630c85ba679b8b1116a4d426d1912e44b042286e334',
 '017':'9808665a3f28e213bcf0beef3fd0b928852e49aefc4c809eb624b4d8ebffb29e',
 '018':'0e80903a7a6d532db072b2f2375da200c420aeb4d35e24e830645d4909ad4ad2'}
for rid,expected in manifest_expected.items():assert sha(BASE/f'runs/{rid}/results/manifest.json')==expected,rid

class HTMLTable(HTMLParser):
 def __init__(self):super().__init__();self.inside=False;self.cells=None;self.rows=[]
 def handle_starttag(self,tag,attrs):
  if tag=='tbody':self.inside=True
  if self.inside and tag=='tr':self.cells=0
  if self.inside and tag in ['th','td']:self.cells+=1
 def handle_endtag(self,tag):
  if tag=='tbody':self.inside=False
  if self.inside and tag=='tr':self.rows.append(self.cells)


def blocks(text):return re.findall(r'(?m)^\|.*(?:\n\|.*)+',text)
def parse(block):return [[x.strip() for x in line.strip('|').split('|')] for line in block.splitlines()]
model_audits=[]
for name in ['llama31_main_table','llama2_main_table']:
 d=BASE/'responses/reviewer2'/name;rows=records(d/'main_table.csv');prov=read(d/'provenance.json')
 assert len(rows)==16 and len(prov['rows'])==16
 for r in rows:
  assert all(math.isfinite(float(v)) for k,v in r.items() if k not in ['model','method'])
  assert math.isclose(float(r['prefill_avg_acc']),sum(float(r[k]) for k in ['arc_c','arc_e','wino','mmlu'])/4,abs_tol=1e-12)
  assert math.isclose(float(r['pd_avg_score']),sum(float(r[k]) for k in ['cnn_dm','dialogsum','iwslt'])/3,abs_tol=1e-12)
 by={r['method']:r for r in rows}
 assert all(by['DASO (Max-speed)'][k]==by['HAWQ-V3 (Max-speed)'][k]
  for k in ['arc_c','arc_e','wino','mmlu','prefill_avg_acc','prefill_ttft_speedup'])
 for r in prov['rows']:
  for scene in ['prefill_only','prefill_decode']:
   entry=r[scene]
   mapping=read(BASE/f'runs/{entry["policy_run"]}/configs/policies/{entry["policy"]}.json')['method_map']
   assert hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()==entry['map_sha256']
 html=HTMLTable();html.feed((d/'table.html').read_text());assert html.rows==[6]*16
 bs=blocks((d/'README.md').read_text());assert len(bs)==2 and all(len(parse(b))-2==16 for b in bs)
 model_audits.append({'table':name,'rows':16,'finite_metric_cells':16*13,'averages_checked':32,'shared_prefill_exact':True,'html_columns':6})

text=(BASE/'REBUTTAL_SELECTED_EVIDENCE.md').read_text()
expected=read(OUT/'overview_tables.json');actual=blocks(text)
assert len(expected)==len(actual)==16
for spec,block in zip(expected,actual):
 parsed=parse(block)
 assert parsed[0]==spec['headers'],spec['title']
 assert parsed[2:]==[[str(v) for v in row] for row in spec['rows']],spec['title']
 assert all(len(row)==len(spec['headers']) for row in parsed),spec['title']
visible=re.sub(r'<details>.*?</details>','',text,flags=re.S)
visible=re.sub(r'\]\([^)]+\)',']',visible)
assert not re.search(r'\b(?:w\d+_|uniform_|hardware_minimum_|daso8_|run0\d\d)',visible), 'raw experiment IDs leaked into main view'
assert 'We added' not in visible and '英文回复草稿：' not in visible
assert '<details>' not in visible
# Current documents and all explicit local anchors/links.
files=[BASE/'REBUTTAL_SELECTED_EVIDENCE.md',BASE/'REBUTTAL_RESPONSE_DRAFT.md',BASE/'responses/README.md',
 BASE/'responses/reviewer2/llama31_main_table/README.md',BASE/'responses/reviewer2/llama2_main_table/README.md']
link_count=0;table_count=0
for p in files:
 s=p.read_text();anchors=set(re.findall(r'<a id="([^"]+)"',s))
 for target in re.findall(r'\]\(([^)]+)\)',s):
  if target.startswith(('http://','https://','mailto:')):continue
  file,_,fragment=target.partition('#');dest=p if not file else p.parent/file
  if dest.resolve()==OUT/'audit.json' and not dest.exists():
   assert p==BASE/'REBUTTAL_SELECTED_EVIDENCE.md'
  else:
   assert dest.exists(),(p,target)
  if fragment and dest==BASE/'REBUTTAL_SELECTED_EVIDENCE.md':
   assert fragment in set(re.findall(r'<a id="([^"]+)"',dest.read_text())),target
  link_count+=1
 assert s.count('<details>')==s.count('</details>'),p
 for line in s.splitlines():assert line.count('**')%2==0,(p,line)
 for block in blocks(s):
  rows=parse(block);assert all(len(row)==len(rows[0]) for row in rows),p
 table_count+=len(blocks(s))
# Updating source pointers must not change the English reply body or historical evidence.
old_reply=(OUT/'reference/REBUTTAL_RESPONSE_DRAFT.md').read_text()
new_reply=(BASE/'REBUTTAL_RESPONSE_DRAFT.md').read_text()
quote=lambda s:[line for line in s.splitlines() if line.startswith('>')]
assert quote(old_reply)==quote(new_reply)
old_audit=read(BASE/'selected_evidence173/source_audit.json')
assert sha(BASE/old_audit['document'])==old_audit['document_sha256']
archives=[{'snapshot':str(p.relative_to(BASE)),'sha256':sha(p)} for p in (OUT/'reference').iterdir() if p.is_file()]
delivery=[BASE/'REBUTTAL_SELECTED_EVIDENCE.md',BASE/'REBUTTAL_RESPONSE_DRAFT.md',BASE/'responses/README.md',
 OUT/'build_results.py',OUT/'verify_results.py',OUT/'overview_tables.json',OUT/'overview_sources.json']
for name in ['llama31_main_table','llama2_main_table']:
 delivery.extend(BASE/'responses/reviewer2'/name/part for part in ['README.md','main_table.csv','table.html','provenance.json'])
audit={'verified':True,'source_files_checked':len(source['source_sha256']),'model_tables':model_audits,
 'overview_tables':len(expected),'overview_data_rows':sum(len(s['rows']) for s in expected),'current_documents_checked':len(files),
 'local_links_checked':link_count,'all_document_tables_checked':table_count,'raw_ids_absent_from_main_view':True,
 'english_reply_body_unchanged':True,'qwen_approved_table_unchanged':True,'experiment_manifests_unchanged':manifest_expected,
 'new_gpu_measurements':0,'original_result_mutations':0,'archives':archives,'errors':[],
 'delivery_sha256':{str(p.relative_to(BASE)):sha(p) for p in delivery}}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k not in ['delivery_sha256','experiment_manifests_unchanged','archives']},ensure_ascii=False,indent=2))
