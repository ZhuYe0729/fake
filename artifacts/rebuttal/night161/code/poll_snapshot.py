"""Read-only compact snapshot for continuous supervision and failure alerts."""
import json
import time
from pathlib import Path

root=Path(__file__).resolve().parents[1]
def read(path):
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return {'pending_write':True,'file':str(path)}

state=read(root/'status.json')
actions={p.stem:read(p)for p in(root/'actions').glob('*.json')}
path=root/'latency_validation/action.json'
if path.exists():actions['latency_gate']=read(path)
print(json.dumps({'unix':time.time(),'utc':time.strftime('%H:%M:%S UTC',time.gmtime()),'status':state,
    'actions':{k:{a:v for a,v in value.items()if a in ['returncode','seconds','pending_write']}for k,value in actions.items()}}))
