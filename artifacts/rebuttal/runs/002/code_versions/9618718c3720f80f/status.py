"""Compact, read-only progress for the long-running full-dataset queues."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def main():
    out={'utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'speed':len(list((RUN/'results/speed_complete').glob('*.json'))),
        'test_nll':len([p for p in (RUN/'results/nll').glob('*.json') if 'phase_trace' not in p.name]),
        'full_tasks':len(list((RUN/'results/tasks').glob('*/*/full/result.json'))),
        'full_generation':len(list((RUN/'results/generation').glob('*/*/metrics_0000_end.json')))}
    out['queues']={}
    for p in sorted((RUN/'results').glob('queue_*.json')):
        value=json.loads(p.read_text())
        if 'status' not in value:continue
        process=Path(f'/proc/{value["pid"]}/stat')
        out['queues'][p.stem]={'status':value['status'],
            'os':process.read_text().split()[2] if process.exists() else 'exited'}
    active=[]
    for p in sorted((RUN/'logs').glob('*tasks*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:3]:
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-1000));tail=f.read().decode(errors='replace')
        matches=re.findall(r'(\d+)/(\d+) \[([^\]]+)\]',tail)
        if matches:active.append({'log':p.name,'progress':'/'.join(matches[-1][:2]),'timing':matches[-1][2]})
    out['latest_task_logs']=active
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()
