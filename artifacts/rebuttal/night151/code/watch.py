"""Read-only stage/process watchdog; heartbeat stays outside sealed runs."""
import datetime
import json
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parent / 'runs'


def main():
    previous = None
    while True:
        processes = list(psutil.process_iter(['pid', 'cmdline']))
        snapshot = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'runs': {}}
        for rid in ['016', '017', '018', '019', '020', '021', '022']:
            base = RUNS / rid
            state = base / 'results/worker.json'
            data = json.loads(state.read_text()) if state.exists() else {'status': 'preparing'}
            drivers = [p.info['pid'] for p in processes
                if any(str(base / 'code/driver.py') == x or (x.endswith(f'runs/{rid}/code/driver.py') or x.endswith(f'runs/{rid}/code/budget_grid.py'))
                       for x in (p.info['cmdline'] or []))]
            entry = {'state': data, 'driver_pids': drivers}
            if data.get('status') == 'running' and not drivers:
                entry['attention'] = 'running state without driver; inspect logs'
            for name in ['training', 'validation', 'calibration', 'confirmation', 'formal', 'full_quality']:
                progress = base / f'results/{name}_progress.json'
                complete = base / f'results/{name}_complete.json'
                if complete.exists():
                    entry[name] = {'status': 'complete', **json.loads(complete.read_text())}
                elif progress.exists():
                    entry[name] = json.loads(progress.read_text())
            for name in ['features', 'profile', 'gpu_preparation']:
                complete = base / f'results/{name}_complete.json'
                if complete.exists():
                    entry[name] = json.loads(complete.read_text())
            snapshot['runs'][rid] = entry
        ROOT.mkdir(parents=True, exist_ok=True)
        target = ROOT / 'status.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(snapshot, indent=2) + '\n')
        temporary.replace(target)
        summary = {rid: {'state': x['state'], 'attention': x.get('attention')}
                   for rid, x in snapshot['runs'].items()}
        if summary != previous:
            with (ROOT / 'stage_events.jsonl').open('a') as handle:
                handle.write(json.dumps({'utc': snapshot['utc'], 'runs': summary}) + '\n')
            print(snapshot['utc'], summary, flush=True)
            previous = summary
        time.sleep(45)


if __name__ == '__main__':
    main()
