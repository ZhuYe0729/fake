"""Finish CPU reporting after every scheduled worker exits successfully."""
import json
import os
import subprocess
import sys
import time

from common import RUN, write_json


def main():
    expected=['speed','quality1','quality2','helper2','helper6']
    if (RUN/'results/worker_quality0.json').exists():expected.append('quality0')
    while True:
        states={name:json.loads((RUN/f'results/worker_{name}.json').read_text()) for name in expected}
        subprocess.run([sys.executable,str(RUN/'code/summarize.py')],check=True)
        failed={name:s for name,s in states.items() if s['status']=='failed'}
        if failed:raise RuntimeError(json.dumps(failed))
        if all(s['status']=='complete' for s in states.values()):break
        time.sleep(30)
    for name in ['write_report.py','audit_results.py']:
        subprocess.run([sys.executable,str(RUN/'code'/name)],check=True)
    write_json(RUN/'results/ready_to_seal.json',{'complete':True,'pid':os.getpid(),
        'next':'inspect report/figures; update README and impl; run package_results.py after logs settle'})
    print('All scheduled measurement and CPU audits complete.',flush=True)


if __name__=='__main__':main()
