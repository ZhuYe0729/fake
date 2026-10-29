"""Continue frozen downstream work automatically after quiet calibration."""
import sys
import time

from launch import RUN, invoke, read, write


def main():
    while not (RUN / 'configs/selected_reanalysis.json').exists():
        time.sleep(10)
    stages = [('freeze_main.py', 'main_design'), ('prepare_full_quality.py', 'full_quality_design'),
              ('full_quality_queue.py', 'full_quality_queue'), ('formal_queue.py', 'formal_queue')]
    for script, label in stages:
        completed = {'freeze_main.py': RUN / 'configs/design.json',
                     'prepare_full_quality.py': RUN / 'configs/full_quality_jobs.json',
                     'full_quality_queue.py': RUN / 'results/full_quality_complete.json'}
        if script in completed and completed[script].exists():
            continue
        write(RUN / 'results/worker.json', {'status': 'running', 'stage': label})
        invoke([sys.executable, RUN / 'code' / script], label, 0)
    write(RUN / 'results/worker.json', {'status': 'complete', 'stage': 'main_quality_and_speed'})
    invoke([sys.executable, RUN / 'code/analyze_constraints.py'], 'constraints_analysis', 0)
    invoke([sys.executable, RUN / 'code/analyze_hawq.py'], 'hawq_analysis', 0)
    invoke([sys.executable, RUN / 'code/build_report.py'], 'main_report', 0)
    invoke([sys.executable, RUN / 'code/audit_package.py'], 'audit_package', 0)


if __name__ == '__main__':
    main()
