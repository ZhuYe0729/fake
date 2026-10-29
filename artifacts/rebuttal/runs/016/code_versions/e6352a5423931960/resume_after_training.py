"""Resume the training stage after diagnosed interruptions; retain failures."""
import os
import sys
import time

from launch import RUN, invoke, read, write


def main():
    original_pid = int(sys.argv[1])
    while True:
        try:
            os.kill(original_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(10)
    jobs = read(RUN / 'configs/training_jobs.json')
    completed = sum((RUN / f'results/completed/training/{j["job_id"]}.json').exists() for j in jobs)
    previous = RUN / 'results/training_resume.json'
    if previous.exists():
        write(RUN / f'results/recovery_history/{time.time_ns()}.json', read(previous))
    write(RUN / 'results/training_resume.json', {
        'original_driver_pid': original_pid, 'completed_before_resume': completed,
        'total': len(jobs), 'reason': 'Resume after GPU collision / diagnosed NVFP4 decode arena assertion; failed logs retained.'})
    invoke([sys.executable, RUN / 'code/driver.py'], 'resumed_driver', 0)


if __name__ == '__main__':
    main()
