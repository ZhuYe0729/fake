"""Resume after the observed GPU1 startup collision; preserve original failure."""
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
    write(RUN / 'results/training_resume.json', {
        'original_driver_pid': original_pid, 'completed_before_resume': completed,
        'total': len(jobs), 'reason': 'GPU1 model startup collided with another task; original failed logs retained.'})
    invoke([sys.executable, RUN / 'code/driver.py'], 'resumed_driver', 0)


if __name__ == '__main__':
    main()
