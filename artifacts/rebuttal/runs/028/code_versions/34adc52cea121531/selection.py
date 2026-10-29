"""Keep every measured feasible incumbent; never filter with the proxy alone."""
import math
import statistics


def guard(differences, baseline=False):
    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(len(differences))
    return {'delta_nll': mean, 'paired_se': se, 'guarded_delta_nll': 0. if baseline else mean + 2 * se + .005}


def select(candidates, budget, metric):
    feasible = [p for p in candidates if p['guarded_delta_nll'] <= budget]
    winner = min(feasible, key=lambda p: (p[metric], p['policy']))
    for anchor in feasible:
        if anchor['policy'] in ['uniform_int8', 'uniform_sq_int8', 'uniform_dense_bf16']:
            assert winner[metric] <= anchor[metric]
    return winner, len(feasible)
