"""Verify old labels on the exact samples; recompute rather than mix protocols."""
import argparse
import hashlib
import shutil
import statistics
import sys
import time

from launch import RUN, checkpoint, cleanup, invoke, read, write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    checks, anchors = [], {}
    for method in ['dense_bf16', 'dense_nvfp4']:
        pid = 'uniform_' + method
        model = checkpoint(pid, args.gpu)
        for scene in ['prefill_only', 'prefill_decode']:
            target = RUN / f'results/training/{scene}/{pid}.json'
            if not target.exists():
                partial = RUN / f'results/captures/training/{pid}/{scene}'
                if partial.exists():
                    partial.rename(partial.with_name(scene+'_failed_'+str(time.time_ns())))
                invoke([sys.executable, RUN / 'code/score_nll.py', '--checkpoint', model,
                    '--policy', pid, '--split', 'training', '--scenarios', scene], 'cache_gate_'+scene+'_'+method, args.gpu)
            manifest = read(RUN / f'configs/calibration_{scene}.json')
            match = []
            for entry in manifest[:72]:
                mapping = read(RUN / f'configs/policies/{entry["policy_id"]}.json')['method_map']
                if all(p['prefill_method'] == method and p['decode_method'] == method for p in mapping.values()):
                    match.append(entry['policy_id'])
            assert len(match) == 1
            old = read(RUN / f'results/training/{scene}/{match[0]}.json')
            new = read(target)
            differences = [a['avg_nll']-b['avg_nll'] for a,b in zip(new['blocks'],old['blocks'],strict=True)]
            assert len(differences) == 100
            check = {'scenario': scene, 'method': method, 'mean_difference': statistics.mean(differences),
                'max_absolute_block_difference': max(map(abs,differences))}
            check['passed'] = abs(check['mean_difference']) < 1e-4 and check['max_absolute_block_difference'] < 1e-3
            checks.append(check)
            anchors[scene, match[0]] = target
        cleanup(model, 'cache_gate_'+method)
    reuse = all(c['passed'] for c in checks)
    if not reuse:
        frozen = RUN / 'configs/frozen_design.json'
        shutil.copy2(frozen, RUN / 'reference/design_before_cache_recompute.json')
        jobs = read(RUN / 'configs/training_jobs.json')
        for scene in ['prefill_only', 'prefill_decode']:
            path = RUN / f'configs/calibration_{scene}.json'
            manifest = read(path)
            shutil.copy2(path, RUN / f'reference/calibration_{scene}_before_recompute.json')
            for entry in manifest[:72]:
                pid = entry['policy_id']
                target = RUN / f'results/training/{scene}/{pid}.json'
                saved = RUN / f'reference/inherited_label_links/{scene}/{pid}.json'
                saved.parent.mkdir(parents=True, exist_ok=True)
                assert target.is_symlink()
                target.rename(saved)
                entry['original_label_sha256'] = entry.pop('source_sha256')
                entry['origin'] = 'paper_recomputed'
                if (scene,pid) in anchors:
                    shutil.copy2(anchors[scene,pid], target)
                else:
                    jobs.append({'job_id': pid, 'policy': pid, 'scenarios': [scene],
                        'policy_sha256': entry['policy_sha256']})
            write(path, manifest)
        write(RUN / 'configs/training_jobs.json', jobs)
        design = read(frozen)
        design['label_cache_reused'] = False
        design['training_jobs_sha256'] = sha(RUN / 'configs/training_jobs.json')
        design['training_manifest_sha256'] = {s:sha(RUN / f'configs/calibration_{s}.json')
            for s in ['prefill_only','prefill_decode']}
        design['cache_amendment_scope'] = 'Preregistered cache-equivalence failure: same policies, data, train/holdout and hyperparameters; recompute old labels before fitting. Old sources retained.'
        write(frozen, design)
    write(RUN / 'results/label_cache_decision.json', {'complete': True, 'reuse': reuse, 'checks': checks,
        'scope': 'Both target-model BF16 and NVFP4 parity checked before fitting; cache mismatch triggers fresh labels for all old policies, without removing failed evidence.'})
    print('Old target-model labels reused:', reuse, checks, flush=True)


if __name__ == '__main__':
    main()
