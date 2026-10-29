"""Independently recompute exported policy costs from fitted coefficients and local errors."""
import collections
import csv
import json

from common import RUN,write_json
from daso.modeling.quality import FUSED_TYPES
from daso.policy import METHODS


def main():
    with (RUN/'data/local_errors.csv').open() as handle:
        errors={(row['module'],row['phase'],row['method']):float(row['output_rel_mse']) for row in csv.DictReader(handle)}
    for scenario in ['prefill_only','prefill_decode']:
        root=RUN/f'results/daso/{scenario}'
        if not (root/'predicted_points.json').exists():continue
        model=json.loads((root/'quality_model.json').read_text());rows=[]
        for point in json.loads((root/'predicted_points.json').read_text()):
            policy=json.loads((RUN/f'configs/policies/{point["policy_id"]}.json').read_text())
            score=model['bias'];counts=collections.Counter();zero_cost=collections.Counter()
            for name,pair in policy['method_map'].items():
                for phase_index,phase in enumerate(['prefill'] if scenario=='prefill_only' else ['prefill','decode']):
                    method=pair[phase+'_method'];counts[phase+':'+method]+=1
                    if method=='dense_bf16':continue
                    coefficient=model['coefficients'][phase_index][METHODS.index(method)][int(name.split('.')[2])*4//36][FUSED_TYPES.index(name.rsplit('.',1)[-1])]
                    cost=errors[name,phase,method]*coefficient/model['feature_scale']
                    score+=cost
                    if cost==0:zero_cost[phase]+=1
            difference=score-point['predicted_delta_nll'];assert abs(difference)<1e-12
            rows.append({'policy_id':point['policy_id'],'counts':dict(counts),
                'zero_predicted_cost_compressed_actions':dict(zero_cost),
                'recomputed_prediction':score,'integration_difference':difference})
        write_json(root/'solver_quality_integration_audit.json',rows)
        print(scenario,len(rows),'policies; integration agrees below 1e-12',flush=True)


if __name__=='__main__':main()
