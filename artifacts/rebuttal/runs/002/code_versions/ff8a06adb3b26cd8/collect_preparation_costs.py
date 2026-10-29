import json
import re
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def main():
    result={'inherited_setup':'../001/results/preparation_costs.json',
        'scope':'incremental run002 preparation; inherited 72 labels per scenario and canonical SparseGPT states are not free from scratch'}
    result['activation_scales_and_local_errors']=json.loads((RUN/'results/smoothing_metadata.json').read_text())
    for scenario in ['prefill_only','prefill_decode']:
        manifest=json.loads((RUN/f'configs/calibration_{scenario}.json').read_text())
        split={p['policy_id']:p['split'] for p in manifest}
        records=[];ends=[]
        for log in sorted((RUN/'logs').glob(f'calibration_{scenario}_slot*.log')):
            ends.append(log.stat().st_mtime)
            for policy,seconds in re.findall(r'(cal8_\w+_p\d+) completed ([0-9.]+)',log.read_text()):
                records.append({'policy_id':policy,'split':split[policy],'seconds':float(seconds),'source':str(log.relative_to(RUN))})
        assert len(records)==48 and len({r['policy_id'] for r in records})==48
        begins=[p.stat().st_mtime for p in (RUN/'logs').glob(f'cal8_{scenario}_p*_export.command.json')]
        result[scenario]={'new_labels':48,'training_labels':36,'holdout_labels':12,
            'parallel_wall_seconds':max(ends)-min(begins),'allocated_gpus':3,
            'sum_label_elapsed_seconds':sum(r['seconds'] for r in records),
            'training_sum_label_elapsed_seconds':sum(r['seconds'] for r in records if r['split']=='train'),
            'holdout_sum_label_elapsed_seconds':sum(r['seconds'] for r in records if r['split']=='holdout'),
            'label_scope':'export, model initialization, NLL evaluation, metadata preservation and temporary binary release',
            'records':records}
    for family in ['daso8','daso5_measured']:
        for scenario in ['prefill_only','prefill_decode']:
            result[family+'/'+scenario]=json.loads((RUN/f'results/{family}/{scenario}/fit_search_cost.json').read_text())
    (RUN/'results/preparation_costs.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
