"""Complete pruning quality/speed table, preserving source-run provenance."""
import csv
import importlib.util
import json
import math
import shutil
import statistics

from common import RUN, REPO, write_json
from launch import read


def save(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def speed(origin,pid,scene):
    output=1 if scene=='prefill_only' else 64
    keys=['e2e_ms'] if output==1 else ['ttft_ms','tpot_ms','e2e_ms']
    repeated={key:[] for key in keys}
    for campaign in range(3):
        directory=RUN.parent/origin/('results/speed/node0' if origin=='017' else 'results/speed')/f'round{campaign}'
        data=read(directory/f'{pid}.json')
        timed=[s for c in data['cases'] if c['output']==output for s in c['samples'] if not s['warmup']]
        anchors=[s for name in ['uniform_dense_bf16','uniform_dense_bf16_bracket_end']
            for c in read(directory/f'{name}.json')['cases'] if c['output']==output for s in c['samples'] if not s['warmup']]
        assert len(timed)==8 and len(anchors)==16
        for key in keys:repeated[key].append(statistics.median(s[key] for s in anchors)/statistics.median(s[key] for s in (anchors if pid=='uniform_dense_bf16' else timed)))
    return repeated


def main():
    assert read(RUN/'results/full_quality_complete.json')['complete']
    assert read(RUN/'results/formal_complete.json')['complete']
    spec=importlib.util.spec_from_file_location('quality_reader',RUN.parent/'013/code/write_report.py')
    quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality);quality.RUN=RUN
    rows,provenance,ranges=[],[],[]
    for title,origin,pid in [('BF16','017','uniform_dense_bf16'),('SparseGPT 2:4','017','uniform_sparse_bf16'),
                            ('Wanda 2:4','020','uniform_wanda'),('Magnitude 2:4','020','uniform_magnitude')]:
        row={'method':title,'source_run':origin}
        trace={'method':title,'source_run':origin,'policy':pid,'quality_sources':{}}
        for scene in ['prefill_only','prefill_decode']:
            metrics,sources=quality.quality([origin,pid],scene)
            row.update(metrics);trace['quality_sources'].update(sources)
            raw=read(RUN.parent/origin/f'results/confirmation/{pid}/{scene}.json')
            base=read(RUN.parent/'017'/f'results/confirmation/uniform_dense_bf16/{scene}.json')
            assert raw['data_sha256']==base['data_sha256']
            differences=[a['avg_nll']-b['avg_nll'] for a,b in zip(raw['blocks'],base['blocks'],strict=True)]
            assert len(differences)==64
            row[scene+'_delta_nll']=statistics.mean(differences)
            trace[scene+'_paired_se']=statistics.stdev(differences)/math.sqrt(64)
            for key,values in speed(origin,pid,scene).items():
                column='prefill_ttft_speedup' if scene=='prefill_only' else 'pd_'+key.replace('_ms','_speedup')
                row[column]=statistics.median(values)
                ranges.append({'method':title,'scenario':scene,'metric':key,'source_run':origin,
                    'median_speedup':statistics.median(values),'min_speedup':min(values),'max_speedup':max(values)})
        rows.append(row);provenance.append(trace)
    save(RUN/'results/pruning_table.csv',rows)
    save(RUN/'results/speed_ranges.csv',ranges)
    write_json(RUN/'results/provenance.json',provenance)
    columns=['method','prefill_only_delta_nll','arc_c','arc_e','wino','mmlu','prefill_ttft_speedup',
        'prefill_decode_delta_nll','cnn_dm','dialogsum','iwslt','pd_ttft_speedup','pd_tpot_speedup','pd_e2e_speedup']
    lines=['# Llama-3.1-8B-Instruct剪枝对照（Reviewer QboE / AK6a）','',
        'BF16和原SparseGPT来自封存run017；Wanda与magnitude在run020完整补测。各自使用相同实际SparseBF16格式和kernel、'
        '相同B8/S2048/O1或64、6GiB BF16 KV、GPU0/NUMA0以及三轮3warmup+8timed。速度按各自三轮BF16首尾锚点归一化，跨campaign来源明确保留。','',
        'Wanda使用层序传播；输入按原SparseGPT记录的128×2048 train样本重建。原加载规则、token总数和seed0起点一致，'
        '原tensor未保留，不能声称与不可用原tensor做过字节哈希比较。完整CNN/DM1000、DialogSum1500、IWSLT333均重新生成，不继承其他剪枝算法的质量。','',
        '| Method | PP ΔNLL | ARC-C | ARC-E | Wino | MMLU | Prefill × | PD ΔNLL | CNN/DM | DialogSum | IWSLT | PD TTFT × | TPOT × | E2E × |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:lines.append('| '+' | '.join(str(row[key]) if key=='method' else f'{row[key]:.3f}' for key in columns)+' |')
    lines += ['', '## 新剪枝准备时间','',
        '| Algorithm | Load s | Pruning/calibration s | Total including save s |',
        '| --- | ---: | ---: | ---: |']
    for method in ['wanda','magnitude']:
        meta=read(RUN/f'results/canonical_{method}.json')
        lines.append(f'| {method} | {meta["load_seconds"]:.2f} | {meta["pruning_and_calibration_seconds"]:.2f} | {meta["total_seconds_including_save"]:.2f} |')
    lines += ['', '原SparseGPT是缓存复用，其历史准备成本另报，不能作为零成本。三轮最小/最大范围不是置信区间；'
        '这里比较剪枝方法，不把共享SparseBF16 kernel的相近速度归因于剪枝算法。','',
        '[完整CSV](pruning_table.csv) · [逐项来源](provenance.json) · [三轮范围](speed_ranges.csv) · [冻结设计](design.json)','']
    (RUN/'REPORT.md').write_text('\n'.join(lines).replace('](pruning_table.csv)','](results/pruning_table.csv)').replace('](provenance.json)','](results/provenance.json)').replace('](speed_ranges.csv)','](results/speed_ranges.csv)').replace('](design.json)','](configs/design.json)'))
    output=REPO/'artifacts/rebuttal/responses/reviewer4/pruning_llama31_run020'
    output.mkdir(parents=True,exist_ok=True)
    for name in ['pruning_table.csv','provenance.json','speed_ranges.csv']:shutil.copy2(RUN/'results'/name,output/name)
    shutil.copy2(RUN/'configs/design.json',output/'design.json')
    (output/'README.md').write_text('\n'.join(lines)+'\n[原始代码和结果](../../../runs/020/README.md)\n')
    write_json(RUN/'results/report_complete.json',{'complete':True,'rows':4,'all_full_tasks':True})
    print('Published Llama pruning controls',flush=True)


if __name__=='__main__':main()
