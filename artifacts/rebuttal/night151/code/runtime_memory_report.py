"""Keep initialization, sampled occupancy, host caches and cycle costs distinct."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())


def main():
    rows,sources=[],{}
    for rid,model in [('016','Qwen3-8B'),('017','Llama-3.1-8B'),('018','Llama-2-7B')]:
        run=ROOT.parent/'runs'/rid
        choices=[('BF16','uniform_dense_bf16'),('SmoothQuant INT8','uniform_sq_int8'),
            ('HAWQ-FP4 r=.80','hawq_signed_prefill_decode_phase_pair_r0.8'),
            ('Common hardware minimum','hardware_minimum_prefill_decode_phase_pair')]
        choices += [(f'DASO-8 epsilon={p["quality_budget"]:g}',p['policy'])
            for p in read(run/'configs/selected_policies.json')['selected'] if p['scenario']=='prefill_decode']
        for title,pid in choices:
            repeated=[]
            for number in range(3):
                path=run/f'results/speed/node0/round{number}/{pid}.json'
                raw=read(path);sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                case=next(c for c in raw['cases'] if c['output']==64)
                samples=[s for s in case['samples'] if not s['warmup']]
                repeated.append({'init_s':raw['initialization_ms']/1000,
                    'capture_ms':raw['capture_model_ms'],
                    'init_torch_peak_gib':raw['memory']['initialization_peak_allocated_bytes']/1024**3,
                    'process_torch_peak_gib':raw['memory']['torch_peak_allocated_bytes']/1024**3,
                    'sampled_nvml_used_gib':raw['memory']['sampled_nvml_max_used_bytes']/1024**3,
                    'gpu_linear_storage_gib':raw['memory']['gpu_linear_unique_storage_bytes']/1024**3,
                    'gpu_phase_arena_gib':raw['memory']['phase_reused_arena_bytes']/1024**3,
                    'host_weight_cache_gib':raw['memory']['host_weight_bytes']/1024**3,
                    'decode_h2d_gib':raw['transfer_bytes_per_generation']['decode']/1024**3,
                    'restore_ms':statistics.median(s['restore_ms'] for s in samples),
                    'e2e_ms':statistics.median(s['e2e_ms'] for s in samples),
                    'cycle_ms':statistics.median(s['cycle_ms'] for s in samples)})
            rows.append({'model':model,'label':title,'effective_policy':pid,
                **{key:statistics.median(r[key] for r in repeated) for key in repeated[0]}})
    with (ROOT/'three_model_runtime_memory.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (ROOT/'runtime_memory_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    cols=['model','label','init_s','init_torch_peak_gib','process_torch_peak_gib','sampled_nvml_used_gib',
        'gpu_linear_storage_gib','gpu_phase_arena_gib','host_weight_cache_gib','decode_h2d_gib','restore_ms','e2e_ms','cycle_ms']
    lines=['# 三模型阶段表示、传输与完整cycle（R1 AK6a / R3 LrT2 / R5 gkP9）','',
        '各项为三轮中位数，P+D为B8/S2048/O64。GPU0/NUMA0；Qwen/Llama-3.1 KV6GiB，Llama-2 KV10GiB。'
        '初始化Torch峰值与全进程Torch峰值分列，后者包含初始化；未在初始化后重置峰值，不能叫纯运行期峰值。'
        'NVML仅在每次生成结束时采样，报告这些样本的最大已用值，不是连续采样的真实瞬时峰值。'
        '同进程若测PP和P+D，内存统计跨两个场景，不能当成P+D独立峰值。','',
        'Linear storage只统计线性层GPU唯一storage，arena是其中子项；两者均非模型总显存。'
        'host cache为阶段权重唯一storage，不能与pinned或RSS重复相加。E2E不含请求间恢复，cycle包含恢复。','',
        '| Model | Method | Init s | Init Torch peak GiB | Process Torch peak GiB | Sampled NVML GiB | Linear storage GiB | Phase arena GiB | Host weights GiB | Decode H2D GiB | Restore ms | E2E ms | Cycle ms |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:lines.append('| '+' | '.join(row[k] if k in ['model','label'] else f'{row[k]:.3f}' for k in cols)+' |')
    lines += ['', '同一effective policy在多个预算选中时，数值复用同一组真实测量，不是额外独立重复。'
        'capture耗时包含在初始化内，CSV另存但不叠加；decode H2D是按实际阶段buffer统计的字节数，不等于测得的暴露传输时长。', '',
        '[完整CSV与实际map](three_model_runtime_memory.csv) · [逐轮来源SHA](runtime_memory_sources.json)','']
    (ROOT/'RUNTIME_MEMORY_REPORT.md').write_text('\n'.join(lines))
    print('Published',len(rows),'runtime memory rows.')


if __name__=='__main__':main()
