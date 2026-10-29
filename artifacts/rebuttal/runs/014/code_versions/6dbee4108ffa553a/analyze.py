"""Build diagnosis evidence from raw traces and all retained controls."""
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
POLICIES = ['uniform_dense_bf16', 'uniform_w4a16_ours', 'uniform_marlin_int4']


def read(path):
    return json.loads(path.read_text())


def csv_file(name, rows):
    with (RUN / 'results' / name).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    controls = []
    for p in sorted((RUN / 'results/generation_controls').glob('*.json')):
        if p.name.endswith('_trace.json'):
            continue
        d = read(p)
        assert d['complete'] and len(d['samples']) == 11
        timed = [x for x in d['samples'] if not x['warmup']]
        controls.append({'label': p.stem, 'policy': d['policy'], 'mode': d['mode'],
            **d['medians'], 'timed_output_variants': d['timed_output_variants'],
            'median_cpu_mhz_after': statistics.median(x['gpu_after']['host_cpu_khz'] / 1000 for x in timed),
            'median_gpu_mhz_after': statistics.median(x['gpu_after']['sm_mhz'] for x in timed)})
    csv_file('generation_controls.csv', controls)
    fixed, counts = [], []
    for policy in POLICIES:
        d = read(RUN / f'results/{policy}/summary.json')
        assert d['complete'] and d['graph_max_abs_diff'] == 0
        assert d['fixed_inputs']['positions'] == [2110] * 8
        assert d['metadata_fields']['seq_lens'] == [2111] * 8
        events = read(RUN / f'results/{policy}/fixed_eager_trace.json')['traceEvents']
        cpu = Counter(e['name'] for e in events if e.get('ph') == 'X' and e.get('cat') == 'cpu_op')
        kernels = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'kernel']
        categories = Counter()
        for e in kernels:
            n = e['name'].lower()
            category = ('linear' if any(x in n for x in ['marlin', 'gemm', 'matmul']) else
                        'attention' if any(x in n for x in ['flash', 'attention', 'paged']) else 'other')
            categories[category] += e['dur'] / 3000
        fixed.append({'policy': policy, 'eager_ms': d['fixed_eager']['wall_ms'],
            'direct_dispatch_after_profiler_ms': d['fixed_direct_dispatch']['wall_ms'],
            'graph_ms': d['fixed_graph']['wall_ms'], 'graph_max_abs_diff': d['graph_max_abs_diff'],
            'profile_gpu_kernel_sum_ms': sum(e['dur'] for e in kernels) / 3000,
            'profile_linear_ms': categories['linear'], 'profile_attention_ms': categories['attention'],
            'profile_other_ms': categories['other']})
        counts.append({'policy': policy, 'aten_empty_per_forward': cpu['aten::empty'] / 3,
                       'gpu_kernels_per_forward': len(kernels) / 3})
    csv_file('fixed_decode.csv', fixed)
    csv_file('operator_counts.csv', counts)
    paired = []
    for policy in POLICIES:
        path = RUN / f'results/paired/{policy}.json'
        if not path.exists():
            continue
        d = read(path)
        assert d['complete'] and len(d['samples']) == 22
        for mode in ['eager', 'graph']:
            rows = [x for x in d['samples'] if not x['warmup'] and x['mode'] == mode]
            assert len(rows) == 8 and all(x['decode_calls'] == 63 for x in rows)
            paired.append({'policy': policy, 'mode': mode, **d['medians'][mode],
                'min_tpot_ms': min(x['tpot_ms'] for x in rows),
                'max_tpot_ms': max(x['tpot_ms'] for x in rows),
                'output_variants': len({x['output_sha256'] for x in rows})})
    if paired:
        csv_file('paired_generation.csv', paired)
    history = []
    for r in range(3):
        for policy in [*POLICIES, 'uniform_dense_bf16_bracket_end',
                       'hawq_fp4_phase_pair_signed_prefill_decode_q0.07']:
            d = read(RUN.parent / f'013/results/speed/node0/round{r}/{policy}.json')
            rows = [x for c in d['cases'] if c['output'] == 64 for x in c['samples'] if not x['warmup']]
            history.append({'round': r, 'policy': policy,
                'tpot_ms': statistics.median(x['decode_remainder_ms'] / 63 for x in rows),
                'cpu_mhz_after': statistics.median(x['gpu_after']['host_cpu_khz'] / 1000 for x in rows),
                'gpu_mhz_after': statistics.median(x['gpu_after']['sm_mhz'] for x in rows)})
    csv_file('historical_cpu_telemetry.csv', history)
    # Verify the sealed prior experiment rather than modifying it.
    original = RUN.parent / '013'
    manifest = read(original / 'results/manifest.json')['files']
    for name, entry in manifest.items():
        path = original / name
        assert path.stat().st_size == entry['bytes'], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], name
    formal = [x for x in controls if not x['label'].endswith('_probe')]
    complete = len(formal) == 18 and len(paired) == 6
    audit = {'complete': complete, 'fixed_decode_policies': len(fixed),
             'generation_control_processes': len(formal), 'paired_processes': len(paired) // 2,
             'protected_run013_files_verified': len(manifest),
             'fixed_graph_max_abs_diff': max(x['graph_max_abs_diff'] for x in fixed),
             'scope': 'Uniform methods only; no replacement of mixed DASO/HAWQ main-table timings.'}
    (RUN / 'results/audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if complete:
        lines = ['# Run014: W4A16 TPOT diagnosis', '',
            '结论：W4A16 的 GPU 计算收益存在。原 eager 完整执行受到 CPU 发射、临时张量操作等开销限制；'
            'CPU 频率变化进一步影响跨进程排名。原 run013 的 TPOT 不能代表优化部署后的量化性能。', '',
            '## 固定真实 KV 的控制', '',
            'GPU0 / NUMA0 CPU0–7，B8；从真实生成末步冻结输入、positions=2110、seq_lens=2111。'
            '包括模型全部层和 lm_head，排除调度/采样。每项5组×30次，计时无 profiler。', '',
            '| 方法 | 普通前向 ms | Graph 前向 ms | Graph 最大 logits 绝对差 |',
            '| --- | ---: | ---: | ---: |']
        labels = dict(zip(POLICIES, ['BF16', 'W4A16 NVFP4', 'W4A16 INT4 Marlin']))
        for x in fixed:
            lines.append(f'| {labels[x["policy"]]} | {x["eager_ms"]:.3f} | {x["graph_ms"]:.3f} | {x["graph_max_abs_diff"]:.1f} |')
        lines += ['', '单次 eager/direct-dispatch 测量受顺序与 CPU 频率影响；direct-dispatch 位于 profiler 之后，'
                  '不能单凭该对照估计阶段管理的因果耗时。下面同进程交替实验是更强的 Graph 干预。', '',
                  '## 正常生成，同进程交替开关 Graph', '',
                  '原生 vLLM FULL_DECODE_ONLY，compilation level=0；B8/S2048/O64、6GiB BF16 KV，'
                  '与 run013 相同输入。均匀策略关闭无用的阶段切换。每个模型只初始化一次，'
                  '两模式各3次 warmup，随后4组 ABBA/BAAB，共每模式8次正式生成。'
                  '只切换 decode wrapper 的 replay / runnable，保持权重、引擎和 metadata 构造一致。', '',
                  '| 方法 | Eager TPOT ms | Graph TPOT ms | Graph 下相对 BF16 |',
                  '| --- | ---: | ---: | ---: |']
        by_pair = {(x['policy'], x['mode']): x for x in paired}
        bf = by_pair[('uniform_dense_bf16', 'graph')]['tpot_ms']
        for policy in POLICIES:
            e, g = by_pair[(policy, 'eager')], by_pair[(policy, 'graph')]
            lines.append(f'| {labels[policy]} | {e["tpot_ms"]:.3f} | {g["tpot_ms"]:.3f} | {bf/g["tpot_ms"]:.3f}× |')
        lines += ['', '另有两轮倒序的独立进程控制：三模型 × eager / frozen-eager / native-graph，'
                  '18个进程，每进程3 warmup+8 timed。全部保留在 generation_controls.csv；探路 Graph 不计入两轮汇总。', '',
                  '## 运行时证据与限制', '',
                  '- 固定前向每次包含688个 GPU kernel、144个被量化 Linear；量化缩短了 Linear kernel，'
                  '完整 eager 仍逐次经过 Python/C++ 调用，不能把权重压缩倍率等同于 TPOT 加速倍率。',
                  '- 独立 profiler 的 aten::empty 调用/前向：' + '；'.join(
                      f'{labels[x["policy"]]}={x["aten_empty_per_forward"]:.0f}' for x in counts) +
                  '。这是张量创建算子次数，包含零长度占位张量，不表示相同次数的 cudaMalloc。'
                  'INT4 通用绑定存在较多这类 CPU 操作；未将 profiler 下的耗时直接代入正式 TPOT。',
                  '- 原 run013 BF16 round2 首尾：CPU约2.1/2.7GHz时 TPOT约18.04/13.97ms；'
                  'GPU时钟相近。频率只在计时外采样，属于关联证据；没有单独锁 CPU 时钟证明精确因果份额。',
                  '- 均匀 W4A16 的 host 权重缓存为零、前后阶段共享 GPU 权重；结果不支持逐 token H2D 搬权重的解释。',
                  '- 三模型固定输入 Graph logits 与 eager 最大绝对差均为零；完整生成 token 变化按原始记录保留，'
                  '尤其不以 INT4 已知的生成非确定性筛除不利样本。',
                  '- Graph 支持在实验局部实现，未修改共享 vLLM/CUDA kernel、量化权重或原表数值。'
                  '均匀策略可直接使用原生 Graph；DASO/HAWQ 的阶段切换和权重地址生命周期仍需一致处理，'
                  '不能把本轮 Graph 基线与旧 eager 异构策略拼成新的比较主表。', '',
                  '## 文件与复现', '',
                  '- `code/diagnose_decode.py`：真实固定 KV、单独 trace、静态回放。',
                  '- `code/generation_control.py` / `control_queue.py`：正常生成的两轮控制。',
                  '- `code/paired_generation.py`：同进程 ABBA 干预；`code/launch.py` 保存精确命令、代码快照及退出码。',
                  '- `results/fixed_decode.csv`、`generation_controls.csv`、`paired_generation.csv`、'
                  '`operator_counts.csv`、`historical_cpu_telemetry.csv`：从原始 JSON/trace 生成。',
                  '- `results/audit.json`：检查样本数、固定 KV、Graph 误差及原 run013 manifest 未改变。',
                  '- 原理参考：[vLLM CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)、'
                  '[Marlin 作者实现](https://github.com/IST-DASLab/marlin)。实际行为以本机固定代码与测量为准。', '']
        (RUN / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
