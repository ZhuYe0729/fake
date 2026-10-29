"""Explicit child-process accounting; no double-counted controller durations."""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT.parent/'runs'
SCRIPTS={'collect_int8_errors.py':'local_error','collect_features.py':'local_error',
    'hessian_trace.py':'hessian','profile_actions.py':'linear_profile','export_model.py':'export',
    'score_nll.py':'nll','fit_quality.py':'quality_fit','allocate_suite.py':'allocation',
    'benchmark.py':'speed','formal_benchmark.py':'formal_speed',
    'run_workflow.py':'full_mc','generate_tasks.py':'full_generation'}


def save(name,rows):
    with (ROOT/name).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    rows=[]
    for rid in ['016','017','018']:
        for path in sorted((RUNS/rid/'logs').glob('*.command.json')):
            command=json.loads(path.read_text())
            args=command['command']
            script=next((Path(x).name for x in args if x.endswith('.py')),'')
            if script not in SCRIPTS:continue
            completion=path.with_name(path.name.replace('.command.json','.completion.json'))
            if not completion.exists():continue
            done=json.loads(completion.read_text())
            stage=SCRIPTS[script]
            if stage=='nll':stage=args[args.index('--split')+1]+'_nll'
            if stage=='speed':stage='calibration_speed' if any(str(x).startswith('calibration_') for x in args) else 'formal_or_anchor_speed'
            rows.append({'run':rid,'stage':stage,'seconds':done['seconds'],'returncode':done['returncode'],
                'gpu':command['gpu'],'command':str(path.relative_to(RUNS/rid)),
                'command_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'completion_sha256':hashlib.sha256(completion.read_bytes()).hexdigest()})
    grouped=defaultdict(list)
    for row in rows:grouped[row['run'],row['stage']].append(row)
    summary=[]
    for (rid,stage),items in sorted(grouped.items()):
        success=[r for r in items if r['returncode']==0]
        summary.append({'run':rid,'stage':stage,'successful_processes':len(success),
            'sum_success_process_seconds':sum(r['seconds'] for r in success),
            'failed_processes':len(items)-len(success),
            'sum_failed_process_seconds':sum(r['seconds'] for r in items if r['returncode']!=0)})
    save('preparation_processes.csv',rows)
    save('preparation_summary.csv',summary)
    cache_rows=[]
    for rid in ['017','018']:
        for path in sorted((RUNS/rid/'results/primary_weight_cache').glob('*.json')):
            item=json.loads(path.read_text())
            cache_rows.append({'run':rid,'policy':path.stem,'bytes':item['bytes'],
                'copy_and_verify_seconds':item['copy_and_verify_seconds'],
                'weight_sha256':item['sha256'],'record_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if cache_rows:save('weight_cache_operations.csv',cache_rows)
    lines=['# 准备、评测和阶段切换开销（R1 AK6a / R3 LrT2 / R4 QboE / R5 gkP9）','',
        '统计叶子计算命令，排除driver/queue父进程。进程壁钟求和包含CPU、模型载入及GPU执行，'
        '不是实际GPU kernel占用时间，也不是并行后的总墙钟。失败耗时单列；正在运行的未完成命令不计入，不能将中间快照当作完整成本。','',
        '| Run | Stage | Success jobs | Success process-sum seconds | Failed jobs | Failed process-sum seconds |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in summary:lines.append(f'| {row["run"]} | {row["stage"]} | {row["successful_processes"]} | '
        f'{row["sum_success_process_seconds"]:.2f} | {row["failed_processes"]} | {row["sum_failed_process_seconds"]:.2f} |')
    lines += ['', '## 精确权重缓存操作', '',
        '以下是导出后的复制及SHA校验操作，已包含在其父队列壁钟内；与上述叶子进程统计分列，'
        '不能相加作为GPU时间或单次准备总墙钟。缓存仅复用完全相同的导出字节，所有primary方法采用相同规则。', '',
        '| Run | Cached exports | Total GiB | Copy and verify process-sum seconds |',
        '| --- | ---: | ---: | ---: |']
    for rid in ['017','018']:
        selected=[r for r in cache_rows if r['run']==rid]
        lines.append(f'| {rid} | {len(selected)} | {sum(r["bytes"] for r in selected)/1024**3:.2f} | '
            f'{sum(r["copy_and_verify_seconds"] for r in selected):.2f} |')
    lines += ['', '[逐份权重缓存记录](weight_cache_operations.csv)', '',
        '训练标签、validation和速度校准属于搜索/准备成本；confirmation和完整MC/生成是评测成本，分别报告。'
        '反复导出属于本次实验工程流程，既不能当作每请求成本，也不能隐去。','',
        'Qwen016继承每场景120标签、新增24，共144（108 train/36 holdout）；Llama017/018各继承72、新增48，共120（90 train/30 holdout）。'
        '缓存继承不是从零免费：Qwen原Hessian1765.64秒、两种SparseGPT各305秒；旧标签阶段壁钟及额外8bit标签成本详见run004，'
        '旧阶段相互重叠，不能直接相加宣称单次总时长。两个Llama的原SparseGPT状态/标签来源另行保留；当前重用前均通过数值门槛。','',
        '## Qwen016实际运行时成本','',
        '以下P+D三轮中位数。初始化含引擎准备，arena捕获为其中子项，不能相加重复计费；NVML是两场景同进程的采样峰值。'
        'host cache与GPU arena均保留，恢复计入cycle；E2E延续原计时边界。','',
        '| Policy | Init s | Arena capture ms | NVML peak GiB | Host weights GiB | GPU arena GiB | Restore ms | Cycle ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    raw=list(csv.DictReader((ROOT/'qwen016_runtime_overhead.csv').open()))
    for pid in ['uniform_dense_bf16','uniform_int8','uniform_sq_int8',
                'hardware_minimum_prefill_decode_phase_pair','daso8_exact_prefill_decode_q0.04']:
        selected=[r for r in raw if r['policy']==pid and r['scenario']=='prefill_decode']
        median=lambda key:statistics.median(float(r[key]) for r in selected)
        lines.append(f'| {pid} | {median("initialization_ms")/1000:.2f} | {median("capture_model_ms"):.2f} | '
            f'{median("sampled_nvml_max_used_bytes")/1024**3:.2f} | {median("host_weight_bytes")/1024**3:.2f} | '
            f'{median("phase_reused_arena_bytes")/1024**3:.2f} | {median("restore_ms"):.2f} | {median("cycle_ms"):.2f} |')
    lines += ['', 'DASO-8高预算点的E2E收益大于包含恢复的cycle收益，host/arena占用也高于共同硬件端点；'
        '不将其解释为免费阶段切换或所有负载均有同等收益。','',
        '[叶子命令/哈希](preparation_processes.csv) · [按阶段汇总](preparation_summary.csv) · '
        '[逐轮运行时原始汇总](qwen016_runtime_overhead.csv) · [旧准备成本](../runs/004/results/preparation_costs.json)','']
    (ROOT/'COST_REPORT.md').write_text('\n'.join(lines))
    print('Cost snapshot',len(rows),'leaf commands',len(summary),'groups',flush=True)


if __name__=='__main__':main()
