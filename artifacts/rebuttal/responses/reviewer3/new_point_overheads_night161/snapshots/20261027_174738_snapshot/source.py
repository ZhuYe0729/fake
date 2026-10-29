"""Audited per-point solving, process wall time and actual phase-cycle costs."""
import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT.parent/'runs'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def save(path,rows):
    with path.open('w')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    rids=['032','033','034','035','036'];sources={};solvers=[];online=[];rounds=[];leaves=[];audited=[]
    for rid in rids:
        run=RUNS/rid;point_file=run/('configs/points.json'if rid in ['032','033']else'configs/sweep_points.json')
        sources[str(point_file)]=sha(point_file)
        for p in read(point_file):
            first=p['latency_solver'];second=p['quality_tie_solver'];assert first['optimal']and second['optimal']
            solvers.append({'run':rid,'label':p['label'],'policy':p['policy'],'q':p['budget'],'prefill_weight':p['prefill_weight'],
                'modules':len(first['indices']),'joint_actions_per_module':64,'latency_solve_seconds':first['seconds'],
                'quality_tie_solve_seconds':second['seconds'],'total_solve_seconds':first['seconds']+second['seconds'],
                'predicted_linear_ms':p['linear_ms']})
        audit=run/'results'/('final_audit.json'if rid in ['032','033']else'sweep_final_audit.json')
        if not audit.exists():continue
        assert read(audit).get('verified',True);sources[str(audit)]=sha(audit);audited.append(rid)
        prefix='formal'if rid in ['032','033']else'sweep_formal'
        orders=read(run/f'configs/{prefix}_rounds.json')['rounds']
        policies=[p for p in orders[0]if not p.endswith('_bracket_end')];records={}
        for ri,order in enumerate(orders):
            for pid in order:
                path=run/f'results/measurements/{prefix}_round{ri}_{pid}.json';record=read(path)
                sources[str(path)]=sha(path);records[ri,pid]=record
                case=next(c for c in record['cases']if c['output']==64)
                assert sum(not s['warmup']for s in case['samples'])==5
                assert all(v['exact_equal']for v in record['exact_weight_audit'].values())
        for pid in policies:
            local=[]
            for ri in range(3):
                record=records[ri,pid];value=next(c for c in record['cases']if c['output']==64)['medians']
                reference=next(c for c in records[ri,'uniform_int8']['cases']if c['output']==64)['medians']
                memory=record['memory'];payload=record['transfer_bytes_per_generation']
                row={'run':rid,'policy':pid,'round':ri,'e2e_ms':value['e2e_ms'],'restore_ms':value['restore_ms'],
                    'cycle_ms':value['cycle_ms'],'e2e_vs_int8':reference['e2e_ms']/value['e2e_ms'],
                    'cycle_vs_int8':reference['cycle_ms']/value['cycle_ms'],
                    'restore_fraction_of_cycle':value['restore_ms']/value['cycle_ms'],
                    'prefill_transfer_bytes':payload['prefill'],'decode_transfer_bytes':payload['decode'],
                    'host_phase_cache_gib':memory['host_weight_bytes']/1024**3,
                    'phase_reused_arena_gib':memory['phase_reused_arena_bytes']/1024**3,
                    'torch_peak_allocated_gib':memory['torch_peak_allocated_bytes']/1024**3,
                    'initialization_seconds':record['initialization_ms']/1000,
                    'arena_capture_ms':record['capture_model_ms']}
                local.append(row);rounds.append(row)
            keys=[k for k in local[0]if k not in ['run','policy','round']]
            online.append({'run':rid,'policy':pid,**{k:statistics.mean(v[k]for v in local)for k in keys},
                'e2e_vs_int8_min':min(v['e2e_vs_int8']for v in local),'e2e_vs_int8_max':max(v['e2e_vs_int8']for v in local),
                'cycle_vs_int8_min':min(v['cycle_vs_int8']for v in local),'cycle_vs_int8_max':max(v['cycle_vs_int8']for v in local)})
        migration=read(run/'configs/sweep_formal_device.json').get('frozen_at_unix')if rid=='035'else None
        for command in(run/'logs').glob('*.command.json'):
            completion=command.with_name(command.name.replace('.command.json','.completion.json'))
            if not completion.exists():continue
            argv=read(command)['command'];script=next((Path(str(a)).name for a in argv if str(a).endswith('.py')),None)
            if script in ['driver.py','sweep_driver.py','job_queue.py','ops.py']or script is None:continue
            result=read(completion)
            stage={'export_model.py':'export','benchmark.py':'formal_speed','generate_tasks.py':'full_generation',
                'run_workflow.py':'full_mc','score.py':'benchmark_nll','score_nll.py':'benchmark_nll'}.get(script,script)
            if stage=='formal_speed'and migration and command.stat().st_mtime<migration:stage='formal_speed_preserved_before_migration'
            leaves.append({'run':rid,'stage':stage,'script':script,'seconds':result['seconds'],'returncode':result['returncode'],
                'command_metadata':str(command),'command_sha256':sha(command),'completion_sha256':sha(completion)})
    if args.final:assert len(audited)==5,'Wait for036 complete before final snapshot'
    preparation=[]
    for rid in ['017','018']:
        run=RUNS/rid
        for stage,name in [('new_8bit_features','features_complete.json'),('linear_profile_80_rows','profile_complete.json'),
            ('quality_fit_prefill','quality_fit_prefill_only.json'),('quality_fit_pd','quality_fit_prefill_decode.json')]:
            path=run/'results'/name;value=read(path);sources[str(path)]=sha(path)
            preparation.append({'run':rid,'stage':stage,'internal_seconds':value['seconds'],'source':str(path)})
    totals=defaultdict(lambda:{'success_jobs':0,'success_process_sum_seconds':0.,'failed_jobs':0,'failed_process_sum_seconds':0.})
    for row in leaves:
        item=totals[row['run'],row['stage']];key='success'if row['returncode']==0 else'failed'
        item[key+'_jobs']+=1;item[key+'_process_sum_seconds']+=row['seconds']
    summary=[{'run':rid,'stage':stage,**values}for(rid,stage),values in sorted(totals.items())]
    parent=ROOT.parent/'responses/reviewer3/new_point_overheads_night161';parent.mkdir(parents=True,exist_ok=True)
    label=time.strftime('%Y%m%d_%H%M%S',time.gmtime())+('_final'if args.final else'_snapshot')
    dest=parent/'snapshots'/label;dest.mkdir(parents=True)
    for name,values in [('solver_points.csv',solvers),('runtime_points.csv',online),('runtime_rounds.csv',rounds),
        ('preparation_reference.csv',preparation),('completed_leaf_processes.csv',leaves),('completed_process_summary.csv',summary)]:save(dest/name,values)
    lines=['# 新点求解、准备与恢复成本（R3 LrT2 / R5 gkP9）','',
        f'本快照已审计速度campaign：{", ".join(audited)}。'+('五campaign完整。'if args.final else'这是中间快照；未完成campaign不参与实测在线排名，CPU求解记录已固定。'), '',
        '## 固定代理精确求解','',
        '所有点沿用本模型既有校准系数与实测LUT，不在每次q/wP变化时重新训练。以下仅HiGHS求解器内部墙钟，两次求解依次最小化延迟、同速最小预测质量损失；不是整个准备/量化/validation成本。','',
        '| Run | 点 | 模块×动作 | 延迟求解s | 同速质量求解s | 合计s |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for r in solvers:lines.append(f'| {r["run"]} | {r["label"]} | {r["modules"]}×64 | {r["latency_solve_seconds"]:.3f} | {r["quality_tie_solve_seconds"]:.3f} | {r["total_solve_seconds"]:.3f} |')
    lines+=['','## 复用的原准备成本','',
        'Llama的目标模型新8bit误差/画像/拟合函数内部时间如下；画像80行含八候选外控制，不能只算求解几秒就宣称从零免费。完整标签、Hessian、导出、validation和旧缓存成本见原成本报告，函数时间与其叶子进程壁钟不同口径。','',
        '| Run | 阶段 | 函数内部s |','| --- | --- | ---: |']
    for r in preparation:lines.append(f'| {r["run"]} | {r["stage"]} | {r["internal_seconds"]:.2f} |')
    lines+=['','## 单批E2E与包含下一批恢复的cycle','',
        '同轮INT8归一化后取三轮算术均值，与源新点主表相同聚合。E2E已经包含prefill→decode；restore是在下一次prefill前显式恢复并等待的实测墙钟，cycle包含它。没有假定异步隐藏或连续服务收益。初始化包含arena捕获，二者不能相加。host/arena是阶段权重子集，Torch峰值含KV/工作区，不当作三份独立总内存相加。','',
        '| Run/配置 | E2E ×INT8 | Cycle ×INT8 | 恢复ms | 恢复占cycle | Host阶段缓存GiB | 复用arena GiB | 初始化s |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in online:lines.append(f'| {r["run"]}/{r["policy"]} | {r["e2e_vs_int8"]:.3f} | {r["cycle_vs_int8"]:.3f} | {r["restore_ms"]:.2f} | {100*r["restore_fraction_of_cycle"]:.2f}% | {r["host_phase_cache_gib"]:.2f} | {r["phase_reused_arena_gib"]:.2f} | {r["initialization_seconds"]:.2f} |')
    lines+=['','## 完成叶子进程的实验成本','',
        '成功/失败进程壁钟求和包括CPU、模型载入、量化与GPU执行；不是GPU-active、并行后的总墙钟或每请求成本。只计已完成叶子，driver/queue等待不计；035迁移前保留的重复测速明确分列。统计这些复现实验的全部重复导出，不隐去工程重跑。','',
        '| Run | 阶段 | 成功进程 | 成功壁钟和s | 失败进程 | 失败壁钟和s |','| --- | --- | ---: | ---: | ---: | ---: |']
    for r in summary:lines.append(f'| {r["run"]} | {r["stage"]} | {r["success_jobs"]} | {r["success_process_sum_seconds"]:.2f} | {r["failed_jobs"]} | {r["failed_process_sum_seconds"]:.2f} |')
    lines+=['','[全部求解](solver_points.csv) · [阶段字节/完整在线配置](runtime_points.csv) · [三轮范围来源](runtime_rounds.csv) · [逐条叶子命令](completed_leaf_processes.csv) · [逐阶段汇总](completed_process_summary.csv) · [来源](provenance.json)','']
    (dest/'README.md').write_text('\n'.join(lines))
    metadata={'complete_snapshot':True,'final_five_campaigns':args.final,'audited_runtime_campaigns':audited,
        'solver_labels':len(solvers),'runtime_maps':len(online),'leaf_processes':len(leaves),'source_sha256':sources,
        'code_sha256':sha(Path(__file__)),'scope':'Measured incremental overheads; no new GPU test, no free preparation/async hiding/GPU-active claim.'}
    metadata['delivery_sha256']={p.name:sha(p)for p in dest.iterdir()if p.is_file()}
    (dest/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (parent/'README.md').write_text('# 新点准备与阶段恢复成本\n\n最新'+('完整'if args.final else'中间')+f'快照：[报告](snapshots/{label}/README.md)。旧快照全部保留。\n\n[原完整准备成本](../../../../night151/COST_REPORT.md)\n')
    print('Published overhead snapshot',label,len(solvers),'solver points',len(online),'runtime maps',flush=True)


if __name__=='__main__':main()
