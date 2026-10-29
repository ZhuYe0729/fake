"""Place final evidence by reviewer, reusing the single frozen code/data source."""
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]
RESPONSES=RUN.parents[1]/'responses'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert json.loads((RUN/'results/progress.json').read_text())['complete']
    with (RUN/'results/summary_final.csv').open() as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
    reviewers=[('reviewer1','AK6a','质量预算与双表示内存',
        'Prefill 10/10 点超出名义预算；decode 7/8 点满足本次实测预算。包含预测/实际 ΔNLL、违约量、CPU/GPU 内存和校准误差分解。'),
        ('reviewer2','Exi5','HAWQ-v3 与 Qwen3 基线',
        '包含七种 uniform、HAWQ-v3 分配器的 INT4 主对照及 FP4 控制、DASO 和阶段统一 hybrid。HAWQ 是公开公式/ILP 的 PTQ 移植，INT8 行不包含 SmoothQuant smoothing。'),
        ('reviewer3','LrT2','Qwen3 泛化、混合精度和建模开销',
        '提供 Qwen3-8B 的完整对比、40 项 kernel 延迟预测误差及准备/搜索成本。速度和内存统一物理 GPU 3 测量。'),
        ('reviewer4','QboE','架构补充与搜索成本',
        '提供 Qwen3-8B 以及校准、Hessian、拟合、DP 的实测成本。此证据仍属 8B 规模，不能作为已测试 30B 的证据。'),
        ('reviewer5','gkP9','局部误差输入与质量模型实证',
        'code/collect_costs.py 在未压缩模型的缓存输入上逐方法测量局部误差，data/local_errors.csv 保留逐模块/阶段数据。包含随机留出策略误差及搜索后策略实际误差。')]
    index=['# Qwen3-8B 最终证据索引','',
        'reviewer 编号按 reference/reviewers.md 中五位人工 reviewer 的出现顺序映射。所有文件来自 runs/001 的完整性校验后结果。','',
        '| 目录 | 原审稿编号 | 证据主题 |','| --- | --- | --- |']
    for directory,identity,title,note in reviewers:
        root=RESPONSES/directory/'qwen3_run001';root.mkdir(parents=True,exist_ok=True)
        selected=[row for row in rows if directory!='reviewer1' or row['family']=='DASO' or row['policy_id']=='uniform_dense_bf16']
        with (root/'results.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(selected)
        for name in ['code','data','configs','reference']:
            link=root/name
            if not link.exists():link.symlink_to(os.path.relpath(RUN/name,root),target_is_directory=True)
        figures=root/'figures';figures.mkdir(exist_ok=True)
        bases=['quality_calibration_and_budgets'] if directory in ['reviewer1','reviewer5'] else ['measured_quality_latency','measured_quality_latency_zoom']
        for base in bases:
            for suffix in ['png','pdf','svg']:shutil.copy2(RUN/f'results/figures/{base}.{suffix}',figures/f'{base}.{suffix}')
        for name in ['preparation_costs.json','latency_prediction_accuracy.json','selected_policy_quality_audit.json']:
            shutil.copy2(RUN/'results'/name,root/name)
        report=os.path.relpath(RUN/'REPORT.md',root)
        (root/'README.md').write_text(f'# {identity}：{title}\n\n{note}\n\n'
            f'- [完整报告]({report})\n- [最终结果表](results.csv)\n- [策略与协议](configs/)\n'
            '- [实验代码](code/)\n- [冻结输入及逐模块误差](data/)\n- [来源代码与论文参考](reference/)\n\n'
            '代码、输入、配置和参考文件链接到同一份 run001 冻结来源；结果 CSV 和图表是最终快照。'
            '全量下游任务只对应预先选定代表点，其余扫描点评测真实速度和独立测试 NLL。\n')
        files=[path for path in root.rglob('*') if path.is_file() and not path.is_symlink() and not any(part in ['code','data','configs','reference'] for part in path.relative_to(root).parts)]
        provenance={'reviewer':identity,'run':'001','source_manifest_sha256':digest(RUN/'results/manifest.json'),
            'source_summary_sha256':digest(RUN/'results/summary_final.csv'),
            'files':{str(path.relative_to(root)):digest(path) for path in files if path.name!='source.json'}}
        (root/'source.json').write_text(json.dumps(provenance,indent=2)+'\n')
        index.append(f'| [{directory}/{identity}]({directory}/qwen3_run001/README.md) | {identity} | {title} |')
    index+=['','[统一完整报告](../runs/001/REPORT.md)；失败尝试与原始日志保留在 runs/001，不混入这些最终 CSV。','']
    (RESPONSES/'README.md').write_text('\n'.join(index))


if __name__=='__main__':main()
