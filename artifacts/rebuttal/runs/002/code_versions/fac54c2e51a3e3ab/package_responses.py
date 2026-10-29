import hashlib
import json
import shutil
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]


def main():
    root=RUN.parents[1]/'responses'
    manifest=hashlib.sha256((RUN/'results/manifest.json').read_bytes()).hexdigest()
    for reviewer,title in [('reviewer2','Exi5：SmoothQuant W8A8 / FP8 baseline 与扩展候选'),
                           ('reviewer5','gkP9：加入 INT8 / FP8 后的候选空间与实测前沿')]:
        dest=root/reviewer/'qwen3_run002';dest.mkdir(parents=True,exist_ok=True)
        copies={}
        for relative in ['REPORT.md','results/summary_final.csv','results/final_audit.json',
            'results/candidate_usage.json','results/smoothing_metadata.json','results/kernel_audit.json',
            'results/action_profile_final.json','configs/final_protocol.json','reference/sources.json']:
            source=RUN/relative;target=dest/source.name;shutil.copy2(source,target)
            copies[target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
        (dest/'figures').mkdir(exist_ok=True)
        for source in (RUN/'results/figures').iterdir():
            if source.is_file():
                target=dest/'figures'/source.name;shutil.copy2(source,target)
                copies['figures/'+source.name]=hashlib.sha256(target.read_bytes()).hexdigest()
        for name in ['code','data','configs','reference']:
            target=dest/name
            if not target.exists():target.symlink_to(Path('../../../runs/002')/name)
        (dest/'README.md').write_text(f'# {title}\n\n'
            '结果来自完整性校验通过的 runs/002；旧 runs/001 证据保持不变。\n\n'
            '- 三种新增 baseline：SmoothQuant INT8、原生 FP8、SmoothQuant-FP8 对照，均使用真实低精度 GEMM。\n'
            '- 原五候选 + SmoothQuant INT8 / FP8 的 DASO，并有同质量拟合、同实测延迟表的五候选控制。\n'
            '- 速度为同一物理 GPU3 的整模型测试；质量按独立测试 NLL 与完整任务报告。\n'
            '- 平滑在候选内部融合进动态激活量化；保留 BF16 注意力与 KV。并非声称复现原论文的完整整型 attention pipeline。\n'
            '- 与 001 HAWQ 的跨轮比较不能隔离分配器差异：INT8 内核实现、候选与延迟输入已变化。\n\n'
            '[完整报告](../../../runs/002/REPORT.md)，[完整 CSV](summary_final.csv)，[质量—速度图](figures/quality_latency_zoom.png)。\n'
            '本目录 REPORT.md 为报告副本；其 results/... 内部链接请使用上方原始完整报告。\n')
        (dest/'source.json').write_text(json.dumps({'run':'002','manifest_sha256':manifest,
            'copies':copies,'links':{name:str(Path('../../../runs/002')/name) for name in ['code','data','configs','reference']}},indent=2)+'\n')
    index=root/'README.md';content=index.read_text()
    marker='## 新增 8bit baseline 与扩展候选（run002）'
    if marker not in content:
        index.write_text(content+'\n'+marker+'\n\n'
            '- [reviewer2 / Exi5](reviewer2/qwen3_run002/README.md)：SmoothQuant W8A8、FP8、扩展 DASO。\n'
            '- [reviewer5 / gkP9](reviewer5/qwen3_run002/README.md)：INT8 / FP8 候选空间与同成本输入控制。\n'
            '- [run002 完整报告](../runs/002/REPORT.md)；001 原证据包保持有效。\n')


if __name__=='__main__':main()
