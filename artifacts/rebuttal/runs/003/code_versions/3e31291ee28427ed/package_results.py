"""Seal run003 and add reviewer evidence without editing previous runs."""
import hashlib
import json
import shutil

from common import RUN, write_json


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    audit=json.loads((RUN/'results/final_audit.json').read_text());assert audit['passed']
    entries=[]
    for path in sorted(RUN.rglob('*')):
        if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts:continue
        relative=str(path.relative_to(RUN))
        if relative in ['results/manifest.json','results/progress.json','results/summary_current.csv']:continue
        if relative.startswith('extensions/'):continue
        entries.append({'path':relative,'bytes':path.stat().st_size,'sha256':digest(path)})
    write_json(RUN/'results/manifest.json',{'files':entries})
    manifest=digest(RUN/'results/manifest.json');root=RUN.parents[1]/'responses'
    for reviewer in ['reviewer1','reviewer2','reviewer5']:
        dest=root/reviewer/'qwen3_run003';dest.mkdir(parents=True,exist_ok=True);copies={}
        for relative in ['REPORT.md','results/summary_final.csv','results/final_audit.json',
            'results/quality_matched_frontiers.json','results/budget_audit.json','results/candidate_usage.json',
            'configs/design.json','configs/execution.json','results/profile_complete.json','results/int8_fusion_audit.json',
            'results/allocation_replay_audit.json']:
            source=RUN/relative;target=dest/source.name;shutil.copy2(source,target);copies[target.name]=digest(target)
        (dest/'figures').mkdir(exist_ok=True)
        for source in (RUN/'results/figures').iterdir():
            if source.is_file():
                target=dest/'figures'/source.name;shutil.copy2(source,target);copies['figures/'+source.name]=digest(target)
        for name in ['code','data','configs','reference','results']:
            link=dest/name
            if not link.exists():link.symlink_to('../../../runs/003/'+name)
        (dest/'README.md').write_text('# Qwen3 工作负载覆盖补充证据\n\n'
            '9组P+D与7组prefill-only，覆盖B/S/O变化；包含真实8bit、phase-uniform、固定策略迁移与按负载重分配的DASO/HAWQ。\n\n'
            '[完整报告](REPORT.md) · [完整数表](summary_final.csv) · [配置](design.json) · [审计](final_audit.json)。\n\n'
            '按实际NLL与绝对延迟比较，所有预先指定场景保留。冻结质量模型跨长度迁移；同步同质batch，不主张已验证mixed continuous batching或严格质量保证。\n')
        write_json(dest/'source.json',{'run':'003','manifest_sha256':manifest,'copies':copies,
            'links':{name:'../../../runs/003/'+name for name in ['code','data','configs','reference','results']}})
    index=root/'README.md';text=index.read_text();marker='## 工作负载覆盖补充（run003）'
    if marker not in text:
        index.write_text(text+'\n'+marker+'\n\n'
            '- [reviewer2](reviewer2/qwen3_run003/README.md)：batch、提示长度、生成长度与按负载重分配。\n'
            '- [reviewer1](reviewer1/qwen3_run003/README.md)：跨负载实际质量与预算满足情况。\n'
            '- [reviewer5](reviewer5/qwen3_run003/README.md)：固定迁移、成本变化和真实端到端排序。\n')
    print(json.dumps({'manifest_files':len(entries),'manifest_sha256':manifest,'reviewer_packages':3}))


if __name__=='__main__':main()
