"""Refresh only measured main-table cells, preserving planned supplement tables."""
import csv
import datetime
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
COLS=['method','arc_c','arc_e','wino','mmlu','prefill_ttft_speedup','cnn_dm','dialogsum','iwslt',
      'pd_ttft_speedup','pd_tpot_speedup','pd_e2e_speedup']


def table(path,status):
    rows=list(csv.DictReader(path.open()))
    lines=['| 方法 | ARC-C | ARC-E | Wino. | MMLU | Prefill TTFT × | CNN/DM | DialogSum | IWSLT | P+D TTFT × | TPOT × | E2E × | 状态/优先级 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        values=[row[c] if c=='method' else f'{float(row[c]):.2f}' if row[c] else '待' for c in COLS]
        lines.append('| '+' | '.join([*values,status])+' |')
    return '\n'.join(lines)


def main():
    path=BASE/'EXPERIMENT_CHECKLIST.md'
    text=path.read_text()
    status=[]
    for number,rid in [(1,'018'),(2,'017')]:
        source=BASE/f'runs/{rid}/results/main_table.csv'
        if not source.exists():continue
        complete=(BASE/f'runs/{rid}/results/final_audit.json').exists()
        note='核心完成' if complete else '三轮速度完成；完整任务运行中'
        block=re.search(rf'(## 表{number} — [^\n]+\n)(.*?)(?=\n## 表)',text,re.S)
        introduction=(f'本表取自run{rid}统一新路径的15行核心主表，旧论文数值保留在归档，不混入以下新评测。'
            f'当前{note}；原生INT4与SQ-FP8等P2控制尚未开展。\n\n'
            f'[完整来源、协议和逐项CSV](runs/{rid}/REPORT.md)。\n\n')
        text=text[:block.start()]+block[1]+'\n'+introduction+table(source,note)+'\n'+text[block.end():]
        status.append(f'Llama run{rid}：{note}')
    source=BASE/'responses/reviewer2/qwen3_table2_run016/main_table.csv'
    block=re.search(r'(## 表3 — [^\n]+\n)(.*?)(?=\n## 表)',text,re.S)
    intro=('以下19行是015原17行加016新DASO-8的跨campaign完整汇总，各campaign使用自身BF16归一化。'
        '不是19行同轮重测；旧质量来源与新完整任务见逐项provenance。\n\n'
        '[完整19行主表与来源](responses/reviewer2/qwen3_table2_run016/README.md)。\n\n')
    text=text[:block.start()]+block[1]+'\n'+intro+table(source,'已完成；015/016跨campaign')+'\n'+text[block.end():]
    supplements=[]
    for rid,title in [('019','六负载'),('020','剪枝'),('021','五预算'),('022','预算测速'),('025','接口'),('026','validation样本量')]:
        marker=BASE/f'runs/{rid}/results/final_audit.json'
        label='已完成' if marker.exists() and json.loads(marker.read_text()).get('verified') else '进行或排队中'
        supplements.append(rid+title+label)
    snapshot=('当前快照：Qwen015/016已完成并封存，完整19行汇总已交付。'+ '；'.join(status)+
        '。'+'；'.join(supplements)+'；'
        '详细状态见[NIGHT151报告](NIGHT151_REPORT.md)及[监督快照](night151/status.json)。')
    text=re.sub(r'当前快照：[^\n]+',lambda _:snapshot,text,count=1)
    text=re.sub(r'更新：[^。]+。',lambda _:'更新：'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')+'。',text,count=1)
    path.write_text(text)
    print('Updated main tables from measured CSV; supplement plans preserved.')


if __name__=='__main__':main()
