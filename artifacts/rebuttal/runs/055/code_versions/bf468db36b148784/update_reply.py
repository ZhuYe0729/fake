"""Insert audited full32B evidence into the existing Chinese rebuttal draft."""
import json
import time

from common import REPO, RUN


def main():
    assert json.loads((RUN / 'results/final_audit_complete.json').read_text())['complete']
    report = REPO / 'artifacts/rebuttal/responses/reviewer4/pro6000_qwen32b_run055'
    rows = json.loads((report / 'results.json').read_text())['points']
    reply = REPO / 'artifacts/rebuttal/REBUTTAL_RESPONSE_CN.md'
    original = reply.read_text()
    before = '对于30B级别，更大模型的完整质量与 E2E 评测尚未完成。'
    if before not in original:
        assert 'responses/reviewer4/pro6000_qwen32b_run055/README.md' in original
        return
    start = original.index(before)
    end = original.index('\n### 3．压缩策略搜索耗时', start)
    lines = ['对于30B级别，我们补充了完整 Qwen3-32B（64层、32.76B参数）在 RTX PRO 6000 Blackwell Server Edition 上的实验，包含 BF16、DASO 与 HAWQ-V3 的 PTQ 适配。新模型重新采集质量校准，新硬件重新测量延迟；与已有32B合成shape测试分开报告。', '',
        'DASO在六档搜索预算和四组阶段组合上探索，在独立validation上采用配对mean+2SE+0.005质量余量，并在预先规定的shortlist中按探索计时E2E选出 ε=0.02/0.07 的代表点；随后冻结策略测test及独立正式三轮。选择池允许uniform锚点，实际模块组成保留。HAWQ匹配相应DASO点的Linear代理延迟上限，表中预算标签不表示HAWQ受同一真实NLL约束。', '',
        '**表 R4-2．完整 Qwen3-32B，B8 / S2048 / O64（Prefill输出1）**', '',
        '| 方法 | Prefill ΔNLL | Prefill Avg ↑ | Prefill TTFT× | P+D ΔNLL | P+D Avg ↑ | P+D TPOT× | E2E× | cycle× |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for family, budget in [('BF16', None), ('DASO', .02), ('DASO', .07), ('HAWQ-V3 adaptation', .02), ('HAWQ-V3 adaptation', .07)]:
        pp = next(row for row in rows if row['family'] == family and row['budget'] == budget and row['scenario'] == 'prefill_only')
        pd = next(row for row in rows if row['family'] == family and row['budget'] == budget and row['scenario'] == 'prefill_decode')
        name = 'BF16' if budget is None else (f'DASO（ε={budget:g}）' if family == 'DASO' else f'HAWQ-V3（匹配ε={budget:g}点）')
        flag = '†' if pp['variation_flag'] else ''
        flag_pd = '†' if pd['variation_flag'] else ''
        lines.append(f"| {name} | {pp['delta_nll']:.6f} | {pp['task_average']:.2f} | {pp['ttft_ms_speedup']:.3f}{flag} | {pd['delta_nll']:.6f} | {pd['task_average']:.2f} | {pd['tpot_ms_speedup']:.3f} | {pd['e2e_ms_speedup']:.3f}{flag_pd} | {pd['cycle_ms_speedup']:.3f} |")
    daso = [row for row in rows if row['family'] == 'DASO']
    feasible = sum(row['delta_nll'] <= row['budget'] for row in daso)
    lines += ['', f'四个DASO预算—场景设定中，{feasible}/4满足实测平均test ΔNLL预算；共享策略映射和固定文本，不解释为四次独立试验或严格质量保证。每点100个WikiText test窗口，ΔNLL单位为nats/token；Prefill/P+D Avg与前述主表定义一致，分项与WikiText word perplexity保留。', '',
        '正式速度固定同卡GPU0、CPU0–7与NUMA0，8GiB BF16 KV，三轮、每点5次warmup加10次计时；按本轮BF16首尾参考归一化后对三轮比值取均值。†保留超过5%的轮间波动或BF16漂移。E2E包含本批切换，cycle进一步包含恢复。完整下游任务按原协议执行，生成上限256且正常EOS，与固定64-token测速分开。', '',
        'HAWQ使用本模型P16/B4/S256 signed Hessian迹及FP4/INT8/BF16后端，是LLM PTQ适配；DASO有八种候选，因此本表不作为仅分配器变化的因果消融。本轮仍为单卡固定批负载，未据此声称完成多卡部署或持续错峰生产服务验证。', '',
        '来源：[完整32B结果、全部任务分项、内存、准备/求解成本与逐文件证据](responses/reviewer4/pro6000_qwen32b_run055/README.md)。既有[合成shape实测](runs/004/REPORT.md)仅保留为后端可执行性证据。', '']
    updated = original[:start] + '\n'.join(lines) + original[end:]
    updated = updated.replace('更新：2026-10-29。',
        '更新：' + time.strftime('%Y-%m-%d', time.gmtime(time.time() + 8 * 3600)) + '（北京时间）。', 1)
    updated = updated.replace('长 Decode 专门 NLL、持续错峰到达、双 worker PD 和完整30/32B模型均未作为已完成证据。',
        '长 Decode 专门 NLL、持续错峰到达和双 worker PD 仍未作为已完成证据；完整32B新增结果见R4第2点。')
    reply.write_text(updated)
    with (RUN / 'README.md').open('a') as stream:
        stream.write('\n正式三方法完整结果与终审已闭合，参见[中文结果表及全部分项](../../responses/reviewer4/pro6000_qwen32b_run055/README.md)与[终审核验](results/final_audit_complete.json)。前面的准备阶段文字属于执行历史，实际完成范围以这两个入口为准。\n')
    with (REPO / 'dev/impls/184_pro6000_qwen32b_rebuttal_impl.md').open('a') as stream:
        stream.write('\n## ' + time.strftime('%Y-%m-%d', time.gmtime()) + ' - 三方法完整对照闭合\n- BF16、DASO两预算及HAWQ匹配点完成冻结后的100窗test、三轮独占速度/内存和完整原协议下游；相同策略共享证据，不计为独立重复。\n- final_report独立核验结果SHA、逐窗/逐token记录、计时中位数、任务计数与分片覆盖，生成R4中文结果、CSV和来源清单；同步中文回复R4规模段与内部核对，完整保留负面点及波动。\n')


if __name__ == '__main__':
    main()
