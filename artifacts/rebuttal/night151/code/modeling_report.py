"""Publish scoped, unfitted diagnostics for the three measured model families."""
import csv
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    lines=['# 三模型建模与质量代理诊断（R3 LrT2 / R5 gkP9）','',
        '延迟预测是冻结的实测Linear LUT相加，加上在分配前固定的BF16 E2E残差；'
        '不是原Hybrid的未知shape预测验证。Qwen包含全部43个冻结场景；Llama只对预先固定的核心primary计速度误差，'
        '其他点仍保留为待测。质量代理则包含各模型完整冻结池，不用confirmation重新拟合或排除差点。','',
        '| Model | Scene | Timed policies | E2E MAPE % | Median APE % | p90 APE % | Max APE % | E2E MAE ms | Quality policies | Quality MAE nats/token |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for rid,model in [('016','Qwen3-8B'),('017','Llama-3.1-8B'),('018','Llama-2-7B')]:
        prefix='qwen016' if rid=='016' else f'llama{rid}'
        rows=list(csv.DictReader((ROOT/f'{prefix}_modeling_points.csv').open()))
        for scene in ['prefill_only','prefill_decode']:
            local=[r for r in rows if r['scenario']==scene]
            timed=[r for r in local if r['measured_e2e_ms']]
            ape=[float(r['absolute_percentage_error']) for r in timed]
            mae=statistics.mean(abs(float(r['predicted_e2e_ms'])-float(r['measured_e2e_ms'])) for r in timed)
            quality=statistics.mean(float(r['quality_absolute_error']) for r in local)
            numbers=[len(timed),statistics.mean(ape),statistics.median(ape),
                statistics.quantiles(ape,n=10,method='inclusive')[-1],max(ape),mae,len(local),quality]
            values=[str(v) if isinstance(v,int) else f'{v:.5f}' for v in numbers]
            lines.append('| '+' | '.join([model,scene,*values])+' |')
    lines += ['', '## 延迟与质量不能互相替代','',
        'LUT已计入各格式Linear内部的激活量化、scale与metadata处理；固定残差近似覆盖attention、KV与其余非Linear工作。'
        '恢复时间不在E2E中，独立保存restore和cycle；残差不能外推为任意B/S/O固定常数。'
        '本轮三模型都在同一GPU0/NUMA0测量，每个正式点三轮、每轮8个timed样本；预测没有读取这些结果调整offset。','',
        '质量误差为冻结八动作加和代理对独立confirmation的MAE，包含原代理不准的提案和较差稀疏配置。'
        'validation选择流程是新增的经验保护；低延迟预测误差不意味着低质量代理误差，也不保证生成指标。','',
        '原Hybrid广shape留出仍单列：[run004 broad validation](../runs/004/results/broad_latency_validation.csv)。'
        '不能用本表的已测shape LUT误差替换原Hybrid未知shape误差。','',
        '## 可导出的图与原始数据','',
        '- [三模型实测/预测 PDF](three_model_lut_prediction.pdf) · [SVG](three_model_lut_prediction.svg)',
        '- [同候选配对NLL PDF](llama_matched_paired_nll.pdf) · [SVG](llama_matched_paired_nll.svg)',
        '- [Qwen逐点](qwen016_modeling_points.csv) · [来源SHA](qwen016_modeling_scope.json)',
        '- [Llama-3.1逐点](llama017_modeling_points.csv) · [来源SHA](llama017_modeling_scope.json)',
        '- [Llama-2逐点](llama018_modeling_points.csv) · [来源SHA](llama018_modeling_scope.json)',
        '- [准备和评测成本](COST_REPORT.md) · [同候选/预算](LLAMA_DIAGNOSTICS.md)','']
    (ROOT/'MODELING_REPORT.md').write_text('\n'.join(lines))
    print('Published six scenario diagnostics with explicit coverage.')


if __name__=='__main__':main()
