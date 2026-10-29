# Llama-3.1-8B-Instruct剪枝对照（Reviewer QboE / AK6a）

BF16和原SparseGPT来自封存run017；Wanda与magnitude在run020完整补测。各自使用相同实际SparseBF16格式和kernel、相同B8/S2048/O1或64、6GiB BF16 KV、GPU0/NUMA0以及三轮3warmup+8timed。速度按各自三轮BF16首尾锚点归一化，跨campaign来源明确保留。

Wanda使用层序传播；输入按原SparseGPT记录的128×2048 train样本重建。原加载规则、token总数和seed0起点一致，原tensor未保留，不能声称与不可用原tensor做过字节哈希比较。完整CNN/DM1000、DialogSum1500、IWSLT333均重新生成，不继承其他剪枝算法的质量。

WikiText列为完整lm-eval word_perplexity（batch1），不是64窗口token平均NLL取exp；其余四MC统一batch8。

| Method | PP ΔNLL | ARC-C | ARC-E | Wino | MMLU | WikiText word PPL | Prefill × | PD ΔNLL | CNN/DM | DialogSum | IWSLT | PD TTFT × | TPOT × | E2E × |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.000 | 55.717 | 82.239 | 73.875 | 68.445 | 9.425 | 1.000 | 0.000 | 19.337 | 13.482 | 10.697 | 1.000 | 1.000 | 1.000 |
| SparseGPT 2:4 | 0.639 | 38.481 | 68.266 | 66.219 | 42.921 | 21.436 | 1.520 | 0.660 | 6.685 | 5.104 | 3.038 | 1.532 | 0.866 | 1.149 |
| Wanda 2:4 | 1.175 | 35.154 | 63.636 | 61.957 | 37.018 | 38.268 | 1.525 | 1.135 | 8.677 | 3.125 | 4.173 | 1.534 | 0.841 | 1.128 |
| Magnitude 2:4 | 3.547 | 28.498 | 45.960 | 52.723 | 28.564 | 516.328 | 1.528 | 3.778 | 3.410 | 1.297 | 0.089 | 1.536 | 0.840 | 1.130 |

## 新剪枝准备时间

| Algorithm | Load s | Pruning/calibration s | Total including save s |
| --- | ---: | ---: | ---: |
| wanda | 16.39 | 98.69 | 134.79 |
| magnitude | 6.16 | 5.26 | 31.61 |

原SparseGPT是缓存复用，其历史准备成本另报，不能作为零成本。三轮最小/最大范围不是置信区间；这里比较剪枝方法，不把共享SparseBF16 kernel的相近速度归因于剪枝算法。

[完整CSV](pruning_table.csv) · [逐项来源](provenance.json) · [三轮范围](speed_ranges.csv) · [冻结设计](design.json)

[原始代码和结果](../../../runs/020/README.md)
