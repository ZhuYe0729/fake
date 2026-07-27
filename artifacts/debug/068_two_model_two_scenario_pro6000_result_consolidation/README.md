# RTX PRO 6000 两模型 × 两场景结果汇总

本目录汇总 064--067 的正式质量与 prefill-only 结果，并在 RTX PRO 6000 上按 RTX 5090
完全相同的隔离进程协议补测 decode TTFT/TPOT/E2E。没有修改历史实验目录。四个源 bundle
的 `validation/all.json` 均为 `ok=true`；原始 `complete_results.csv` 和源验证文件的逐字节
快照位于 `data/`，SHA-256 与来源及选点记录位于 `data/selection.json`。

## 论文表

- `result_v3.tex`：保留 `result_v2.tex` 的 RTX 5090 部分，并填入 RTX PRO 6000
  的两模型、两场景结果。
- `data/selected_results.csv`：表格中 Pro 6000 的 28 行机器可读数据。
- `validation/all.json`：汇总完整性与来源验证结果。
- `measurements/decode_components/summary.csv`：14 个 decode 策略的 TTFT、TPOT、E2E
  中位数与 BF16-relative speedup。
- `validation/decode_components.json`：168 份当前原始样本的协议、进程、GPU 与稳定性审计。

Prefill-only 的速度列是 B=8、input=2048、output=1 的实测 TTFT speedup；
prefill-decode 对每个策略分别以 O=1 和 O=64 测量：每个 phase 先 warmup 1 次，再测量
5 次，每次均启动全新的 Python/vLLM 进程并重新加载模型。TTFT 取 O=1 的五次中位数，
E2E 取 O=64 的五次中位数，`TPOT=(median E2E-median TTFT)/63`；配置为 B=8、
input=2048、BF16 KV、eager、prefix caching/chunked prefill 均关闭。所有 speedup 均相对
同模型、同 GPU、同协议的 BF16，没有使用 predictor 或跨 checkpoint 推算值。

## Ours 选点

Max-speed 只在表格所需下游指标齐全的 ours 点中取实测最快点。Balanced 使用同一套
模型无关规则：prefill-only 要求四项准确率相对 BF16 的平均保持率至少 95%、单项至少
90%；prefill-decode 要求三项生成指标平均保持率至少 87.5%、单项至少 75%，再取合格点
中实测最快者。

| Model | Scenario | Role | Point | Speedup | Mean retention | Min retention |
|---|---|---|---|---:|---:|---:|
| Llama-2-7B-Chat | Prefill-only | Max-speed | point_024 | 1.49× | 67.5% | 50.6% |
| Llama-2-7B-Chat | Prefill-only | Balanced | point_014 | 1.25× | 98.4% | 95.4% |
| Llama-2-7B-Chat | Prefill-decode | Max-speed | point_018 | 1.45× | 55.3% | 13.2% |
| Llama-2-7B-Chat | Prefill-decode | Balanced | point_014 | 1.32× | 88.3% | 76.9% |
| Llama-3.1-8B-Instruct | Prefill-only | Max-speed | point_023 | 1.65× | 67.7% | 46.6% |
| Llama-3.1-8B-Instruct | Prefill-only | Balanced | point_015 | 1.34× | 97.5% | 94.2% |
| Llama-3.1-8B-Instruct | Prefill-decode | Max-speed | point_020 | 1.61× | 53.8% | 18.4% |
| Llama-3.1-8B-Instruct | Prefill-decode | Balanced | point_013 | 1.34× | 95.5% | 94.6% |

Balanced 点是更适合正文强调的结果：两模型的 decode Balanced E2E 分别达到 1.32× 和
1.34×，高于各自所有 uniform baseline 的 1.21× 和 1.27×；对应 TTFT/TPOT 分别为
1.16×/1.44× 与 1.18×/1.48×。Llama-2 prefill Balanced
以 98.4% 平均质量保持率达到 1.25×，Llama-3.1 prefill Balanced 以 97.5% 平均质量
保持率达到 1.34×。Max-speed 行用于展示速度上界，也明确展示了相应质量代价。

完整审计覆盖 14 个策略 × 12 个当前样本，共 168 个互不相同的进程 ID，全部运行在
GPU UUID `305f915b-c789-ebb0-e184-56b64931412f`；TTFT/E2E 最大 CV 分别为 0.64% 和
0.78%。Llama-3.1 balanced 首轮 TTFT 的 CV 为 4.67%，因此按同一协议整组重测，首轮
6 份原始记录和 summary 保存在 `measurements/decode_components/superseded_unstable_ttft/`
供审计，而不是从首轮结果中挑选较快样本。

## 来源与注意事项

| Model | Scenario | Source |
|---|---|---|
| Llama-2-7B-Chat | Prefill-only | `artifacts/debug/064_llama2_pro6000_prefill_only` |
| Llama-2-7B-Chat | Prefill-decode | `artifacts/debug/065_llama2_pro6000_prefill_decode` |
| Llama-3.1-8B-Instruct | Prefill-decode | `artifacts/debug/066_llama31_pro6000_prefill_decode` |
| Llama-3.1-8B-Instruct | Prefill-only | `artifacts/debug/067_llama31_pro6000_prefill_only` |

066 的下游任务按已冻结的 Legacy/raw-text PMPD prompt 协议生成；选点和速度/质量结论均
使用同一 066 内的 BF16 与压缩点进行比较。不要把这些绝对生成分数与 native chat-template
结果混用。

## 重建

已有测量的离线重建在仓库根目录执行：

```bash
python artifacts/debug/068_two_model_two_scenario_pro6000_result_consolidation/scripts/validate_decode_components.py
python artifacts/debug/068_two_model_two_scenario_pro6000_result_consolidation/scripts/consolidate.py
```

如需从 checkpoint 重新执行全部 GPU 分项测量，使用 vLLM 环境运行
`scripts/run_decode_components.py --gpu 0`。所有新增策略快照、原始记录、临时 checkpoint
和派生结果均限制在 068；成功测量后临时 checkpoint 会自动删除。
