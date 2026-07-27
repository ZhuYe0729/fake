## 2026-07-17 - 当前 GPU 校准入口
- 开发目的：避免 Llama2 代理建模混入历史 GPU 数据，并让 Pareto solver 使用本次 phase-local quality proxy 与本机 kernel predictor。
- 修改内容：重写根目录 `profile.sh` 为 `quality`、`kernel`、`solve`、`all` 分阶段入口；修复统一 CUDA visible-device 映射并隔离 run root。033 local-error/NLL 脚本新增本机 model/prepared-root 参数。034 solver 新增 `--predictor-root` 与 `--phase-local-errors`，并修复 wrapper 路径。
- 验证：`bash -n profile.sh`、三个 Python 文件 `py_compile`、034 `--help` 通过；实际完成 `profile.sh kernel`，输出 `104_llama2_rtx_pro_6000_kernel_smoke/kernel_profile/modeling`；使用历史 033 输入完成 034 临时接口 smoke（12 个候选点），仅验证连接，不作为新质量结果。
- 后续注意：`quality` 会运行 72 条 policy 的两场景 WikiText NLL，耗时较长；`all` 产物仍是 predicted candidates，必须再以当前 GPU 的 vLLM E2E 实测拟合新的 035/037 单调校准器后才可作为最终候选筛选依据。

## 2026-07-17 - 修复本机离线 WikiText 输入
- 开发目的：解决新流程在 `generate_inputs.py` 中回退到旧机器模型路径、并尝试对只读 Hugging Face dataset cache 加锁的问题。
- 修改内容：为 WikiText block builder 增加显式 tokenizer 与 Arrow dataset 输入；033 生成器要求传入本机 `--model-path`、`--cache-dir`、`--dataset-arrow`。`profile.sh` 固定使用本机 `/root/data/huggingface/datasets/.../wikitext-train.arrow`，并启用离线 Hugging Face 环境变量。
- 验证：用当前 Llama2 tokenizer 与只读本地 Arrow cache 成功生成 1 个 `(1, 2129)` WikiText block 和 72 个 policy manifest；相关 Python 文件 `py_compile` 与 `bash -n profile.sh` 通过。

## 2026-07-17 - 统一 checkpoint 导出入口
- 开发目的：按 Runbook §2.2 将 uniform 压缩、vLLM checkpoint 导出和内容校验整理到单一入口。
- 修改内容：新增根目录 `01_export_uniform_checkpoints.sh`，固定当前模型/wrapper/离线 WikiText Arrow 路径，准备 sparse BF16、dense NVFP4、sparse NVFP4、Marlin NVFP4，导出 fused vLLM checkpoint，并校验每个 `quant_method`、manifest 与 safetensors。prepare 脚本新增 `--dataset-arrow`，可在只读 HF cache 上运行。
- 验证：`bash -n`、prepare 脚本 `py_compile`，以及 prepare/export CLI 参数检查通过；未执行压缩或 checkpoint 导出。

## 2026-07-17 - Uniform speed 与质量入口
- 开发目的：按 Runbook §2.3/§2.4 为 uniform baselines 整理当前机器可执行的 speed、PMPD quality、ARC-Challenge 与汇总流程。
- 修改内容：新增根目录 `02_run_uniform_speed.sh` 与 `03_run_uniform_quality.sh`；speed 固定运行两个论文 workload。quality 对五种方法运行 CNN/DM、DialogSum、IWSLT，并通过扩展后的 ARC evaluator 生成 uniform policy 后测 ARC-Challenge。speed runner 读取当前模型环境变量；ARC evaluator 接受本机 model/prepared paths；summary 新增 `acc`/`acc_norm` 和 ARC 表格列。
- 验证：两个 shell `bash -n`、三个 Python 文件 `py_compile`、ARC/speed CLI 参数检查、相关 fake diff whitespace 检查通过；未执行实际 speed 或 quality workload。

## 2026-07-17 - 当前 GPU 的 policy E2E 速度校准入口
- 开发目的：在本机 profile 生成的 Llama2 Pareto policy 上执行 035/037 风格的真实 vLLM E2E 测量，避免沿用历史 GPU 的校正数据。
- 修改内容：新增根目录 `05_e2e_calibration.sh`，按 prefill-only / prefill-decode 的训练点与严格留出点导出 phase-heterogeneous checkpoint、以统一 vLLM runner 测量 main phase、拟合 PAVA 单调 raw-kernel-sum→E2E 映射，并重新输出含 `predicted_e2e_ms` 的 Pareto CSV。
- 后续注意：PAVA 是全局单调变换，因此同一质量约束下 policy assignment 与 raw-latency solver 一致；该步骤校正速度数值并验证代理，而不是凭空改变 policy。最终仍需对少量入选 policy 实测 WikiText NLL 和 E2E 后进行实测 Pareto 筛选。

## 2026-07-18 - ARC 本地缓存与 datasets 3.x 兼容
- 开发目的：修复 vLLM 环境运行 ARC-Challenge 时，旧离线 Arrow metadata 中的 `List` feature 无法被 datasets 3.x 识别的问题。
- 修改内容：ARC evaluator 不再调用 datasets 3.x 的旧 cache builder；而是在 lm-eval 加载 ARC 时，将请求定向到本地 train/validation/test Arrow splits。旧 `List` feature 注册为与 Arrow schema 一致的 `Sequence`，不下载、不改写原始数据集缓存。
- 验证：先复现 `LargeList` 与 Arrow list 的 schema mismatch；直接加载本地 Arrow 后验证 validation 299 条、test 1172 条和 choices 字段可读。

## 2026-07-18 - ARC summary 字段归一化
- 开发目的：修复 uniform quality 汇总中 ARC 的样本数显示为 Python 字典、空预测显示为 `None`。
- 修改内容：summary reader 对 lm-eval 的 `num_samples` 字典取 `effective` 样本数；缺少 `empty_predictions` 的 ARC 记录写为空字段（不适用），而不是字符串 `None`。
- 验证：以已有五个 ARC metrics 重建 baseline summary，预期每条 ARC 显示 1172 samples 与空 empty 列。

## 2026-07-18 - ARC 准确率的展示单位
- 开发目的：让 Markdown summary 的 ARC `acc_norm` 与同表的 Rouge-L、BERTScore 保持百分制展示一致。
- 修改内容：仅在 Markdown 展示层将 0--1 的 `acc_norm` 乘以 100，并在列名标注 `(%)`；CSV 继续保存原始比例，方便后处理。
