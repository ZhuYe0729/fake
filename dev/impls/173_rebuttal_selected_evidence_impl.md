## 2026-10-28 - 五位人工reviewer精选证据与回复草稿
- 目的：按用户要求从已完成实验中筛选有利且可追溯的rebuttal证据，不汇编所有结果或失败记录。
- 核对034–048共15个run实际worker及各自审计，均complete/audited；047/048已完成，旧night161索引运行中状态未作为实时依据。未启动、停止或改动任何实验控制器、原始结果或主表。
- 新增artifacts/rebuttal/REBUTTAL_SELECTED_EVIDENCE.md：按五位人工reviewer整理12张表（含reviewer映射），覆盖实际预算、双表示内存、Qwen同轮强基线、Llama-2代表点、同候选Hessian控制、六负载、真实mixed、两类预测验证、剪枝及精确代理求解；提供五段英文回复草稿和使用优先级。保留影响所选结论成立的比较条件，q/实际ε、E2E/cycle、Graph/mixed及求解/全准备成本分开表述。
- 来源CSV自动生成数值表，49份输入记录SHA；质量六点重新计算mean+2SE，六负载12点E2E/cycle对INT8优势复核，18个原绝对延迟上限/每点24次测量与768条接口记录核对。12张表/60行数据、文档链接和强调格式检查通过，未将经验统计当硬保证。
- 影响文件：上述精选Markdown、selected_evidence173/source_audit.json、responses/README.md新增入口、plan/impl173。完整结果与不利点仍保留在原run；30/32B仅有shape/kernel证据，不写成完整部署。
