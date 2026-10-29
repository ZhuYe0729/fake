# llama2_7b_chat：SQ-FP8完整基线

固定原alpha=.5和activation scales，完整MC/GEN及两场景64窗口NLL；与原BF16/INT8/SQ-INT8/FP8同轮三轮测速。质量等037，正式速度等041。

[设计](configs/design.json) · [计划](../../../../dev/plans/167_llama_smoothquant_fp8_baseline_plan.md)
