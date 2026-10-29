# Run014: W4A16 TPOT diagnosis

Independent diagnosis of the run013 operator / full-model decode discrepancy.
Existing experiments and the published working Table 2 are retained unchanged.
See `dev/plans/147_qwen3_decode_diagnosis_plan.md` for scope and success criteria.

Completed: 3 fixed-KV model controls, 18 independent generation controls,
and 3 same-engine ABBA experiments. See [report](REPORT.md) and
[verified audit](results/final_audit.json). Native decode graphs recover
1.85× / 1.89× TPOT speedups for NVFP4 / INT4 W4A16 against graph-enabled BF16.
The mixed DASO/HAWQ table retains its original eager protocol.
