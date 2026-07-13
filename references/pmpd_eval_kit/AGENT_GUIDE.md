# Agent Guide

Use this folder as the only evaluation interface for early experiments.

Do not use older non-PMPD regular-evaluation scripts from the source project.
The intended standard is PMPD-style evaluation:

- `pmpd_eval.py` is the main entrypoint.
- `make_cnn_dm_subset.py` creates the fixed CNN/DM 1000-example quick-test set.
- Default generation is greedy, batch size 1, and `max_new_tokens=256`.
- Prompts follow PMPD's FastChat Claude-style wrapping.
- CNN/DM and DialogSum report Rouge-L and BERTScore.
- IWSLT reports Rouge-L and SacreBLEU, using French -> English prompts.
- IWSLT must be filtered with `lmsys/vicuna-7b-v1.5` tokenizer for strict PMPD compatibility.
- Prefer chat/instruction-tuned models. Base Llama-style models can produce
  whitespace-only or run-on outputs under PMPD's chat-style prompt.
- Always inspect `empty_predictions` in `metrics.json`; it should usually be
  near zero for a usable early-test baseline.

Minimum smoke test after copying:

```bash
CUDA_VISIBLE_DEVICES=0 python pmpd_eval_kit/pmpd_eval.py \
  --dataset cnn_dm_1000 \
  --question-end 2 \
  --model-path /path/to/chat-model \
  --model-id model-name \
  --bertscore-model /path/to/roberta-large \
  --output-dir outputs/pmpd_style_smoke
```

Full quick-test run:

```bash
CUDA_VISIBLE_DEVICES=0 python pmpd_eval_kit/pmpd_eval.py \
  --dataset cnn_dm_1000 \
  --model-path /path/to/chat-model \
  --model-id model-name \
  --bertscore-model /path/to/roberta-large \
  --output-dir outputs/pmpd_style_quick
```
