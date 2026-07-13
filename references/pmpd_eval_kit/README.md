# PMPD-Style Evaluation Kit

This folder is intended to be copied into another project and used as the
standard evaluation entrypoint.

It follows the PMPD repository's regular-task evaluation style:

- batch size 1
- greedy decoding by logits argmax
- `max_new_tokens=256`
- FastChat Claude-style prompt wrapping
- CNN/DM test split for summarization
- DialogSum test split for dialogue summarization
- IWSLT 2017 En-Fr data, evaluated as French -> English
- IWSLT filtering: keep examples whose English reference length is greater than
  60 tokens under `lmsys/vicuna-7b-v1.5`

## Expected Local Assets

Default paths in the original environment are:

```text
/home/agent/wja/data/datasets/flaxquant
/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf
/home/agent/wja/data/models/bert_score/roberta-large
```

Use a chat/instruction-tuned model when possible. PMPD's prompt format is a
chat-style FastChat template; base Llama models can produce whitespace-only or
run-on outputs under this prompt and are not a good early-test baseline.

The dataset root may contain either Hugging Face repo snapshots:

```text
cnn_dailymail_repo/
dialogsum_repo/
iwslt2017_en_fr_repo/
```

or Hugging Face `save_to_disk` directories where applicable.

For strict IWSLT PMPD filtering, make sure this tokenizer is available:

```bash
hf download lmsys/vicuna-7b-v1.5 \
  --local-dir /home/agent/wja/data/models/lmsys/vicuna-7b-v1.5
```

Then pass:

```bash
--iwslt-filter-tokenizer /home/agent/wja/data/models/lmsys/vicuna-7b-v1.5
```

If you omit it, the script defaults to `lmsys/vicuna-7b-v1.5` and may access
Hugging Face.

## CNN/DM 1000-Example Subset

Create a stable 1000-example CNN/DM test subset once:

```bash
python pmpd_eval_kit/make_cnn_dm_subset.py \
  --data-root /home/agent/wja/data/datasets/flaxquant
```

This creates:

```text
/home/agent/wja/data/datasets/flaxquant/cnn_dailymail_3.0.0_test_random1000_seed42
```

Use it with `--dataset cnn_dm_1000`.

## Full PMPD-Style Runs

```bash
CUDA_VISIBLE_DEVICES=0 python pmpd_eval_kit/pmpd_eval.py \
  --dataset cnn_dm \
  --model-path /home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf \
  --model-id llama2-7b-chat \
  --bertscore-model /home/agent/wja/data/models/bert_score/roberta-large \
  --output-dir outputs/pmpd_style_llama2_7b_chat
```

```bash
CUDA_VISIBLE_DEVICES=1 python pmpd_eval_kit/pmpd_eval.py \
  --dataset dsum \
  --model-path /home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf \
  --model-id llama2-7b-chat \
  --bertscore-model /home/agent/wja/data/models/bert_score/roberta-large \
  --output-dir outputs/pmpd_style_llama2_7b_chat
```

```bash
CUDA_VISIBLE_DEVICES=2 python pmpd_eval_kit/pmpd_eval.py \
  --dataset IWSLT \
  --model-path /home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf \
  --model-id llama2-7b-chat \
  --iwslt-filter-tokenizer /home/agent/wja/data/models/lmsys/vicuna-7b-v1.5 \
  --output-dir outputs/pmpd_style_llama2_7b_chat
```

## Quick CNN/DM Subset Run

```bash
CUDA_VISIBLE_DEVICES=0 python pmpd_eval_kit/pmpd_eval.py \
  --dataset cnn_dm_1000 \
  --model-path /home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf \
  --model-id llama2-7b-chat \
  --bertscore-model /home/agent/wja/data/models/bert_score/roberta-large \
  --output-dir outputs/pmpd_style_quick
```

## Metrics-Only

Recompute metrics from an existing PMPD-style JSONL answer file:

```bash
CUDA_VISIBLE_DEVICES=0 python pmpd_eval_kit/pmpd_eval.py \
  --dataset cnn_dm_1000 \
  --metrics-only outputs/pmpd_style_quick/cnn_dm_1000/llama2-7b-fp16.jsonl \
  --bertscore-model /home/agent/wja/data/models/bert_score/roberta-large
```

Outputs:

- `<output-dir>/<dataset>/<model-id>-fp16.jsonl`
- `<output-dir>/<dataset>/metrics.json`
- `<output-dir>/<dataset>/run_config.json`

`metrics.json` includes both PMPD-style ratio values such as `rougeL` and
paper-table-friendly percentage values such as `rougeL_percent`. It also
reports `empty_predictions`. Whitespace-only generations are treated as wrong:
Rouge/SacreBLEU receive a minimal placeholder prediction, and BERTScore assigns
those examples score `0` while also reporting `bert_score_non_empty`.

## Sanity Reference From This Environment

Using `/home/agent/wja/data/models/shakechen/Llama-2-7b-chat-hf`:

```text
cnn_dm_1000: Rouge-L 23.46, BERTScore 87.26, empty_predictions 0/1000
dsum:        Rouge-L 21.52, BERTScore 87.28, empty_predictions 0/1500
IWSLT:       Rouge-L 46.74, SacreBLEU 18.70, empty_predictions 0/333
```

The earlier base Llama2-7B run was much less reliable under PMPD's chat-style
prompt: it produced whitespace-only outputs and substantially lower scores.
