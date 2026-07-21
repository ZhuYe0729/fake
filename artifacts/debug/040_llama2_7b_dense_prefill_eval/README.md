# Llama2-7B dense quality evaluation

This experiment evaluates only the original dense `Llama-2-7b-chat-hf` model.
It intentionally follows the recent 037 ARC-Challenge quality path: lm-eval's
`HFLM` adapter over Transformers, not vLLM.

## Tasks

- PPL: `wikitext`, `c4`
- Accuracy, all 0-shot: `winogrande`, `arc_easy`, `mmlu`

## Run

```bash
source /home/agent/wja/miniconda3/etc/profile.d/conda.sh
conda activate cospaq
python artifacts/debug/040_llama2_7b_dense_prefill_eval/scripts/run_all.py --gpus 0,1,2,3
```

Use `--limit 8` for a smoke run. Results are separated by profile, so smoke
outputs never suppress the full run.

Rolling PPL tasks use batch size 1 by default to accommodate long documents;
the multiple-choice tasks use batch size 4.

If a dataset download needs the local proxy, append `--proxy` to the command.
This also sets `HF_HUB_DISABLE_XET=1`, avoiding the Xet transfer path that
failed for the C4 validation shard on this machine.
