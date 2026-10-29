# FakeVLM eval.py 适配修改说明

将 `scripts/eval.py` 从原始 transformers 4.45.2 + flash_attn 环境迁移到新版 transformers 5.9.0 + PyTorch SDPA 环境所需的修改。

## 环境差异

| 项目 | 原始环境 | 目标环境 |
|------|----------|----------|
| transformers | 4.45.2 | 5.9.0 |
| 推理后端 | flash_attention_2 | PyTorch SDPA |
| 模型加载 | 远程 HuggingFace | 本地路径 |

## 修改清单

### 1. 替换 flash_attention_2 为 SDPA

**位置：** `load_model()` 函数内 `LlavaForConditionalGeneration.from_pretrained()` 调用

```diff
-        use_flash_attention_2=True,
-        revision='a272c74',
+        attn_implementation="sdpa",
```

同时也删除了 `revision='a272c74'`（本地模型不需要指定 revision）。

### 2. Processor 改为从本地模型路径加载

**位置：** 两处 — `legion_cls_dataset.__init__()` 和 `validate()` 函数

```diff
-        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')
+        self.processor = AutoProcessor.from_pretrained(args.model_path)
```

```diff
-    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')
+    processor = AutoProcessor.from_pretrained(args.model_path)
```

### 3. 修复 processor 的 image token 数量计算

**位置：** 紧跟上面两处 `AutoProcessor.from_pretrained()` 之后，各新增一行

```python
processor.vision_feature_select_strategy = None
```

**原因：** transformers 5.9.0 的 processor 在 `vision_feature_select_strategy == "default"` 时会对 image token 数量减 1（576 → 575），但模型实际输出的 image feature 数量为 576，导致运行时报错：

> `ValueError: Image features and image tokens do not match, tokens: 575, features: 2359296`

设为 `None` 跳过该减法，使 token 数与 feature 数一致。

### 4. 修复 tensor 维度挤压

**位置：** `validate()` 函数的推理循环内

```diff
-            inputs["input_ids"] = inputs["input_ids"].squeeze().to(device)
-            inputs["attention_mask"] = inputs["attention_mask"].squeeze().to(device)
-            inputs["pixel_values"] = inputs["pixel_values"].squeeze().to(device)
+            inputs["input_ids"] = inputs["input_ids"].squeeze(1).to(device)
+            inputs["attention_mask"] = inputs["attention_mask"].squeeze(1).to(device)
+            inputs["pixel_values"] = inputs["pixel_values"].squeeze(1).to(device)
```

**原因：** 新版 transformers 的 processor 返回的 tensor 已带 batch 维度（shape `[1, seq_len]`），DataLoader 叠加后变成 `[1, 1, seq_len]`。`.squeeze()` 会移除所有大小为 1 的维度，导致 batch 维度也被去掉（变成 `[seq_len]`），后续 `model.generate()` 报错 `IndexError: tuple index out of range`。`.squeeze(1)` 只移除 processor 多余的维度，保留 DataLoader 的 batch 维度。

## 最终 diff 总览

```diff
--- a/scripts/eval.py
+++ b/scripts/eval.py
@@ -44,7 +44,8 @@ class legion_cls_dataset(Dataset):
         elif train == False:
             with open(args.test_json_file, 'r') as f:
                 self.data = json.load(f)
-        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')
+        self.processor = AutoProcessor.from_pretrained(args.model_path)
+        self.processor.vision_feature_select_strategy = None
 
     def __len__(self):
         return len(self.data)
@@ -79,9 +80,8 @@ def load_model(args):
     model = LlavaForConditionalGeneration.from_pretrained(
         args.model_path,
         torch_dtype=torch.bfloat16,
         low_cpu_mem_usage=True,
-        use_flash_attention_2=True,
-        revision='a272c74',
+        attn_implementation="sdpa",
     ).eval().cuda()
     print("Successfully loaded model from:", args.model_path)
     return model
 
@@ -131,7 +131,8 @@ def calculate_results_acc(results):
 
 def validate(args, model, cls_test_dataloader):
-    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')
+    processor = AutoProcessor.from_pretrained(args.model_path)
+    processor.vision_feature_select_strategy = None
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     results = {}
     outputs = []
@@ -139,9 +140,9 @@ def validate(args, model, cls_test_dataloader):
         for inputs, labels, paths, cates in tqdm(cls_test_dataloader):
 
-            inputs["input_ids"] = inputs["input_ids"].squeeze().to(device)
-            inputs["attention_mask"] = inputs["attention_mask"].squeeze().to(device)
-            inputs["pixel_values"] = inputs["pixel_values"].squeeze().to(device)
+            inputs["input_ids"] = inputs["input_ids"].squeeze(1).to(device)
+            inputs["attention_mask"] = inputs["attention_mask"].squeeze(1).to(device)
+            inputs["pixel_values"] = inputs["pixel_values"].squeeze(1).to(device)
```

## 运行命令

```bash
python scripts/eval.py \
  --model_path /path/to/your/model \
  --val_batch_size 8 \
  --workers 4 \
  --output_path results/fakevlm.json \
  --test_json_file "/path/to/test.json" \
  --data_base_test "/path/to/test/images/"
```