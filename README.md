# Fake
这是一个用于针对一些模型进行模型轻量化和测试的仓库。

- 目录结构（粗略规划，如果后续实现的时候需要具体修改，请视情况进行修改，不要太过于被当前的设置约束）
```
fake/
├── README.md
├── AGENTS.md
├── configs/
│   ├── models/
│   │   ├── maxvit.yaml                 # MaxViT 模型加载、输入尺寸、预处理配置
│   │   └── dinov3_7b.yaml              # DINOv3 7B 模型加载、输入尺寸、预处理配置
│   ├── datasets/
│   │   └── imagenet.yaml               # ImageNet val/subset 路径与 dataloader 配置
│   ├── methods/
│   │   ├── dense_bf16.yaml             # dense bf16 baseline
│   │   ├── dense_nvfp4.yaml            # dense nvfp4 baseline
│   │   ├── sparse_unstructured_bf16.yaml
│   │   ├── sparse_2_4_bf16.yaml
│   │   ├── sparse_unstructured_nvfp4.yaml
│   │   └── sparse_2_4_nvfp4.yaml
│   └── experiments/
│       ├── accuracy_imagenet.yaml      # 精度评估实验组合
│       ├── kernel_bench.yaml           # kernel micro-benchmark 实验组合
│       └── e2e_bench.yaml              # 端到端速度实验组合
├── fake/
│   ├── __init__.py
│   ├── models/
│   │   ├── registry.py                 # 统一注册/加载 maxvit、dinov3 7B
│   │   ├── maxvit.py
│   │   └── dinov3.py
│   ├── data/
│   │   ├── imagenet.py                 # ImageNet dataset 与预处理
│   │   └── transforms.py
│   ├── compression/
│   │   ├── quantization/
│   │   │   ├── nvfp4.py                # nvfp4 权重量化与 packing 入口
│   │   │   └── calibration.py          # 标定/scale 统计
│   │   ├── pruning/
│   │   │   ├── unstructured.py         # 非结构化稀疏 mask 生成与应用
│   │   │   └── semi_structured.py      # 2:4 半结构化稀疏 mask 生成与应用
│   │   └── pipeline.py                 # dense/sparse/quant+sparse 组合流程
│   ├── kernels/
│   │   ├── cutlass/
│   │   │   ├── dense_bf16/             # 已有 dense bf16 CUTLASS 算子封装
│   │   │   ├── dense_nvfp4/            # 已有 dense nvfp4 CUTLASS 算子封装
│   │   │   ├── sparse_2_4_nvfp4/       # 已有 2:4 sparse nvfp4 算子封装
│   │   │   ├── sparse_unstructured_bf16/
│   │   │   └── sparse_unstructured_nvfp4/
│   │   ├── bindings/                   # Python/CUDA extension 绑定
│   │   └── dispatch.py                 # 按精度/稀疏模式选择 kernel
│   ├── evaluation/
│   │   ├── accuracy.py                 # ImageNet top-1/top-5 评估
│   │   ├── kernel_bench.py             # 单 kernel latency/throughput benchmark
│   │   └── e2e_bench.py                # 端到端模型推理速度 benchmark
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       ├── reproducibility.py
│       └── profiler.py
├── scripts/
│   ├── eval_accuracy.py                # 跑 ImageNet 精度矩阵
│   ├── bench_kernel.py                 # 跑 kernel 速度矩阵
│   ├── bench_e2e.py                    # 跑 e2e 速度矩阵
│   ├── prepare_model.py                # 导出/转换不同精度与稀疏格式模型
│   └── summarize_results.py            # 汇总实验结果表格
├── tests/
│   ├── test_quantization.py
│   ├── test_pruning.py
│   ├── test_kernel_dispatch.py
│   └── test_accuracy_smoke.py
├── third_party/
│   └── dinov3/                         # DINOv3 官方/外部代码，尽量隔离修改
├── artifacts/
│   ├── checkpoints/                    # 转换后的模型权重，不提交大文件
│   ├── masks/                          # 稀疏 mask 缓存
│   ├── calibration/                    # nvfp4 标定产物
│   └── results/                        # accuracy/kernel/e2e 结果
└── dev/
    ├── plans/                          # 规划文档
    └── impls/                          # 每个 plan 的实现记录
```

# 基础配置
- 当前关注的模型：
    - timm/maxvit_tiny_tf_224.in1k，路径：/data/home/scxj523/run/wja/data/models/timm/maxvit_tiny_tf_224.in1k/
    - dinov3-vit7b16-pretrain-lvd1689m
    > backbone路径：/data/home/scxj523/run/wja/data/models/facebook/dinov3-vit7b16-pretrain-lvd1689m  
    > imagenet head路径：/data/home/scxj523/run/wja/data/models/facebook/dinov3_vit7b16_imagenet1k_linear_head

- 当前关注的数据集
    - imagenet-1k val的subset，路径：/data/home/scxj523/run/wja/data/datasets/imagenet_val/

- 主要轻量化手段：
    - nvfp4量化
    - unstructured pruning
    - semi-structured pruning
    - joint method

# MaxViT dense baseline

当前已支持 `timm/maxvit_tiny_tf_224.in1k` dense 模型的 ImageNet 精度测试和纯模型 forward 速度测试。

- 精度测试：
```shell
sbatch scripts/slurm/eval_maxvit_dense_accuracy.sh
```

- 速度测试：
```shell
sbatch scripts/slurm/bench_maxvit_dense_speed.sh
```

- 结果路径：
    - 精度：`artifacts/results/maxvit_dense/accuracy.csv`
    - 速度：`artifacts/results/maxvit_dense/speed.csv`

速度测试使用随机输入，仅统计模型 forward，不包含数据读取、图片解码和预处理开销；CSV 中会记录 batch size、输入尺寸、dtype、warmup、iters、GPU、torch/cuda 版本等测试配置。

# DINOv3 ViT-7B dense baseline

当前已支持 `facebook/dinov3-vit7b16-pretrain-lvd1689m` dense 模型的 ImageNet linear head 精度测试和纯模型 forward 速度测试。

- 精度测试：
```shell
sbatch scripts/slurm/eval_dinov3_vit7b16_dense_accuracy.sh
```

- 速度测试：
```shell
sbatch scripts/slurm/bench_dinov3_vit7b16_dense_speed.sh
```

- 结果路径：
    - 精度：`artifacts/results/dinov3_vit7b16_dense/accuracy.csv`
    - 速度：`artifacts/results/dinov3_vit7b16_dense/speed.csv`

DINOv3 dense baseline 使用 backbone 原始 dtype，通过本地 ImageNet linear head 做分类；分类输入遵循 DINOv3 hub classifier 逻辑，即 `cls token` 拼接 `patch tokens mean`。图像预处理参考 `third_party/dinov3/README.md` 中 LVD-1689M 的 ImageNet transform，默认 resize 到 256。
