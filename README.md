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

- 当前关注的数据集
    - imagenet-1k val的subset，路径：/data/home/scxj523/run/wja/data/datasets/imagenet_val/

- 主要轻量化手段：
    - nvfp4量化
    - unstructured pruning
    - semi-structured pruning
    - joint method
