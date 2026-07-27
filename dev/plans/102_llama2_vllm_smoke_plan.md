# Llama2 vLLM 单请求验证计划

1. 核验 vLLM 环境、自定义 backend 和本地模型路径。
2. 提供可由 multiprocessing spawn 启动的文件式单请求 smoke runner。
3. 使用原始 dense BF16 Llama2 模型完成一次确定性生成；随后等待可用 uniform checkpoint 验证各量化 backend。
