"""Require the entire submitted batch to fit BF16 KV, preventing phase mixing."""
import math


def check(llm, batch, prompt, output):
    cache = llm.llm_engine.vllm_config.cache_config
    assert cache.num_gpu_blocks is not None
    capacity = cache.num_gpu_blocks*cache.block_size
    required = batch*math.ceil((prompt+output)/cache.block_size)*cache.block_size
    assert capacity >= required, {'kv_capacity_tokens':capacity,'required_tokens':required,
        'batch':batch,'prompt':prompt,'output':output,
        'reason':'Global-phase execution requires the whole homogeneous batch to be admitted.'}
    return {'kv_capacity_tokens':capacity,'required_tokens':required,'block_size':cache.block_size,
            'whole_batch_fits':True}
