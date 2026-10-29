# DINOv3

Supports DINOv3 ViT transformer q/k/v/o and gate/up/down projections. The model
repository and ImageNet data are external dependencies. This directory is a
staging adapter, not part of the paper LLM workflow.

`model.py` loads the backbone and official linear head, while `transforms.py`
reproduces the ImageNet preprocessing. Shared dataset/evaluation code is under
`../common/`.

Run `python -m extras.dinov3.evaluate --config CONFIG --output RESULT.json`
from the DASO root after filling in the external backbone, head, and dataset
paths.
