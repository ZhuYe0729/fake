# MIRROR

Supports the DINOv3 backbone projection layers inside MIRROR. Install MIRROR at
the revision recorded by your run manifest and provide model, memory, backbone,
and evaluation dataset paths through configuration.

`model.py` loads either the upstream dense checkpoint or a prepared compressed
state without embedding a repository or data path. `datasets.py` and
`evaluate.py` provide the retained Chameleon/GenImage discovery and detection
metrics. Use `--discover-only` to validate a dataset without loading a model.
