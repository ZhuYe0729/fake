# MaxViT

Uses timm MaxViT checkpoints and ImageNet validation data. Compressible modules
are attention/MLP Linear layers and MBConv pointwise 1x1 convolutions, excluding
the classifier head, stem, SE, shortcut, grouped, and spatial convolutions.

See `config.example.toml`, `model.py`, and `../smoke_selectors.py`. ImageNet zip
loading and top-k evaluation are shared under `../common/`.

Run from the DASO root:

```bash
python -m extras.maxvit.evaluate --config extras/maxvit/config.example.toml \
  --output /tmp/maxvit.json
```
