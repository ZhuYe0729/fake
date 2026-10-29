#!/usr/bin/env python3
"""No-checkpoint smoke test for optional model module selectors."""
from __future__ import annotations

import torch.nn as nn

from daso.compression import select_compressible_modules


class TinyLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 16))])
        self.vision_tower = nn.Linear(8, 8)
        self.multi_modal_projector = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 8)


class TinyMaxViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(8, 8)
        self.conv1_1x1 = nn.Conv2d(8, 8, 1)
        self.head = nn.Linear(8, 2)


class Attention(nn.Module):
    def __init__(self, wrapped: bool = False) -> None:
        super().__init__()
        if wrapped:
            self.q_proj = nn.Module(); self.q_proj.base_layer = nn.Linear(8, 8)
        else:
            self.q_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)


class TinyVisionTransformer(nn.Module):
    def __init__(self, mirror: bool = False) -> None:
        super().__init__()
        block = nn.Module(); block.attention = Attention(mirror)
        self.backbone = nn.Module()
        if mirror:
            self.backbone.dino = nn.Module(); self.backbone.dino.layer = nn.ModuleList([block])
        else:
            self.backbone.layer = nn.ModuleList([block])


def main() -> None:
    llama = select_compressible_modules(TinyLanguage(), "llama")
    fakevlm = select_compressible_modules(TinyLanguage(), "fakevlm")
    maxvit = select_compressible_modules(TinyMaxViT(), "maxvit")
    dinov3 = select_compressible_modules(TinyVisionTransformer(), "dinov3")
    mirror = select_compressible_modules(TinyVisionTransformer(mirror=True), "mirror")
    assert llama and fakevlm and maxvit and dinov3 and mirror
    assert all("vision_tower" not in item.name for item in fakevlm)
    assert all("multi_modal_projector" not in item.name for item in fakevlm)
    assert all(item.name != "head" for item in maxvit)
    print({"llama": len(llama), "fakevlm": len(fakevlm), "maxvit": len(maxvit),
           "dinov3": len(dinov3), "mirror": len(mirror)})


if __name__ == "__main__":
    main()
