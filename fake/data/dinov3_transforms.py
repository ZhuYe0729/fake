from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_dinov3_lvd1689m_transform(resize_size: int = 256) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

