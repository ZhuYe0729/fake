import os
import torch
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fake.models.maxvit import load_maxvit_dense
from fake.models.dinov3 import load_dinov3_vit7b16_dense_classifier

def get_model_info_string(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dtypes = set(p.dtype for p in model.parameters())
    
    info = f"Model: {model_name}\n"
    info += f"dtypes: {dtypes}\n"
    info += f"Total Parameters: {total_params:,}\n"
    info += f"Trainable Parameters: {trainable_params:,}\n"
    info += "="*50 + "\n"
    info += str(model) + "\n"
    return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["maxvit", "dinov3"])
    args = parser.parse_args()

    output_dir = Path("artifacts/model_details")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model == "maxvit":
        from fake.models.maxvit import MAXVIT_VARIANTS
        for variant in MAXVIT_VARIANTS.keys():
            print(f"Loading MaxViT {variant}...")
            try:
                maxvit_model, _ = load_maxvit_dense(device="cpu", variant=variant)
                maxvit_info = get_model_info_string(maxvit_model, f"MaxViT {variant.capitalize()} TF 224 in1k")
                maxvit_path = output_dir / f"maxvit_{variant}_arch.txt"
                maxvit_path.write_text(maxvit_info)
                print(f"Saved MaxViT {variant} architecture to {maxvit_path}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error loading MaxViT {variant}: {e}")

    if args.model == "dinov3":
        print("Loading DINOv3...")
        try:
            dinov3_model, _ = load_dinov3_vit7b16_dense_classifier(device="cpu")
            dinov3_info = get_model_info_string(dinov3_model, "DINOv3 ViT-7B 16 Pretrain LVD-1689M (with Linear Head)")
            dinov3_path = output_dir / "dinov3_arch.txt"
            dinov3_path.write_text(dinov3_info)
            print(f"Saved DINOv3 architecture to {dinov3_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading DINOv3: {e}")

if __name__ == "__main__":
    main()
