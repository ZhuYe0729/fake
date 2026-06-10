import os
import torch
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fake.models.maxvit import load_maxvit_dense
from fake.models.dinov3 import load_dinov3_vit7b16_dense_classifier
from fake.models.qwen3_5 import load_qwen3_5_dense
from fake.models.llama import load_llama2_dense, load_llama31_dense

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
    output_dir = Path("artifacts/model_details")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading MaxViT...")
    try:
        maxvit_model, _ = load_maxvit_dense(device="cpu")
        maxvit_info = get_model_info_string(maxvit_model, "MaxViT Tiny TF 224 in1k")
        maxvit_path = output_dir / "maxvit_arch.txt"
        maxvit_path.write_text(maxvit_info)
        print(f"Saved MaxViT architecture to {maxvit_path}")
    except Exception as e:
        print(f"Error loading MaxViT: {e}")

    print("Loading DINOv3...")
    try:
        dinov3_model, _ = load_dinov3_vit7b16_dense_classifier(device="cpu")
        dinov3_info = get_model_info_string(dinov3_model, "DINOv3 ViT-7B 16 Pretrain LVD-1689M (with Linear Head)")
        dinov3_path = output_dir / "dinov3_arch.txt"
        dinov3_path.write_text(dinov3_info)
        print(f"Saved DINOv3 architecture to {dinov3_path}")
    except Exception as e:
        print(f"Error loading DINOv3: {e}")

    print("Loading Qwen3.5...")
    try:
        qwen3_5_model, _ = load_qwen3_5_dense(device="cpu")
        qwen3_5_info = get_model_info_string(qwen3_5_model, "Qwen3.5-0.6B")
        qwen3_5_path = output_dir / "qwen3_5_arch.txt"
        qwen3_5_path.write_text(qwen3_5_info)
        print(f"Saved Qwen3.5 architecture to {qwen3_5_path}")
    except Exception as e:
        print(f"Error loading Qwen3.5: {e}")

    print("Loading Llama-2-7B...")
    try:
        llama2_model, _ = load_llama2_dense(device="cpu")
        llama2_info = get_model_info_string(llama2_model, "Llama-2-7B")
        llama2_path = output_dir / "llama2_7b_arch.txt"
        llama2_path.write_text(llama2_info)
        print(f"Saved Llama-2-7B architecture to {llama2_path}")
    except Exception as e:
        print(f"Error loading Llama-2-7B: {e}")

    print("Loading Llama-3.1-8B...")
    try:
        llama31_model, _ = load_llama31_dense(device="cpu")
        llama31_info = get_model_info_string(llama31_model, "Llama-3.1-8B-Instruct")
        llama31_path = output_dir / "llama3_1_8b_arch.txt"
        llama31_path.write_text(llama31_info)
        print(f"Saved Llama-3.1-8B architecture to {llama31_path}")
    except Exception as e:
        print(f"Error loading Llama-3.1-8B: {e}")

if __name__ == "__main__":
    main()
