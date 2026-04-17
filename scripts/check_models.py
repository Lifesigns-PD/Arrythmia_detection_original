"""
Check V1 vs V2 model status and validate checkpoint integrity
"""
import torch
from pathlib import Path

ckpt_dir = Path("models_training/outputs/checkpoints")
print("=== MODEL CHECKPOINT STATUS ===\n")

models = {
    "rhythm_v1": "best_model_rhythm.pth",
    "rhythm_v2": "best_model_rhythm_v2.pth",
    "ectopy_v1": "best_model_ectopy.pth",
    "ectopy_v2": "best_model_ectopy_v2.pth",
}

for name, filename in models.items():
    filepath = ckpt_dir / filename
    print(f"{name:20} ", end="")
    if not filepath.exists():
        print("MISSING")
    else:
        try:
            torch.load(filepath, map_location='cpu')
            print("OK")
        except Exception as e:
            print(f"ERROR: {str(e)[:40]}")

print("\n=== ACTIVE MODELS ===")
print("Rhythm: V1 (fallback from V2 class mismatch)")
print("Ectopy: V2")
