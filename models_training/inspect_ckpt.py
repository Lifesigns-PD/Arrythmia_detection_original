
import torch
import numpy as np
from pathlib import Path

# Add global to allow unpickling if needed
# torch.serialization.add_safe_globals([np._core.multiarray.scalar])

ckpt_path = Path(__file__).resolve().parent / "outputs" / "checkpoints" / "best_model_rhythm.pth"
if ckpt_path.exists():
    try:
        # Using weights_only=False because the metadata in the checkpoint needs it
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "class_names" in state:
            print(f"Count: {len(state['class_names'])}")
            print("Classes in checkpoint:")
            for i, name in enumerate(state["class_names"]):
                print(f"  {i}: {name}")
        else:
            print("No 'class_names' key in state dict.")
        
        # Check model architecture size
        sd = state.get("model_state", state)
        if "classifier.3.weight" in sd:
            print(f"Classifier weight shape: {sd['classifier.3.weight'].shape}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print(f"Checkpoint not found: {ckpt_path}")
