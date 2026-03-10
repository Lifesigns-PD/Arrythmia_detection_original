import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from models_training.retrain import ECGEventDataset, collate_fn
from models_training.models import CNNTransformerClassifier
from torch.utils.data import DataLoader
import torch

def test_loader_and_model():
    tasks = ["rhythm", "ectopy"]
    
    for task in tasks:
        print(f"\n--- Testing Task: {task} ---")
        try:
            ds = ECGEventDataset(task=task, source_filter="all")
            if len(ds) == 0:
                print(f"Dataset for {task} is empty. (Expected if no {task} events in DB)")
                continue
                
            loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
            x, y = next(iter(loader))
            print(f"Batch Shape: {x.shape} (Expected: torch.Size([4, 1, 2500]))")
            
            # Test Model Forward Pass
            num_classes = 22 if task == "rhythm" else 4
            model = CNNTransformerClassifier(num_classes=num_classes)
            output = model(x)
            print(f"Model Output Shape: {output.shape} (Expected: torch.Size([4, {num_classes}]))")
            
            if x.shape == torch.Size([4, 1, 2500]) and output.shape == torch.Size([4, num_classes]):
                print(f"✅ {task.upper()} Architectural Match Success!")
            else:
                print(f"❌ {task.upper()} Shape Mismatch!")
                
        except Exception as e:
            print(f"Error during {task} test: {e}")

if __name__ == "__main__":
    test_loader_and_model()
