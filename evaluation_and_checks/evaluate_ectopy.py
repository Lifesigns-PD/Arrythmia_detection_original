
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "models_training"))

from models_training.data_loader import ECTOPY_CLASS_NAMES, ECGRawDatasetSQL, collate_fn
from models_training.models import CNNTransformerClassifier

def evaluate_ectopy(ckpt_path):
    print(f"📊 EVALUATING ECTOPY MODEL: {ckpt_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(ECTOPY_CLASS_NAMES)
    
    # Load Model
    model = CNNTransformerClassifier(num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()

    print("Loading Validation Data...")
    dataset = ECGRawDatasetSQL(task="ectopy", sql_limit=1000, augment=False)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    y_true = []
    y_pred = []

    print("Running Inference...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())

    # METRICS for ECTOPY
    # F1 Score is better for events (Recall + Precision balance)
    print("\n" + "="*60)
    print("ECTOPY EVALUATION REPORT")
    print("="*60)
    
    print(classification_report(
        y_true, y_pred, 
        target_names=ECTOPY_CLASS_NAMES, 
        labels=range(len(ECTOPY_CLASS_NAMES)),
        zero_division=0
    ))

    # False Alarm Rate for Runs/PVCs
    # "None" is class 0 usually
    none_idx = 0
    total_non_events = sum(1 for y in y_true if y == none_idx)
    false_alarms = sum(1 for yt, yp in zip(y_true, y_pred) if yt == none_idx and yp != none_idx)
    
    far = false_alarms / total_non_events if total_non_events > 0 else 0.0
    print(f"\nNON-EVENT FALSE ALARM RATE: {far:.2%} ({false_alarms}/{total_non_events})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    else:
        # Default to latest ectopy checkpoint
        ckpt = "outputs/checkpoints/best_model_ectopy.pth"
    
    if os.path.exists(ckpt):
        evaluate_ectopy(ckpt)
    else:
        print(f"Checkpoint not found: {ckpt}")
