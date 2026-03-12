
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models_training"))

from models_training.data_loader import RHYTHM_CLASS_NAMES, ECGRawDatasetSQL, collate_fn
from models_training.models import CNNTransformerClassifier
from models_training.calibration import TemperatureScaling

def evaluate_rhythm(ckpt_path):
    print(f"📊 EVALUATING RHYTHM MODEL: {ckpt_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(RHYTHM_CLASS_NAMES)
    
    # Load Model
    model = CNNTransformerClassifier(num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()

    # Apply Temperature Scaling if valid
    # calibrator = TemperatureScaling(model).to(device)
    # calibrator.set_temperature(val_loader) (Requires separate val loader, skipping for now)

    # Load Data (Validation Set Only? Use full dataset for audit or holdout?)
    # Assuming we want to evaluate on a holdout or specific set.
    # For now, let's load the validation split from SQL if possible, or just a new pull.
    # Actually, let's just use the SQL dataset in eval mode.
    print("Loading Validation Data...")
    dataset = ECGRawDatasetSQL(task="rhythm", sql_limit=1000, augment=False) # Limit for speed in demo
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    y_true = []
    y_pred = []
    y_probs = []

    print("Running Inference...")
    with torch.no_grad():
        for x, y in loader:
            # OPTIONAL: SQI Check Simulation
            # if check_sqi(x) < 0.5: 
            #     preds = ARTIFACT_INDEX
            # else: ...
            
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # METRICS for RHYTHM
    # Sensitivity (Recall) is critical for Arrhythmia
    print("\n" + "="*60)
    print("RHYTHM EVALUATION REPORT")
    print("="*60)
    
    print(classification_report(
        y_true, y_pred, 
        target_names=RHYTHM_CLASS_NAMES, 
        labels=range(len(RHYTHM_CLASS_NAMES)),
        zero_division=0
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(RHYTHM_CLASS_NAMES)))
    present = sorted(set(y_true) | set(y_pred))
    if present:
        print("CONFUSION MATRIX (rows=true, cols=predicted):")
        hdr = "{:>25s}".format("") + "".join(f" {RHYTHM_CLASS_NAMES[c][:8]:>8s}" for c in present)
        print(hdr)
        for r in present:
            row_str = "{:>25s}".format(RHYTHM_CLASS_NAMES[r][:25])
            for c in present:
                row_str += f" {cm[r, c]:>8d}"
            print(row_str)
        print()

    # Specificity per class (TN / (TN + FP))
    print("PER-CLASS SPECIFICITY:")
    for i, cls in enumerate(RHYTHM_CLASS_NAMES):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        support = int(np.sum(cm[i, :]))
        if support > 0:
            print(f"  {cls:30s} | Specificity: {spec:.4f}  (support={support})")
    print()

    # Check specific critical classes
    print("\nCRITICAL CLASS PERFORMANCE (Recall/Sensitivity):")
    report = classification_report(y_true, y_pred, target_names=RHYTHM_CLASS_NAMES, output_dict=True, zero_division=0)
    
    criticals = ["Atrial Fibrillation", "Ventricular Tachycardia", "2nd Degree AV Block Type 2", "3rd Degree AV Block", "Pause"]
    print("\nCRITICAL CLASS PERFORMANCE (Recall & False Negative Rate):")
    for c in criticals:
        if c in report:
            rec = report[c]['recall']
            prec = report[c]['precision']
            fnr = 1.0 - rec
            print(f"  {c:30s} | Recall: {rec:.4f} | FNR: {fnr:.4f}")
        else:
            print(f"  {c:30s} | NOT PRESENT IN DATASET")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    else:
        # Default to latest rhythm checkpoint
        ckpt = str(BASE_DIR / "models_training" / "outputs" / "checkpoints" / "best_model_rhythm.pth")
    
    if os.path.exists(ckpt):
        evaluate_rhythm(ckpt)
    else:
        print(f"Checkpoint not found: {ckpt}")
