import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models_training"))

from data_loader import ECGRawDatasetSQL, CLASS_NAMES
from models import CNNTransformerClassifier

def collate_fn(batch):
    xs = torch.stack([torch.from_numpy(b["signal"]).unsqueeze(0) for b in batch], dim=0).float()
    ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    metas = [b["meta"] for b in batch]
    return xs, ys, metas

def main():
    print("Initializing directories...")
    output_dir = BASE_DIR / "test_output"
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    ckpt_path = BASE_DIR / "models_training" / "outputs" / "checkpoints" / "best_model_rhythm.pth"
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading existing state...")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Try to determine classes
    if "class_names" in state:
        used_class_names = state["class_names"]
    else:
        used_class_names = CLASS_NAMES

    num_classes = len(used_class_names)
    print(f"Model initialized with {num_classes} classes.")
    model = CNNTransformerClassifier(num_classes=num_classes)
    
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
        
    model.to(device)
    model.eval()

    print("Loading dataset...")
    # Try rhythm task if 13 classes, otherwise all
    task = "rhythm" if num_classes == 13 else "all"
    dataset = ECGRawDatasetSQL(task=task)
    
    # Limit dataset if it's too big, or use the whole thing.
    # For evaluation, doing the whole thing is good, but might take too long.
    # We will process all to get a good confusion matrix.
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    y_true = []
    y_pred = []
    metadata = []
    signals_stored = []
    
    print("Running inference...")
    with torch.no_grad():
        for x, y, metas in tqdm(loader, desc="Inference"):
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            y_pred.extend(preds)
            y_true.extend(y.numpy())
            metadata.extend(metas)
            
            # Store some signals for plotting later (convert to CPU right away to save VRAM)
            for i in range(len(preds)):
                signals_stored.append(x[i][0].cpu().numpy())

    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    
    plt.figure(figsize=(10, 8))
    # Basic heatmap using imshow
    cax = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(cax)
    
    # Add numbers in the cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.xticks(np.arange(num_classes), used_class_names, rotation=45, ha="right")
    plt.yticks(np.arange(num_classes), used_class_names)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
    
    with open(output_dir / "confusion_matrix.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=used_class_names, labels=np.arange(num_classes), zero_division=0))
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    print("Finding 5 examples per predicted arrhythmia class...")
    output_results = {}

    for cls_idx, cls_name in enumerate(used_class_names):
        # find predictions for this class
        indices = [i for i, p in enumerate(y_pred) if p == cls_idx]
        # just pick first 5
        selected = indices[:5]
        
        output_results[cls_name] = []
        for i, idx in enumerate(selected):
            true_label = used_class_names[y_true[idx]]
            meta = metadata[idx]
            
            entry = {
                "segment_id": meta.get("id"),
                "filename": meta.get("filename"),
                "true_label": true_label,
                "predicted_label": cls_name,
                "match": (cls_name == true_label)
            }
            output_results[cls_name].append(entry)
            
            # Plot the signal
            if meta.get("id") is not None:
                plt.figure(figsize=(10, 3))
                plt.plot(signals_stored[idx])
                plt.title(f"Segment ID: {meta.get('id')} - True: {true_label} | Pred: {cls_name}")
                plt.tight_layout()
                safe_name = cls_name.replace("/", "_").replace(" ", "_").replace("+", "plus")
                plt.savefig(plots_dir / f"{safe_name}_example_{i+1}_seg_{meta.get('id')}.png")
                plt.close()
                
    with open(output_dir / "used_segments_and_files.json", "w") as f:
        json.dump(output_results, f, indent=4)
        
    print(f"Done! Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
