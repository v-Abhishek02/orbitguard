"""
ORBITGUARD — Stage 2: CNN+LSTM Detection Model (FIXED)
File: src/cnn_lstm_fixed.py
Problem solved: 13 HIGH / 493 MED / 3439 LOW extreme imbalance
Solution: SMOTE oversampling + strong class weights + threshold tuning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, classification_report,
    confusion_matrix, precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
OUTPUTS_DIR   = Path("outputs")
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using Apple MPS (GPU acceleration)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✅ Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("✅ Using CPU")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING + SMOTE OVERSAMPLING
# ═════════════════════════════════════════════════════════════════════════════

def load_and_oversample():
    """
    Load data, apply SMOTE to fix extreme class imbalance.

    SMOTE (Synthetic Minority Oversampling TEchnique):
    - Takes your 13 HIGH risk samples
    - Finds their nearest neighbours in feature space
    - Interpolates between them to create NEW synthetic HIGH risk samples
    - Not just duplicates — genuinely new samples along the decision boundary
    - Result: balanced classes that your model can actually learn from
    """
    print("="*60)
    print("LOADING AND BALANCING DATA")
    print("="*60)

    X = np.load(PROCESSED_DIR / "X_sequences.npy")
    y = np.load(PROCESSED_DIR / "y_labels.npy")

    print(f"\nOriginal data:")
    print(f"  X shape: {X.shape}")
    print(f"  Total samples: {len(y):,}")

    class_names = {0: "LOW", 1: "MED", 2: "HIGH"}
    for label in [0, 1, 2]:
        count = (y == label).sum()
        print(f"  {class_names[label]}: {count:,}  ({count/len(y)*100:.2f}%)")

    # ── Split BEFORE oversampling ──────────────────────────────────────────
    # CRITICAL: never apply SMOTE to test data
    # Test data must stay as real samples only
    # Only oversample training data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nBefore SMOTE:")
    print(f"  Train: {len(y_train)} samples")
    for label in [0, 1, 2]:
        count = (y_train == label).sum()
        print(f"    {class_names[label]}: {count}")

    # ── Flatten sequences for SMOTE ────────────────────────────────────────
    # SMOTE works on 2D arrays: (n_samples, n_features)
    # Our X is 3D: (n_samples, seq_len, features)
    # Solution: flatten the last 2 dims, apply SMOTE, reshape back
    n_train, seq_len, n_feat = X_train.shape
    X_train_flat = X_train.reshape(n_train, seq_len * n_feat)

    # ── Normalise before SMOTE ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)

    # Save scaler
    with open(MODELS_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ── Apply SMOTE ────────────────────────────────────────────────────────
    # With only 13 HIGH samples, we need k_neighbors < 13
    # k_neighbors=5 is the default but we use 4 to be safe with 13 samples
    print(f"\nApplying SMOTE oversampling...")
    print(f"  This creates synthetic HIGH risk samples from your 13 real ones")

    # Target: balance all classes to same count as majority (LOW)
    # But we cap to avoid overfitting — use 500 per class minimum
    low_count  = (y_train == 0).sum()
    target_count = max(low_count, 500)

    smote = SMOTE(
        sampling_strategy={
            2: min(target_count, 800),   # HIGH: create up to 800 samples
            1: min(target_count, 800),   # MED:  create up to 800 samples
        },
        k_neighbors=min(4, (y_train == 2).sum() - 1),
        random_state=42
    )

    try:
        X_resampled, y_resampled = smote.fit_resample(
            X_train_scaled, y_train
        )
        print(f"  SMOTE succeeded!")
    except Exception as e:
        print(f"  SMOTE failed: {e}")
        print(f"  Falling back to manual oversampling...")
        # Manual oversampling: repeat minority samples
        X_resampled, y_resampled = manual_oversample(
            X_train_scaled, y_train, target_per_class=500
        )

    print(f"\nAfter SMOTE:")
    print(f"  Train: {len(y_resampled)} samples")
    for label in [0, 1, 2]:
        count = (y_resampled == label).sum()
        print(f"    {class_names[label]}: {count}")

    # ── Reshape back to 3D ─────────────────────────────────────────────────
    X_resampled_3d = X_resampled.reshape(-1, seq_len, n_feat)

    # ── Scale val and test with same scaler ────────────────────────────────
    n_val, n_test = len(X_val), len(X_test)
    X_val_scaled  = scaler.transform(
        X_val.reshape(n_val, seq_len * n_feat)
    ).reshape(n_val, seq_len, n_feat)
    X_test_scaled = scaler.transform(
        X_test.reshape(n_test, seq_len * n_feat)
    ).reshape(n_test, seq_len, n_feat)

    return (X_resampled_3d, y_resampled,
            X_val_scaled,   y_val,
            X_test_scaled,  y_test,
            seq_len, n_feat)


def manual_oversample(X, y, target_per_class=500):
    """
    Fallback if SMOTE fails.
    Repeats minority class samples with small random noise.
    """
    X_new, y_new = list(X), list(y)
    for label in [1, 2]:
        idx = np.where(y == label)[0]
        if len(idx) == 0:
            continue
        current = len(idx)
        needed  = target_per_class - current
        if needed <= 0:
            continue
        for _ in range(needed):
            # Pick random sample from this class
            src = X[np.random.choice(idx)]
            # Add tiny noise (1% of std)
            noise = np.random.normal(0, 0.01, src.shape)
            X_new.append(src + noise)
            y_new.append(label)

    X_new = np.array(X_new, dtype=np.float32)
    y_new = np.array(y_new, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(y_new))
    return X_new[perm], y_new[perm]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATASET
# ═════════════════════════════════════════════════════════════════════════════

class DebrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: MODEL ARCHITECTURE (same as before — architecture was correct)
# ═════════════════════════════════════════════════════════════════════════════

class DebrisDetector(nn.Module):
    def __init__(self, n_features=18, seq_length=20,
                 cnn_channels=64, lstm_hidden=128,
                 lstm_layers=2, n_classes=3, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels * 2),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn(x_cnn)
        x_lstm = x_cnn.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_lstm)
        attn_scores  = self.attention(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return self.classifier(context)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)

    def predict_proba(self, x):
        """Return class probabilities (softmax of logits)."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: TRAINING WITH STRONG CLASS WEIGHTS
# ═════════════════════════════════════════════════════════════════════════════

def compute_class_weights(y, device, power=2.0):
    """
    Compute class weights as inverse frequency raised to a power.
    power=2.0 means HIGH risk gets 264² = 69,696x more weight than LOW.
    This forces the model to never ignore HIGH risk events.
    """
    unique, counts = np.unique(y, return_counts=True)
    freq = counts / len(y)
    weights = (1.0 / freq) ** power
    weights = weights / weights.sum() * len(unique)  # normalise

    weight_tensor = torch.FloatTensor(weights).to(device)

    print(f"\nClass weights (power={power}):")
    names = ["LOW", "MED", "HIGH"]
    for i, (w, c) in enumerate(zip(weights, counts)):
        print(f"  {names[i]}: count={c:,}  weight={w:.2f}")

    return weight_tensor


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader), f1


def evaluate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item()
            probs  = F.softmax(logits, dim=1)
            preds  = torch.argmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return (total_loss / len(loader), f1,
            np.array(all_preds), np.array(all_labels),
            np.array(all_probs))


def train_full(model, train_loader, val_loader,
               class_weights, n_epochs=40, lr=5e-4):
    """
    Full training loop.
    Key differences from original:
    - Stronger class weights (power=2)
    - Lower learning rate (5e-4 instead of 1e-3) for stability
    - More epochs (40 instead of 30)
    - Patience of 10 epochs
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1":   [], "val_f1":   []
    }

    best_val_f1    = 0.0
    best_epoch     = 0
    patience_count = 0
    patience       = 10

    print(f"\n{'Epoch':>6} | {'TrLoss':>8} | {'VaLoss':>8} | "
          f"{'Tr F1':>7} | {'Va F1':>7} | {'LR':>9}")
    print("-" * 58)

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_f1 = train_epoch(
            model, train_loader, criterion, optimizer
        )
        va_loss, va_f1, _, _, _ = evaluate_epoch(
            model, val_loader, criterion
        )
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)

        marker = ""
        if va_f1 > best_val_f1:
            best_val_f1    = va_f1
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_f1':      va_f1,
                'history':     history,
            }, MODELS_DIR / "best_detector.pt")
            marker = " ← best"
        else:
            patience_count += 1

        print(f"{epoch:>6} | {tr_loss:>8.4f} | {va_loss:>8.4f} | "
              f"{tr_f1:>7.4f} | {va_f1:>7.4f} | {lr_now:>9.7f}"
              f"{marker}")

        if patience_count >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\n✅ Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    return history


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: THRESHOLD TUNING
# ═════════════════════════════════════════════════════════════════════════════

def find_best_threshold(model, val_loader, criterion):
    """
    Default threshold is 0.5 — predict whichever class has highest prob.
    But with imbalanced data, HIGH risk needs a LOWER threshold.
    We scan thresholds 0.1 to 0.9 and pick the one that maximises HIGH F1.

    Example: if threshold=0.3, we predict HIGH whenever
             P(HIGH) > 0.3, not just when P(HIGH) > 0.5
    This catches more HIGH risk events at cost of some false alarms.
    """
    print("\nTuning decision threshold for HIGH risk class...")

    _, _, _, labels, probs = evaluate_epoch(
        model, val_loader, criterion
    )

    high_probs = probs[:, 2]   # probability of HIGH class

    best_threshold = 0.5
    best_high_f1   = 0.0

    thresholds = np.arange(0.1, 0.8, 0.05)
    results = []

    for thresh in thresholds:
        # Apply threshold: if P(HIGH) > thresh, predict HIGH
        preds_thresh = np.argmax(probs, axis=1).copy()
        preds_thresh[high_probs > thresh] = 2

        f1_scores = f1_score(labels, preds_thresh, average=None,
                             zero_division=0)
        high_f1   = f1_scores[2] if len(f1_scores) > 2 else 0
        macro_f1  = f1_score(labels, preds_thresh, average='macro',
                             zero_division=0)

        results.append({
            "threshold": thresh,
            "high_f1":   high_f1,
            "macro_f1":  macro_f1
        })

        if high_f1 > best_high_f1:
            best_high_f1   = high_f1
            best_threshold = thresh

    print(f"\n  Threshold scan results:")
    print(f"  {'Thresh':>8} | {'HIGH F1':>8} | {'Macro F1':>9}")
    print(f"  {'-'*32}")
    for r in results:
        marker = " ← best" if r["threshold"] == best_threshold else ""
        print(f"  {r['threshold']:>8.2f} | "
              f"{r['high_f1']:>8.4f} | "
              f"{r['macro_f1']:>9.4f}{marker}")

    print(f"\n  Best threshold: {best_threshold:.2f}")
    print(f"  Best HIGH F1:   {best_high_f1:.4f}")

    return best_threshold, best_high_f1


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: FINAL EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def final_evaluation(model, test_loader, criterion, threshold):
    """
    Evaluate on held-out test set using the tuned threshold.
    Generate all metrics for your paper.
    """
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)

    # Load best checkpoint
    ckpt = torch.load(MODELS_DIR / "best_detector.pt",
                      map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])

    _, _, preds_default, labels, probs = evaluate_epoch(
        model, test_loader, criterion
    )

    # Apply tuned threshold
    high_probs    = probs[:, 2]
    preds_tuned   = preds_default.copy()
    preds_tuned[high_probs > threshold] = 2

    class_names = ["LOW", "MED", "HIGH"]

    print(f"\nResults with DEFAULT threshold (0.50):")
    print(classification_report(
        labels, preds_default,
        target_names=class_names, digits=4
    ))

    print(f"\nResults with TUNED threshold ({threshold:.2f}):")
    print(classification_report(
        labels, preds_tuned,
        target_names=class_names, digits=4
    ))

    macro_f1 = f1_score(labels, preds_tuned, average='macro',
                        zero_division=0)
    f1_each  = f1_score(labels, preds_tuned, average=None,
                        zero_division=0)
    high_f1  = f1_each[2] if len(f1_each) > 2 else 0

    print(f"Macro F1 (tuned):  {macro_f1:.4f}")
    print(f"HIGH F1  (tuned):  {high_f1:.4f}")

    return preds_tuned, labels, probs, macro_f1, high_f1


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_all_results(history, preds, labels, probs):
    """Generate 4 publication-ready result plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "ORBITGUARD — CNN+LSTM Detection Model Results (Fixed)",
        fontsize=13, fontweight='bold'
    )
    class_names = ["LOW", "MED", "HIGH"]
    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], 'b-o',
            markersize=3, label='Train Loss')
    ax.plot(epochs, history["val_loss"], 'r-o',
            markersize=3, label='Val Loss')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── F1 ────────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, history["train_f1"], 'b-o',
            markersize=3, label='Train F1')
    ax.plot(epochs, history["val_f1"], 'r-o',
            markersize=3, label='Val F1')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro F1 Score")
    ax.set_title("Training & Validation F1 Score")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

    # ── Confusion matrix ──────────────────────────────────────────────────
    ax = axes[1, 0]
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalised)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{cm_norm[i,j]:.2f}',
                    ha='center', va='center', fontsize=11,
                    color='white' if cm_norm[i,j] > 0.5 else 'black')

    # ── Per-class metrics bar chart ───────────────────────────────────────
    ax = axes[1, 1]
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(labels, preds, average=None,
                                zero_division=0)
    recall    = recall_score(labels, preds, average=None,
                             zero_division=0)
    f1_each   = f1_score(labels, preds, average=None,
                         zero_division=0)
    x     = np.arange(3)
    width = 0.25
    ax.bar(x - width, precision, width,
           label='Precision', color='#3182CE', alpha=0.85)
    ax.bar(x,          recall,   width,
           label='Recall',    color='#E53E3E', alpha=0.85)
    ax.bar(x + width, f1_each,  width,
           label='F1',        color='#38A169', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 per Class")
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.15)
    for i, (p, r, f) in enumerate(zip(precision, recall, f1_each)):
        ax.text(i - width, p + 0.02, f'{p:.2f}',
                ha='center', fontsize=8)
        ax.text(i,          r + 0.02, f'{r:.2f}',
                ha='center', fontsize=8)
        ax.text(i + width, f + 0.02, f'{f:.2f}',
                ha='center', fontsize=8)

    plt.tight_layout()
    out = OUTPUTS_DIR / "detector_results_fixed.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {out}")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("="*60)
    print("ORBITGUARD — CNN+LSTM Detection Model (FIXED)")
    print("="*60)

    # ── Step 1: Load data + SMOTE ─────────────────────────────────────────
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     seq_len, n_feat) = load_and_oversample()

    # ── Step 2: DataLoaders ───────────────────────────────────────────────
    train_ds = DebrisDataset(X_train, y_train)
    val_ds   = DebrisDataset(X_val,   y_val)
    test_ds  = DebrisDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=32,
                              shuffle=False, num_workers=0)

    print(f"\nDataLoaders ready:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # ── Step 3: Build model ───────────────────────────────────────────────
    print(f"\nBuilding model...")
    model = DebrisDetector(
        n_features=n_feat,
        seq_length=seq_len,
        cnn_channels=64,
        lstm_hidden=128,
        lstm_layers=2,
        n_classes=3,
        dropout=0.3
    ).to(DEVICE)

    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device:     {DEVICE}")

    # Sanity check
    dummy  = torch.randn(4, seq_len, n_feat).to(DEVICE)
    output = model(dummy)
    assert output.shape == (4, 3), "Wrong output shape!"
    print(f"  ✅ Forward pass OK — output shape: {output.shape}")

    # ── Step 4: Compute strong class weights ──────────────────────────────
    class_weights = compute_class_weights(
        y_train, DEVICE, power=2.0
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Step 5: Train ─────────────────────────────────────────────────────
    history = train_full(
        model, train_loader, val_loader,
        class_weights,
        n_epochs=40,
        lr=5e-4
    )

    # ── Step 6: Find best threshold ───────────────────────────────────────
    best_threshold, best_high_f1 = find_best_threshold(
        model, val_loader, criterion
    )

    # ── Step 7: Final test evaluation ─────────────────────────────────────
    preds, labels, probs, macro_f1, high_f1 = final_evaluation(
        model, test_loader, criterion, best_threshold
    )

    # ── Step 8: Plot results ──────────────────────────────────────────────
    plot_all_results(history, preds, labels, probs)

    # ── Step 9: Save metrics ──────────────────────────────────────────────
    metrics = {
        "macro_f1":        round(float(macro_f1), 4),
        "high_risk_f1":    round(float(high_f1), 4),
        "best_threshold":  round(float(best_threshold), 2),
        "n_train_after_smote": int(len(y_train)),
        "n_test_samples":  int(len(y_test)),
        "n_parameters":    model.count_parameters(),
        "seq_length":      int(seq_len),
        "n_features":      int(n_feat),
    }
    with open(MODELS_DIR / "detector_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"STEP 5 (FIXED) COMPLETE")
    print(f"{'='*60}")
    print(f"  Macro F1:       {macro_f1:.4f}")
    print(f"  HIGH Risk F1:   {high_f1:.4f}")
    print(f"  Threshold used: {best_threshold:.2f}")
    print(f"  Model saved:    models/best_detector.pt")
    print(f"  Plot saved:     outputs/detector_results_fixed.png")
    print(f"  Metrics saved:  models/detector_metrics.json")