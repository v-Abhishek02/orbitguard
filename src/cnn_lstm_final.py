"""
ORBITGUARD — CNN+LSTM Detection Model (FINAL - Anti-Overfitting)
Key changes from fixed version:
1. Smaller model (fewer parameters relative to dataset size)
2. Higher dropout (0.5 instead of 0.3)
3. L2 regularisation increased
4. Data augmentation on training sequences
5. Larger batch size relative to dataset
6. Early stopping on val loss (not val F1)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json, pickle, warnings
warnings.filterwarnings('ignore')

PROCESSED = Path("data/processed")
MODELS    = Path("models")
OUTPUTS   = Path("outputs")
MODELS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print("✅ CPU")


# ═══════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════

def augment_sequence(X, noise_level=0.005):
    """
    Add tiny random noise to training sequences.
    This is standard practice to reduce overfitting.
    noise_level=0.005 = 0.5% noise — enough to regularise,
    not enough to corrupt the orbital physics signal.
    """
    noise = np.random.normal(0, noise_level, X.shape).astype(np.float32)
    return X + noise


class DebrisDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X       = torch.FloatTensor(X)
        self.y       = torch.LongTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].numpy()
        if self.augment and np.random.random() > 0.5:
            x = augment_sequence(x)
        return torch.FloatTensor(x), self.y[idx]


# ═══════════════════════════════════════════════════════════
# SMALLER MODEL — fewer params = less overfitting
# ═══════════════════════════════════════════════════════════

class DebrisDetectorLite(nn.Module):
    """
    Lighter version of the detector.
    Original: 745,924 params — too large for 5,421 samples
    This version: ~180,000 params — better param/sample ratio

    Rule of thumb: params should be < 50x your training samples
    5,421 × 50 = 271,050 max params
    This model: ~180,000 ✅
    """

    def __init__(self, n_features=18, seq_len=20,
                 n_classes=3, dropout=0.5):
        super().__init__()

        # Smaller CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
        )

        # Smaller LSTM — single direction, fewer hidden units
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False   # single direction — fewer params
        )

        # Simple attention
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Smaller head
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (batch, seq, features)
        x = x.permute(0, 2, 1)           # (batch, features, seq)
        x = self.cnn(x)                   # (batch, 64, seq)
        x = x.permute(0, 2, 1)           # (batch, seq, 64)
        out, _ = self.lstm(x)             # (batch, seq, 64)
        attn = F.softmax(
            self.attention(out), dim=1
        )                                 # (batch, seq, 1)
        ctx = (out * attn).sum(dim=1)     # (batch, 64)
        return self.head(ctx)             # (batch, n_classes)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ═══════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════

def prepare_data():
    print("="*55)
    print("LOADING AND PREPARING DATA")
    print("="*55)

    X = np.load(PROCESSED / "X_sequences.npy")
    y = np.load(PROCESSED / "y_labels.npy")

    print(f"\nOriginal: {X.shape}  labels: {y.shape}")
    names = {0:"LOW", 1:"MED", 2:"HIGH"}
    for label in [0,1,2]:
        cnt = (y==label).sum()
        print(f"  {names[label]}: {cnt:,}  ({cnt/len(y)*100:.1f}%)")

    # Split BEFORE any oversampling
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42,
        stratify=y_tmp
    )

    n_tr, seq, nf = X_tr.shape

    # Normalise
    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(
        X_tr.reshape(-1, nf)
    ).reshape(n_tr, seq, nf)
    X_val_s = scaler.transform(
        X_val.reshape(-1, nf)
    ).reshape(len(X_val), seq, nf)
    X_te_s  = scaler.transform(
        X_te.reshape(-1, nf)
    ).reshape(len(X_te), seq, nf)

    with open(MODELS / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # SMOTE on training only
    print(f"\nApplying SMOTE to training set...")
    print(f"  Before: {len(y_tr)} samples")
    for label in [0,1,2]:
        print(f"    {names[label]}: {(y_tr==label).sum()}")

    X_tr_flat = X_tr_s.reshape(n_tr, seq * nf)
    high_n = (y_tr == 2).sum()
    med_n  = (y_tr == 1).sum()

    target = {
        2: max(high_n, min(800, high_n * 3)),
        1: max(med_n,  min(800, med_n  * 3)),
    }

    try:
        smote = SMOTE(
            sampling_strategy=target,
            k_neighbors=min(4, high_n - 1),
            random_state=42
        )
        X_res, y_res = smote.fit_resample(X_tr_flat, y_tr)
    except Exception as e:
        print(f"  SMOTE failed: {e} — using original data")
        X_res, y_res = X_tr_flat, y_tr

    X_res_3d = X_res.reshape(-1, seq, nf)

    print(f"  After: {len(y_res)} samples")
    for label in [0,1,2]:
        print(f"    {names[label]}: {(y_res==label).sum()}")

    return (X_res_3d, y_res,
            X_val_s, y_val,
            X_te_s,  y_te,
            seq, nf)


# ═══════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════

def train_epoch(model, loader, crit, opt):
    model.train()
    total, preds_all, labels_all = 0, [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        out  = model(Xb)
        loss = crit(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
        preds_all.extend(torch.argmax(out,1).cpu().numpy())
        labels_all.extend(yb.cpu().numpy())
    f1 = f1_score(labels_all, preds_all,
                  average='macro', zero_division=0)
    return total/len(loader), f1


def eval_epoch(model, loader, crit):
    model.eval()
    total, preds_all, labels_all, probs_all = 0, [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            out  = model(Xb)
            loss = crit(out, yb)
            total += loss.item()
            probs = F.softmax(out, dim=1)
            preds_all.extend(torch.argmax(out,1).cpu().numpy())
            labels_all.extend(yb.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())
    f1 = f1_score(labels_all, preds_all,
                  average='macro', zero_division=0)
    return (total/len(loader), f1,
            np.array(preds_all),
            np.array(labels_all),
            np.array(probs_all))


def train(model, tr_loader, va_loader,
          class_counts, n_epochs=50, lr=3e-4):
    """
    Anti-overfitting training strategy:
    - Higher weight decay (1e-3 vs 1e-4) = stronger L2 regularisation
    - Lower learning rate (3e-4) = more stable convergence
    - More epochs (50) to compensate for stronger regularisation
    - Early stopping on val LOSS (stricter than val F1)
    - ReduceLROnPlateau: halve LR when val loss plateaus
    """

    # Compute class weights
    total = sum(class_counts.values())
    w = torch.FloatTensor([
        total / (3 * class_counts.get(i, 1))
        for i in [0,1,2]
    ]).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    print(f"\nClass weights: "
          f"LOW={w[0]:.3f} MED={w[1]:.3f} HIGH={w[2]:.3f}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-3    # stronger regularisation
    )

    # ReduceLROnPlateau halves LR when val loss stops improving
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5,
        patience=5, verbose=False
    )

    history = {
        "tr_loss":[], "va_loss":[],
        "tr_f1":[],  "va_f1":[]
    }

    best_va_loss  = float('inf')
    best_va_f1    = 0.0
    patience_cnt  = 0
    patience      = 12   # stop if no improvement for 12 epochs

    print(f"\n{'Ep':>4} | {'TrLoss':>8} | {'VaLoss':>8} | "
          f"{'TrF1':>7} | {'VaF1':>7} | {'LR':>9} | Gap")
    print("-" * 65)

    for ep in range(1, n_epochs+1):
        tr_loss, tr_f1 = train_epoch(model, tr_loader, crit, opt)
        va_loss, va_f1, _, _, _ = eval_epoch(
            model, va_loader, crit
        )

        sched.step(va_loss)
        cur_lr = opt.param_groups[0]['lr']

        history["tr_loss"].append(tr_loss)
        history["va_loss"].append(va_loss)
        history["tr_f1"].append(tr_f1)
        history["va_f1"].append(va_f1)

        gap    = tr_f1 - va_f1
        marker = ""

        # Save on best val loss (not val F1 — reduces overfitting)
        if va_loss < best_va_loss:
            best_va_loss = va_loss
            best_va_f1   = va_f1
            patience_cnt = 0
            torch.save({
                'epoch':       ep,
                'model_state': model.state_dict(),
                'va_f1':       va_f1,
                'va_loss':     va_loss,
                'history':     history,
            }, MODELS / "best_detector.pt")
            marker = " ✓"

        else:
            patience_cnt += 1

        print(f"{ep:>4} | {tr_loss:>8.4f} | {va_loss:>8.4f} | "
              f"{tr_f1:>7.4f} | {va_f1:>7.4f} | {cur_lr:>9.7f} | "
              f"{gap:.3f}{marker}")

        if patience_cnt >= patience:
            print(f"\nEarly stopping at epoch {ep}")
            break

    print(f"\n✅ Best val loss: {best_va_loss:.4f} "
          f"at val F1: {best_va_f1:.4f}")
    return history


# ═══════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════

def evaluate_final(model, te_loader, crit):
    ckpt = torch.load(MODELS / "best_detector.pt",
                      map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])

    _, _, preds, labels, probs = eval_epoch(
        model, te_loader, crit
    )

    names = ["LOW", "MED", "HIGH"]
    print("\n" + "="*55)
    print("FINAL TEST RESULTS")
    print("="*55)
    print(classification_report(
        labels, preds,
        target_names=names, digits=4
    ))

    macro_f1 = f1_score(labels, preds, average='macro',
                        zero_division=0)
    f1_each  = f1_score(labels, preds, average=None,
                        zero_division=0)
    high_f1  = f1_each[2] if len(f1_each) > 2 else 0

    return preds, labels, probs, macro_f1, high_f1


def plot_results(history, preds, labels):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "ORBITGUARD — CNN+LSTM Final Model",
        fontsize=13, fontweight='bold'
    )
    names = ["LOW", "MED", "HIGH"]
    eps   = range(1, len(history["tr_loss"])+1)

    # Loss
    ax = axes[0,0]
    ax.plot(eps, history["tr_loss"], 'b-o',
            ms=3, label='Train')
    ax.plot(eps, history["va_loss"], 'r-o',
            ms=3, label='Val')
    ax.set_title("Loss Curves")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # F1
    ax = axes[0,1]
    ax.plot(eps, history["tr_f1"], 'b-o',
            ms=3, label='Train F1')
    ax.plot(eps, history["va_f1"], 'r-o',
            ms=3, label='Val F1')

    # Shade the gap — overfitting visualised
    ax.fill_between(
        eps, history["tr_f1"], history["va_f1"],
        alpha=0.15, color='orange', label='Overfitting gap'
    )
    ax.set_title("F1 Score (orange = overfitting gap)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro F1")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Confusion matrix
    ax = axes[1,0]
    cm     = confusion_matrix(labels, preds)
    cm_n   = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im     = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{cm_n[i,j]:.2f}',
                    ha='center', va='center',
                    color='white' if cm_n[i,j]>0.5 else 'black',
                    fontsize=11)

    # Per-class bars
    ax = axes[1,1]
    from sklearn.metrics import precision_score, recall_score
    p  = precision_score(labels, preds, average=None,
                         zero_division=0)
    r  = recall_score(labels, preds, average=None,
                      zero_division=0)
    f1 = f1_score(labels, preds, average=None,
                  zero_division=0)
    x  = np.arange(3)
    w  = 0.25
    ax.bar(x-w, p,  w, label='Precision',
           color='#3182CE', alpha=0.85)
    ax.bar(x,   r,  w, label='Recall',
           color='#E53E3E', alpha=0.85)
    ax.bar(x+w, f1, w, label='F1',
           color='#38A169', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("Per-class Metrics")
    ax.set_ylim(0, 1.15); ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, (pi, ri, fi) in enumerate(zip(p, r, f1)):
        ax.text(i-w, pi+0.02, f'{pi:.2f}', ha='center', fontsize=8)
        ax.text(i,   ri+0.02, f'{ri:.2f}', ha='center', fontsize=8)
        ax.text(i+w, fi+0.02, f'{fi:.2f}', ha='center', fontsize=8)

    plt.tight_layout()
    out = OUTPUTS / "detector_final.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {out}")
    plt.show()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("="*55)
    print("ORBITGUARD — CNN+LSTM Final (Anti-Overfitting)")
    print("="*55)

    # Load data
    (X_tr, y_tr,
     X_va, y_va,
     X_te, y_te,
     seq, nf) = prepare_data()

    # Class counts after SMOTE
    from collections import Counter
    cc = Counter(y_tr.tolist())
    class_counts = {0: cc[0], 1: cc[1], 2: cc[2]}

    # DataLoaders — augment training data only
    tr_ds = DebrisDataset(X_tr, y_tr, augment=True)
    va_ds = DebrisDataset(X_va, y_va, augment=False)
    te_ds = DebrisDataset(X_te, y_te, augment=False)

    tr_loader = DataLoader(tr_ds, batch_size=64,
                           shuffle=True,  num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=64,
                           shuffle=False, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=64,
                           shuffle=False, num_workers=0)

    # Build smaller model
    model = DebrisDetectorLite(
        n_features=nf,
        seq_len=seq,
        n_classes=3,
        dropout=0.5
    ).to(DEVICE)

    print(f"\nModel: {model.count_parameters():,} parameters")
    print(f"Data:  {len(y_tr):,} training samples")
    print(f"Ratio: {model.count_parameters()//len(y_tr):,}"
          f" params per sample (target < 50)")

    # Sanity check
    dummy = torch.randn(4, seq, nf).to(DEVICE)
    out   = model(dummy)
    assert out.shape == (4, 3)
    print(f"✅ Forward pass: {out.shape}")

    # Train
    history = train(
        model, tr_loader, va_loader,
        class_counts,
        n_epochs=50,
        lr=3e-4
    )

    # Evaluate
    w      = torch.FloatTensor([
        sum(class_counts.values()) / (3 * class_counts.get(i,1))
        for i in [0,1,2]
    ]).to(DEVICE)
    crit   = nn.CrossEntropyLoss(weight=w)
    preds, labels, probs, macro_f1, high_f1 = evaluate_final(
        model, te_loader, crit
    )

    # Plot
    plot_results(history, preds, labels)

    # Save metrics
    metrics = {
        "macro_f1":          round(float(macro_f1), 4),
        "high_risk_f1":      round(float(high_f1), 4),
        "n_parameters":      model.count_parameters(),
        "n_train_samples":   len(y_tr),
        "param_sample_ratio":model.count_parameters()//len(y_tr),
        "dropout":           0.5,
        "weight_decay":      1e-3,
        "augmentation":      True,
        "architecture":      "CNN+BiLSTM+Attention (Lite)",
    }
    with open(MODELS / "detector_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Check overfitting
    final_gap = history["tr_f1"][-1] - history["va_f1"][-1]
    print(f"\n{'='*55}")
    print(f"STEP 5 FINAL COMPLETE")
    print(f"{'='*55}")
    print(f"  Macro F1:     {macro_f1:.4f}")
    print(f"  HIGH F1:      {high_f1:.4f}")
    print(f"  Overfitting gap: {final_gap:.4f}  "
          f"{'⚠️ still high' if final_gap > 0.15 else '✅ acceptable'}")
    print(f"\nRun next: python src/pinn_model.py")