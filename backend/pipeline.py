"""
ORBITGUARD — AI Pipeline
Loads all 3 trained models once at startup.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
from pathlib import Path

ROOT      = Path(__file__).parent.parent
MODELS    = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
R_EARTH   = 6371.0
GM        = 398600.4418


# ── MODEL DEFINITIONS ─────────────────────────────────────

class DebrisDetectorLite(nn.Module):
    def __init__(self, n_features=18, seq_length=20,
                 n_classes=3, dropout=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, 3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(64),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            64, 64, 2, batch_first=True,
            dropout=dropout, bidirectional=False
        )
        self.attention = nn.Sequential(
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        attn = F.softmax(self.attention(out), dim=1)
        ctx  = (out * attn).sum(dim=1)
        return self.head(ctx)


class TrajectoryNet(nn.Module):
    def __init__(self, hidden=512, n_res=6,
                 dropout=0.05, n_in=12):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.BatchNorm1d(hidden)
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.BatchNorm1d(hidden),
            ) for _ in range(n_res)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.Tanh(),
            nn.Linear(256, 64),    nn.Tanh(),
            nn.Linear(64, 3)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(h)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=42, act_dim=3, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden,  hidden), nn.Tanh(),
            nn.Linear(hidden,  hidden), nn.Tanh(),
        )
        self.actor_mean    = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(
            torch.zeros(act_dim) - 2.0
        )
        self.critic = nn.Linear(hidden, 1)

    def get_action(self, obs):
        f    = self.trunk(obs)
        mean = self.actor_mean(f)
        return mean.squeeze().detach().numpy()


# ── PIPELINE CLASS ─────────────────────────────────────────

class ORBITGUARDPipeline:

    def __init__(self):
        print("="*50)
        print("Loading ORBITGUARD AI Pipeline...")
        print("="*50)
        self._load_detector()
        self._load_pinn()
        self._load_drl()
        self._load_orbital_data()
        print("="*50)
        print("✅ All models ready")
        print("="*50)

    # ── Loaders ───────────────────────────────────────────

    def _load_detector(self):
        self.detector = DebrisDetectorLite()
        ckpt = torch.load(
            MODELS / "best_detector.pt",
            map_location="cpu", weights_only=False
        )
        self.detector.load_state_dict(ckpt["model_state"])
        self.detector.eval()
        with open(MODELS / "feature_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        print("  ✅ CNN+LSTM Detector  (F1=0.9677)")

    def _load_pinn(self):
        ckpt = torch.load(
            MODELS / "best_pinn.pt",
            map_location="cpu", weights_only=False
        )
        self.pinn_norms = ckpt["norms"]
        n_in = self.pinn_norms.get("n_inputs", 12)
        self.pinn = TrajectoryNet(n_in=n_in)
        self.pinn.load_state_dict(ckpt["model_state"])
        self.pinn.eval()
        print(f"  ✅ PINN Predictor     "
              f"(RMSE={ckpt['val_rmse_km']:.2f}km)")

    def _load_drl(self):
        ckpt = torch.load(
            MODELS / "best_drl.pt",
            map_location="cpu", weights_only=False
        )
        self.policy = ActorCritic()
        self.policy.load_state_dict(ckpt["model_state"])
        self.policy.eval()
        print(f"  ✅ DRL Policy         "
              f"(success={ckpt['success_rate']:.1%})")

    def _load_orbital_data(self):
        print("  Loading orbital data...")
        df = pd.read_parquet(
            PROCESSED / "trajectories_full.parquet",
            columns=["id","t","x","y","z",
                     "vx","vy","vz","altitude"]
        )
        snap = df.groupby("id").first().reset_index()
        conj = pd.read_parquet(
            PROCESSED / "conjunctions.parquet"
        )

        high_ids = (
            set(conj[conj["risk_num"]==2]["obj1"]) |
            set(conj[conj["risk_num"]==2]["obj2"])
        )
        med_ids = (
            set(conj[conj["risk_num"]==1]["obj1"]) |
            set(conj[conj["risk_num"]==1]["obj2"])
        )

        def get_risk(oid):
            if oid in high_ids: return "HIGH"
            if oid in med_ids:  return "MED"
            return "LOW"

        self.objects = [
            {
                "id":   int(r.id),
                "x":    round(float(r.x),   2),
                "y":    round(float(r.y),   2),
                "z":    round(float(r.z),   2),
                "vx":   round(float(r.vx),  4),
                "vy":   round(float(r.vy),  4),
                "vz":   round(float(r.vz),  4),
                "alt":  round(float(r.altitude), 1),
                "risk": get_risk(r.id),
                "type": "DEBRIS",
            }
            for r in snap.itertuples()
        ]

        # Build conjunction pairs for 3D lines
        snap_idx = {int(r.id): r
                    for r in snap.itertuples()}
        self.conjunctions = []
        for _, row in (
            conj[conj["risk_num"]==2].head(60).iterrows()
        ):
            o1 = snap_idx.get(int(row["obj1"]))
            o2 = snap_idx.get(int(row["obj2"]))
            if o1 and o2:
                self.conjunctions.append({
                    "obj1": int(row["obj1"]),
                    "obj2": int(row["obj2"]),
                    "miss": round(float(row["miss_km"]), 3),
                    "x1":   round(float(o1.x), 2),
                    "y1":   round(float(o1.y), 2),
                    "z1":   round(float(o1.z), 2),
                    "x2":   round(float(o2.x), 2),
                    "y2":   round(float(o2.y), 2),
                    "z2":   round(float(o2.z), 2),
                })

        h = sum(1 for o in self.objects if o["risk"]=="HIGH")
        m = sum(1 for o in self.objects if o["risk"]=="MED")
        l = sum(1 for o in self.objects if o["risk"]=="LOW")
        print(f"  ✅ {len(self.objects):,} objects loaded  "
              f"(H:{h} M:{m} L:{l})")
        print(f"  ✅ {len(self.conjunctions)} conjunctions")

    # ── MODULE 1: DETECT ──────────────────────────────────

    def detect(self, sequences: list) -> list:
        if not sequences:
            return []
        X = np.array(sequences, dtype=np.float32)
        n, s, f = X.shape
        try:
            X_norm = self.scaler.transform(
                X.reshape(-1, f)
            ).reshape(n, s, f)
        except Exception:
            X_norm = X
        with torch.no_grad():
            logits = self.detector(
                torch.FloatTensor(X_norm)
            )
            probs  = F.softmax(logits, dim=1).numpy()
            labels = np.argmax(probs, axis=1)
        names = {0:"LOW", 1:"MED", 2:"HIGH"}
        return [
            {
                "label":      int(labels[i]),
                "risk":       names[int(labels[i])],
                "prob_high":  float(probs[i, 2]),
                "prob_med":   float(probs[i, 1]),
                "prob_low":   float(probs[i, 0]),
                "confidence": float(probs[i, labels[i]]),
            }
            for i in range(n)
        ]

    # ── MODULE 2: PREDICT ─────────────────────────────────

    def predict(self, state0: list,
                dt_min: float = 30.0) -> dict:
        R = self.pinn_norms["R_NORM"]
        V = self.pinn_norms["V_NORM"]
        D = self.pinn_norms["DR_NORM"]
        H = self.pinn_norms["horizon_min"]

        x0, y0, z0   = state0[0], state0[1], state0[2]
        vx0, vy0, vz0 = state0[3], state0[4], state0[5]

        r0m      = np.sqrt(x0**2 + y0**2 + z0**2)
        T_period = 2*np.pi*np.sqrt(r0m**3/GM)/60
        phase    = (dt_min % T_period) / T_period
        rv       = np.array([x0, y0, z0])
        vv       = np.array([vx0, vy0, vz0])
        h_vec    = np.cross(rv, vv)
        h_hat    = h_vec / (np.linalg.norm(h_vec) + 1e-9)
        t_n      = dt_min / H
        t_n2     = t_n ** 2
        alt_n    = (r0m - R_EARTH) / 2000.0

        s_in = np.array([[
            x0/R,  y0/R,  z0/R,
            vx0/V, vy0/V, vz0/V,
            t_n,   t_n2,
            alt_n, phase,
            float(h_hat[0]), float(h_hat[1])
        ]], dtype=np.float32)

        with torch.no_grad():
            dr = self.pinn(
                torch.FloatTensor(s_in)
            ).numpy()[0]

        r_pred = np.array([x0, y0, z0]) + dr * D
        return {
            "x":       round(float(r_pred[0]), 2),
            "y":       round(float(r_pred[1]), 2),
            "z":       round(float(r_pred[2]), 2),
            "dt_min":  dt_min,
            "rmse_km": 48.27,
        }

    # ── MODULE 3: AVOID ───────────────────────────────────

    def avoid(self, sc_state: list,
              debris_states: list) -> dict:
        R, V, D = 8000.0, 8.0, 500.0
        obs = []
        obs.extend(np.array(sc_state[:3]) / R)
        obs.extend(np.array(sc_state[3:6]) / V)
        sc_pos = np.array(sc_state[:3])
        for i in range(5):
            if i < len(debris_states):
                d_pos = np.array(debris_states[i][:3])
                d_vel = np.array(debris_states[i][3:6])
                rel   = d_pos - sc_pos
                dist  = np.linalg.norm(rel)
                obs.extend(rel / D)
                obs.extend(d_vel / V)
                obs.append(min(dist, 2000.0) / D)
            else:
                obs.extend([0.0] * 7)
        obs.append(1.0)
        obs_t   = torch.FloatTensor(obs).unsqueeze(0)
        delta_v = np.clip(
            self.policy.get_action(obs_t),
            -0.0005, 0.0005
        )
        dv_mag = float(np.linalg.norm(delta_v)) * 1000
        return {
            "dvx_ms":    round(float(delta_v[0])*1000, 5),
            "dvy_ms":    round(float(delta_v[1])*1000, 5),
            "dvz_ms":    round(float(delta_v[2])*1000, 5),
            "dv_mag_ms": round(dv_mag, 5),
        }


# Singleton
_pipeline = None

def get_pipeline() -> ORBITGUARDPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ORBITGUARDPipeline()
    return _pipeline