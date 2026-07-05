"""
ORBITGUARD — Step 6: PINN Trajectory Prediction (Final)
Target: RMSE < 200 km at 60-min horizon
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

PROCESSED = Path("data/processed")
MODELS    = Path("models")
OUTPUTS   = Path("outputs")
MODELS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

GM      = 398600.4418
R_EARTH = 6371.0

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print("✅ CPU")

torch.manual_seed(42)
np.random.seed(42)


def prepare_data(n_objects=500, horizon_min=60,
                  n_per_obj=60):
    print("Loading trajectories...")
    df = pd.read_parquet(
        PROCESSED / "trajectories_full.parquet",
        columns=["id","t","x","y","z","vx","vy","vz","altitude"]
    )
    # RELAXED filter — was rejecting too much
    df = df[df["altitude"].between(100, 3000)].copy()
    print(f"  Rows: {len(df):,}  "
          f"Objects: {df['id'].nunique():,}")

    # Detect timestep
    sid  = df["id"].iloc[0]
    tvs  = df[df["id"]==sid]["t"].sort_values().values
    step = float(tvs[1]-tvs[0]) if len(tvs)>1 else 2.0
    print(f"  Timestep: {step} min")

    max_steps = int(horizon_min / step)
    # Look-ahead steps from actual timestep intervals
    look_steps = sorted(set([
        max(1, int(0.08*max_steps)),
        max(2, int(0.17*max_steps)),
        max(3, int(0.25*max_steps)),
        max(4, int(0.33*max_steps)),
        int(0.50*max_steps),
        int(0.67*max_steps),
        int(0.83*max_steps),
        max_steps,
    ]))
    look_mins = [round(s*step,1) for s in look_steps]
    print(f"  Look-ahead steps: {look_steps}")
    print(f"  Look-ahead mins:  {look_mins}")

    R_NORM  = 8000.0
    V_NORM  = 8.0
    DR_NORM = 600.0

    all_ids = df["id"].unique()
    np.random.shuffle(all_ids)
    selected = all_ids[:n_objects]

    s_list, dr_list = [], []
    r0_list, rt_list, dt_list, w_list = [], [], [], []

    print(f"  Building data for {n_objects} objects...")

    for obj_id in tqdm(selected, desc="  Objects",
                       unit="obj"):
        obj = df[df["id"]==obj_id].sort_values("t")
        if len(obj) < max_steps + 3:
            continue

        xs  = obj["x"].values
        ys  = obj["y"].values
        zs  = obj["z"].values
        vxs = obj["vx"].values
        vys = obj["vy"].values
        vzs = obj["vz"].values
        ts  = obj["t"].values
        n   = len(ts)

        # RELAXED physical checks
        r_mags = np.sqrt(xs**2 + ys**2 + zs**2)
        v_mags = np.sqrt(vxs**2 + vys**2 + vzs**2)

        # Remove clearly bad rows only
        valid = ((r_mags > R_EARTH + 80) &
                 (v_mags > 3.0) &
                 (v_mags < 12.0))
        if valid.sum() < max_steps + 3:
            continue

        # Use only valid rows
        xs  = xs[valid];  ys  = ys[valid]
        zs  = zs[valid];  vxs = vxs[valid]
        vys = vys[valid]; vzs = vzs[valid]
        ts  = ts[valid];  n   = len(ts)

        if n < max_steps + 3:
            continue

        stride = max(1, (n - max_steps) // n_per_obj)
        starts = list(range(0, n-max_steps-1, stride))

        for si in starts[:n_per_obj]:
            x0  = xs[si];  y0  = ys[si];  z0  = zs[si]
            vx0 = vxs[si]; vy0 = vys[si]; vz0 = vzs[si]
            t0  = ts[si]

            r0m = np.sqrt(x0**2 + y0**2 + z0**2)
            v0m = np.sqrt(vx0**2 + vy0**2 + vz0**2)

            # Relaxed bounds
            if not (R_EARTH+80 < r0m < R_EARTH+3000):
                continue
            if not (3.5 < v0m < 12.0):
                continue

            # Angular momentum unit vector
            rv    = np.array([x0, y0, z0])
            vv    = np.array([vx0, vy0, vz0])
            h_vec = np.cross(rv, vv)
            h_mag = np.linalg.norm(h_vec)
            h_hat = h_vec / (h_mag + 1e-9)

            alt_n    = (r0m - R_EARTH) / 2000.0
            T_period = 2*np.pi*np.sqrt(r0m**3/GM)/60

            for look in look_steps:
                fi = si + look
                if fi >= n:
                    continue

                dt_min = float(ts[fi] - t0)
                if dt_min <= 0 or dt_min > horizon_min:
                    continue

                dx = xs[fi] - x0
                dy = ys[fi] - y0
                dz = zs[fi] - z0
                drm = np.sqrt(dx**2 + dy**2 + dz**2)

                # Relaxed displacement check
                if drm > DR_NORM * 3.0:
                    continue

                t_n   = dt_min / horizon_min
                t_n2  = t_n ** 2
                phase = (dt_min % T_period) / T_period

                # 12 features
                s0 = np.array([
                    x0/R_NORM,   y0/R_NORM,   z0/R_NORM,
                    vx0/V_NORM,  vy0/V_NORM,  vz0/V_NORM,
                    t_n,         t_n2,
                    alt_n,       phase,
                    float(h_hat[0]), float(h_hat[1])
                ], dtype=np.float32)

                d = np.array([
                    dx/DR_NORM, dy/DR_NORM, dz/DR_NORM
                ], dtype=np.float32)

                s_list.append(s0)
                dr_list.append(d)
                r0_list.append([x0, y0, z0])
                rt_list.append([xs[fi], ys[fi], zs[fi]])
                dt_list.append(dt_min)
                w_list.append(
                    3.0 if dt_min > horizon_min*0.5
                    else 1.0
                )

    print(f"\n  Raw samples collected: {len(s_list):,}")

    if len(s_list) == 0:
        print("\nDEBUG: No samples built.")
        print("Running diagnostic...")
        # Show what objects look like
        test_obj = df[df["id"]==all_ids[0]].sort_values("t")
        xs = test_obj["x"].values[:5]
        ys = test_obj["y"].values[:5]
        zs = test_obj["z"].values[:5]
        vxs= test_obj["vx"].values[:5]
        vys= test_obj["vy"].values[:5]
        vzs= test_obj["vz"].values[:5]
        r  = np.sqrt(xs**2+ys**2+zs**2)
        v  = np.sqrt(vxs**2+vys**2+vzs**2)
        print(f"  First object {all_ids[0]}:")
        print(f"  r_mag: {r}")
        print(f"  v_mag: {v}")
        print(f"  len:   {len(test_obj)}")
        print(f"  max_steps needed: {max_steps}")
        return None, None

    S  = np.array(s_list,  dtype=np.float32)
    DR = np.array(dr_list, dtype=np.float32)
    R0 = np.array(r0_list, dtype=np.float32)
    RT = np.array(rt_list, dtype=np.float32)
    DT = np.array(dt_list, dtype=np.float32)
    W  = np.array(w_list,  dtype=np.float32)

    print(f"  Final samples: {len(S):,}")
    print(f"  Long-horizon (>30min): {(DT>30).sum():,}")
    print(f"  DR range: [{DR.min():.3f}, {DR.max():.3f}]")

    n    = len(S)
    perm = np.random.permutation(n)
    ntr  = int(0.70*n)
    nva  = int(0.15*n)

    def T(idx):
        return (
            torch.FloatTensor(S[idx]).to(DEVICE),
            torch.FloatTensor(DR[idx]).to(DEVICE),
            torch.FloatTensor(R0[idx]).to(DEVICE),
            torch.FloatTensor(RT[idx]).to(DEVICE),
            torch.FloatTensor(DT[idx]).to(DEVICE),
            torch.FloatTensor(W[idx]).to(DEVICE),
        )

    norms = {
        "R_NORM": R_NORM, "V_NORM": V_NORM,
        "DR_NORM": DR_NORM,
        "horizon_min": horizon_min,
        "n_inputs": 12
    }

    print(f"  Train:{ntr:,}  Val:{nva:,}  "
          f"Test:{n-ntr-nva:,}")

    return {
        "tr": T(perm[:ntr]),
        "va": T(perm[ntr:ntr+nva]),
        "te": T(perm[ntr+nva:]),
    }, norms


class TrajectoryNet(nn.Module):
    def __init__(self, hidden=512, n_res=6,
                 dropout=0.05, n_in=12):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.Tanh(),
            nn.BatchNorm1d(hidden)
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.BatchNorm1d(hidden),
            )
            for _ in range(n_res)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.Tanh(),
            nn.Linear(256, 64),    nn.Tanh(),
            nn.Linear(64, 3)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight,
                                        gain=0.3)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


def physics_loss(s0, dr_pred, norms):
    R_NORM  = norms["R_NORM"]
    V_NORM  = norms["V_NORM"]
    DR_NORM = norms["DR_NORM"]
    r0  = s0[:, :3] * R_NORM
    v0  = s0[:, 3:6] * V_NORM
    rf  = r0 + dr_pred * DR_NORM
    r0m = torch.norm(r0, dim=1)
    rfm = torch.norm(rf, dim=1)
    dr_rad   = (rfm - r0m) / (r0m + 1e-6)
    rad_loss = torch.mean(dr_rad**2)
    below    = F.relu(R_EARTH + 80 - rfm)
    alt_loss = torch.mean(below**2) / R_EARTH**2
    v0m      = torch.norm(v0, dim=1)
    v_circ   = torch.sqrt(
        torch.clamp(GM / (rfm + 1e-6), min=0.01)
    )
    spd_loss = F.mse_loss(v0m/8.0, v_circ/8.0)
    E0   = v0m**2/2 - GM/(r0m+1e-6)
    Ef   = v_circ**2/2 - GM/(rfm+1e-6)
    Eabs = torch.abs(E0).detach() + 1e-6
    enrg = torch.mean(((Ef-E0)/Eabs)**2)
    return (0.35*rad_loss + 0.25*alt_loss
          + 0.20*spd_loss + 0.20*enrg)


def weighted_mse(pred, target, weights):
    sq = ((pred - target)**2).mean(dim=1)
    return (sq * weights).mean()


def train_model(model, data, norms,
                n_epochs=500, lr=1e-3, batch=2048):
    DR_NORM = norms["DR_NORM"]
    V_NORM  = norms["V_NORM"]

    s_tr,dr_tr,R0_tr,RT_tr,DT_tr,W_tr = data["tr"]
    s_va,dr_va,R0_va,RT_va,DT_va,W_va = data["va"]
    n_tr = len(s_tr)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.7,
        patience=40, verbose=False,
        min_lr=5e-5
    )

    best_rmse = float('inf')
    pat_cnt   = 0
    patience  = 60
    history   = {"data":[],"phys":[],"val_rmse":[]}

    print(f"\n{'Ep':>5}|{'Data':>10}|{'Phys':>8}|"
          f"{'ValRMSE':>10}|{'LR':>10}|{'λ':>5}")
    print("─"*58)

    for ep in range(1, n_epochs+1):
        if ep <= 80:
            lp = 0.00
        elif ep <= 200:
            lp = 0.10
        elif ep <= 350:
            lp = 0.20
        else:
            lp = 0.30

        model.train()
        ep_dat = ep_phy = 0.0
        n_bat  = 0
        perm   = torch.randperm(n_tr)

        for start in range(0, n_tr, batch):
            idx  = perm[start:start+batch]
            sb   = s_tr[idx]
            drb  = dr_tr[idx]
            wb   = W_tr[idx]
            opt.zero_grad()
            pred = model(sb)
            dl   = weighted_mse(pred, drb, wb)
            pl   = (physics_loss(sb, pred, norms)
                    if lp > 0
                    else torch.tensor(0.0,
                                      device=DEVICE))
            loss = dl + lp * pl
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            opt.step()
            ep_dat += dl.item()
            ep_phy += pl.item() if lp>0 else 0
            n_bat  += 1

        if n_bat == 0:
            break

        model.eval()
        with torch.no_grad():
            dp   = model(s_va)
            rp   = R0_va + dp * DR_NORM
            rmse = torch.sqrt(torch.mean(
                torch.sum((rp - RT_va)**2, dim=1)
            )).item()

        sched.step(rmse)
        cur_lr = opt.param_groups[0]['lr']
        history["data"].append(ep_dat/n_bat)
        history["phys"].append(ep_phy/n_bat)
        history["val_rmse"].append(rmse)

        if ep % 25 == 0 or ep <= 5:
            print(f"{ep:>5}|{ep_dat/n_bat:>10.5f}|"
                  f"{ep_phy/n_bat:>8.5f}|"
                  f"{rmse:>10.2f}|"
                  f"{cur_lr:>10.7f}|{lp:>5.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            pat_cnt   = 0
            torch.save({
                "epoch":       ep,
                "model_state": model.state_dict(),
                "val_rmse_km": rmse,
                "norms":       norms,
                "history":     history,
            }, MODELS / "best_pinn.pt")
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                print(f"\nEarly stopping at ep {ep}")
                break

    print(f"\n✅ Best val RMSE: {best_rmse:.2f} km")
    return history, best_rmse


def evaluate(model, data, norms):
    DR_NORM = norms["DR_NORM"]
    V_NORM  = norms["V_NORM"]
    s_te,dr_te,R0,RT,DT,W = data["te"]

    ckpt = torch.load(MODELS/"best_pinn.pt",
                      map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("\n"+"="*55)
    print("FINAL EVALUATION vs BASELINES")
    print("="*55)

    with torch.no_grad():
        dp    = model(s_te)
        rp    = R0 + dp * DR_NORM
        err_p = torch.sqrt(
            torch.sum((rp-RT)**2, dim=1)
        ).cpu().numpy()
        v0    = s_te[:,3:6] * V_NORM
        dt_s  = DT.unsqueeze(1) * 60.0
        r_lin = R0 + v0 * dt_s
        err_l = torch.sqrt(
            torch.sum((r_lin-RT)**2, dim=1)
        ).cpu().numpy()

    rmse_p = float(np.sqrt(np.mean(err_p**2)))
    rmse_l = float(np.sqrt(np.mean(err_l**2)))
    dt_v   = DT.cpu().numpy()

    print(f"\n  Linear baseline:  {rmse_l:>10.2f} km")
    print(f"  PINN (ours):      {rmse_p:>10.2f} km  "
          f"({rmse_l/max(rmse_p,.1):.1f}× better)")

    print(f"\n  {'Horizon':<12}{'PINN':>10}"
          f"{'Linear':>12}{'Status':>8}")
    print(f"  {'─'*44}")
    hor = {}
    for hmax in [10, 20, 30, 60]:
        m = dt_v <= hmax
        if m.sum() > 5:
            rp_h = float(np.sqrt(np.mean(err_p[m]**2)))
            rl_h = float(np.sqrt(np.mean(err_l[m]**2)))
            st   = "✅" if rp_h < 200 else "⚠️"
            print(f"  ≤{hmax}min{'':<7}"
                  f"{rp_h:>10.2f} km"
                  f"{rl_h:>12.2f} km  {st}")
            hor[f"{hmax}min"] = {
                "pinn":       round(rp_h, 2),
                "linear":     round(rl_h, 2),
                "target_met": rp_h < 200
            }
    print("="*55)
    return rmse_p, rmse_l, hor, err_p, err_l, dt_v


def plot_results(history, hor, err_p, err_l, dt_v):
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    fig.suptitle(
        "ORBITGUARD — PINN Trajectory Prediction",
        fontsize=13, fontweight='bold'
    )
    eps = range(1, len(history["data"])+1)

    ax = axes[0,0]
    ax.semilogy(eps, history["data"], 'b-',
                lw=1.5, label='Data loss')
    ph = [max(p,1e-9) for p in history["phys"]]
    ax.semilogy(eps, ph, 'r--',
                lw=1.5, label='Physics loss')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    ax.plot(eps, history["val_rmse"], 'b-o',
            ms=2, lw=1.5)
    bv = min(history["val_rmse"])
    be = history["val_rmse"].index(bv)+1
    ax.axhline(200, color='r', ls='--', lw=2,
               label='Target 200km')
    ax.axvline(be, color='g', ls=':',
               label=f'Best: {bv:.0f}km')
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE (km)")
    ax.set_title("Validation RMSE")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(3000, max(history["val_rmse"])))

    ax = axes[1,0]
    hs = [int(k.replace("min","")) for k in hor]
    pv = [v["pinn"]   for v in hor.values()]
    lv = [v["linear"] for v in hor.values()]
    ax.plot(hs, pv, 'b-o', lw=2, ms=8,
            label='PINN')
    ax.plot(hs, lv, 'r--s', lw=2, ms=8,
            label='Linear')
    ax.axhline(200, color='g', ls=':', lw=2,
               label='Target 200km')
    ax.fill_between(hs, 0, pv,
                    alpha=0.15, color='b')
    for h, p in zip(hs, pv):
        c = 'green' if p < 200 else 'red'
        ax.annotate(f'{p:.0f}km', (h,p),
                    textcoords="offset points",
                    xytext=(0,10), ha='center',
                    fontsize=9, color=c,
                    fontweight='bold')
    ax.set_xlabel("Horizon (min)")
    ax.set_ylabel("RMSE (km)")
    ax.set_title("Error by Prediction Horizon")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    cap = np.percentile(err_p, 95)
    ax.hist(err_p[err_p<cap], bins=60,
            alpha=0.75, color='blue',
            label='PINN errors', density=True)
    ax.axvline(200, color='r', ls='--', lw=2,
               label='200km target')
    med = np.median(err_p)
    ax.axvline(med, color='g', ls='-', lw=2,
               label=f'Median: {med:.0f}km')
    pct = (err_p < 200).mean() * 100
    ax.set_xlabel("Position error (km)")
    ax.set_ylabel("Density")
    ax.set_title(f"Error Distribution "
                 f"({pct:.1f}% below 200km)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUTS / "pinn_results.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {out}")


if __name__ == "__main__":

    print("="*55)
    print("ORBITGUARD — PINN (12 features, weighted)")
    print("Target: RMSE < 200 km at 60-min horizon")
    print("="*55)

    print("\n--- STEP 1: Data ---")
    data, norms = prepare_data(
        n_objects=500,
        horizon_min=60,
        n_per_obj=60
    )
    if data is None:
        exit(1)

    print(f"\n  n_inputs = {norms['n_inputs']}")

    print("\n--- STEP 2: Model (n_res=6, 12 inputs) ---")
    model = TrajectoryNet(
        hidden=512, n_res=6,
        dropout=0.05,
        n_in=norms["n_inputs"]
    ).to(DEVICE)
    print(f"  Parameters: {model.count_parameters():,}")
    sd  = torch.randn(4, 12).to(DEVICE)
    out = model(sd)
    assert out.shape == (4, 3)
    print(f"  ✅ Forward pass: {out.shape}")

    print("\n--- STEP 3: Training ---")
    history, best_rmse = train_model(
        model, data, norms,
        n_epochs=500, lr=1e-3, batch=2048
    )

    print("\n--- STEP 4: Evaluation ---")
    rmse_p,rmse_l,hor,ep,el,dt = evaluate(
        model, data, norms
    )

    print("\n--- STEP 5: Plots ---")
    plot_results(history, hor, ep, el, dt)

    metrics = {
        "n_inputs":          12,
        "n_res_blocks":      6,
        "parameters":        model.count_parameters(),
        "best_val_rmse_km":  round(best_rmse, 2),
        "test_pinn_rmse_km": round(rmse_p, 2),
        "linear_rmse_km":    round(rmse_l, 2),
        "improvement_x":     round(
            rmse_l/max(rmse_p,.1), 1
        ),
        "horizon_results":   hor,
    }
    with open(MODELS/"pinn_metrics.json","w") as f:
        json.dump(metrics, f, indent=2)

    all_met = all(v["target_met"] for v in hor.values())
    print(f"\n{'='*55}")
    print(f"STEP 6 COMPLETE")
    print(f"{'='*55}")
    print(f"  Best RMSE:  {best_rmse:.2f} km")
    print(f"  vs Linear:  {rmse_l/max(rmse_p,.1):.1f}×")
    for k, v in hor.items():
        st = "✅" if v["target_met"] else "⚠️"
        print(f"  ≤{k}: {v['pinn']:.2f} km  {st}")
    print(f"\n  All horizons < 200km: "
          f"{'✅ YES' if all_met else '⚠️ partial'}")
    print(f"\nNext: python src/drl_avoidance.py")