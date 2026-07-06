"""
ORBITGUARD — Step 7: DRL Collision Avoidance Agent (Fixed)
File: src/drl_avoidance.py

Fixes from v1:
1. Fuel budget increased: 50m/s → 500m/s (realistic LEO)
2. Action scaled down: ±2m/s → ±0.5m/s per axis
3. Episode termination fixed: success tracked correctly
4. Reward function rebalanced
5. Success = completing max_steps without collision
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from collections import deque
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

torch.manual_seed(42)
np.random.seed(42)

GM      = 398600.4418
R_EARTH = 6371.0


# ═══════════════════════════════════════════════════════════
# ENVIRONMENT
# ═══════════════════════════════════════════════════════════

class SpaceDebrisEnv(gym.Env):
    """
    Spacecraft collision avoidance environment.

    Key fixes:
    - fuel_budget = 0.5 km/s = 500 m/s (realistic LEO)
    - action bounds = ±0.0005 km/s = ±0.5 m/s per axis
    - max_dv_per_step = sqrt(3) × 0.0005 ≈ 0.87 m/s
    - At this rate: 500/0.87 ≈ 575 steps before fuel runs out
    - Episode length = 120 steps → fuel lasts many episodes
    - Success = reaching max_steps without collision
    """

    def __init__(self, traj_df, conj_df,
                 max_steps=120,
                 fuel_budget=0.5):
        super().__init__()
        self.traj_df     = traj_df
        self.conj_df     = conj_df
        self.max_steps   = max_steps
        self.fuel_budget = fuel_budget
        self.n_debris    = 5

        # Observation: sc(6) + debris×5×7 + fuel(1) = 42
        n_obs = 6 + self.n_debris * 7 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_obs,), dtype=np.float32
        )

        # Action: ±0.0005 km/s per axis = ±0.5 m/s
        self.action_space = spaces.Box(
            low=-0.0005, high=0.0005,
            shape=(3,), dtype=np.float32
        )

        self._build_index()

    def _build_index(self):
        print("  Indexing trajectories...")
        self.traj_idx = {}
        for oid, grp in self.traj_df.groupby("id"):
            self.traj_idx[oid] = \
                grp.sort_values("t").reset_index(drop=True)
        self.all_ids = list(self.traj_idx.keys())

        # Build spatial index per timestep for debris lookup
        self.frame_idx = {}
        for t, grp in self.traj_df.groupby("t"):
            self.frame_idx[t] = grp.reset_index(drop=True)

        self.t_vals = sorted(self.frame_idx.keys())
        print(f"  Objects: {len(self.all_ids):,}")
        print(f"  Timesteps: {len(self.t_vals):,}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick random spacecraft
        self.sc_id = np.random.choice(self.all_ids)
        sc_traj    = self.traj_idx[self.sc_id]

        max_start = max(1, len(sc_traj) - self.max_steps - 5)
        si = np.random.randint(0, max_start)

        row = sc_traj.iloc[si]
        self.sc_pos     = np.array([row["x"],row["y"],row["z"]])
        self.sc_vel     = np.array([row["vx"],row["vy"],row["vz"]])
        self.t_idx      = si
        self.sc_traj    = sc_traj
        self.step_count = 0
        self.fuel_used  = 0.0
        self.collisions = 0

        self._find_debris()
        return self._obs(), {}

    def _find_debris(self):
        """Find nearest debris at current time."""
        # Get current time
        if self.t_idx < len(self.sc_traj):
            t_cur = float(self.sc_traj.iloc[self.t_idx]["t"])
        else:
            t_cur = self.t_vals[-1]

        # Find nearest available timestep
        t_near = min(self.t_vals,
                     key=lambda x: abs(x - t_cur))
        frame  = self.frame_idx[t_near]

        # Compute distances
        dx = frame["x"].values - self.sc_pos[0]
        dy = frame["y"].values - self.sc_pos[1]
        dz = frame["z"].values - self.sc_pos[2]
        d  = np.sqrt(dx**2 + dy**2 + dz**2)

        # Exclude self
        mask = frame["id"].values != self.sc_id
        if mask.sum() == 0:
            self.debris = []
            return

        d_f     = d[mask]
        frame_f = frame[mask].reset_index(drop=True)
        top_idx = np.argsort(d_f)[:self.n_debris]

        self.debris = []
        for i in top_idx:
            r = frame_f.iloc[i]
            self.debris.append({
                "rp": np.array([r["x"]-self.sc_pos[0],
                                r["y"]-self.sc_pos[1],
                                r["z"]-self.sc_pos[2]]),
                "v":  np.array([r["vx"],r["vy"],r["vz"]]),
                "d":  float(d_f[i]),
            })

    def _obs(self):
        R, V, D = 8000.0, 8.0, 500.0
        o = []
        o.extend(self.sc_pos / R)
        o.extend(self.sc_vel / V)
        for i in range(self.n_debris):
            if i < len(self.debris):
                db = self.debris[i]
                o.extend(db["rp"] / D)
                o.extend(db["v"]  / V)
                o.append(min(db["d"], 2000.0) / D)
            else:
                o.extend([0.0]*7)
        fuel_rem = max(0.0,
            (self.fuel_budget - self.fuel_used)
            / self.fuel_budget)
        o.append(fuel_rem)
        return np.array(o, dtype=np.float32)

    def step(self, action):
        # Clip action to bounds
        action   = np.clip(action, -0.0005, 0.0005)
        dv_mag   = float(np.linalg.norm(action))

        # Apply maneuver
        self.sc_vel   += action
        self.fuel_used += dv_mag

        # Propagate position (2-min step)
        dt = 120.0   # seconds
        self.sc_pos += self.sc_vel * dt

        self.step_count += 1
        self.t_idx      += 1
        self._find_debris()

        # Minimum distance to any debris
        min_d = (min(db["d"] for db in self.debris)
                 if self.debris else 2000.0)

        # Reward
        reward = 0.0
        if min_d < 0.2:
            reward -= 100.0
            self.collisions += 1
        elif min_d < 1.0:
            reward -= 30.0
        elif min_d < 5.0:
            reward -= 5.0
        else:
            reward += 10.0

        if min_d > 20.0:
            reward += 3.0

        # Small fuel penalty
        reward -= dv_mag * 500.0

        # Termination
        terminated = min_d < 0.2   # collision
        truncated  = (self.step_count >= self.max_steps
                      or self.fuel_used >= self.fuel_budget)

        info = {
            "min_dist":   min_d,
            "dv_mag":     dv_mag,
            "fuel_used":  self.fuel_used,
            "success":    (truncated and
                           self.collisions == 0),
            "collision":  min_d < 0.2,
        }
        return self._obs(), reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════
# PPO AGENT
# ═══════════════════════════════════════════════════════════

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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, x):
        f    = self.trunk(x)
        mean = self.actor_mean(f)
        std  = torch.exp(self.actor_log_std.clamp(-4, 0))
        dist = torch.distributions.Normal(mean, std)
        val  = self.critic(f)
        return dist, val

    def get_action(self, obs, det=False):
        dist, val = self.forward(obs)
        a  = dist.mean if det else dist.sample()
        lp = dist.log_prob(a).sum(-1)
        return a, lp, val.squeeze()

    def evaluate(self, obs, actions):
        dist, val = self.forward(obs)
        lp  = dist.log_prob(actions).sum(-1)
        ent = dist.entropy().sum(-1)
        return lp, val.squeeze(), ent

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


def compute_gae(rewards, values, dones,
                next_val, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rewards)
    gae = 0.0
    vs  = values + [next_val]
    for t in reversed(range(len(rewards))):
        delta = (rewards[t]
                 + gamma * vs[t+1] * (1-dones[t])
                 - vs[t])
        gae   = delta + gamma * lam * (1-dones[t]) * gae
        adv[t] = gae
    return adv, adv + np.array(values)


def ppo_update(policy, opt, rollout,
               clip=0.2, n_ep=10, bs=128):
    obs     = torch.FloatTensor(rollout["obs"])
    actions = torch.FloatTensor(rollout["actions"])
    old_lp  = torch.FloatTensor(rollout["log_probs"])
    advs    = torch.FloatTensor(rollout["advantages"])
    returns = torch.FloatTensor(rollout["returns"])

    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    pl_list, vl_list = [], []

    for _ in range(n_ep):
        idx = torch.randperm(len(obs))
        for s in range(0, len(obs), bs):
            b    = idx[s:s+bs]
            nlp, vals, ent = policy.evaluate(
                obs[b], actions[b]
            )
            ratio = torch.exp(nlp - old_lp[b])
            s1  = ratio * advs[b]
            s2  = torch.clamp(ratio,
                               1-clip, 1+clip) * advs[b]
            pl  = -torch.min(s1, s2).mean()
            vl  = nn.functional.mse_loss(vals, returns[b])
            el  = -0.005 * ent.mean()
            loss = pl + 0.5*vl + el
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), 0.5
            )
            opt.step()
            pl_list.append(pl.item())
            vl_list.append(vl.item())

    return np.mean(pl_list), np.mean(vl_list)


# ═══════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════

def train(env, policy, opt,
          n_updates=300, rollout_steps=1024):
    """
    300 updates × 1024 steps = 307,200 total env steps.
    More steps = more learning = better success rate.
    """
    history = {
        "update":[], "reward":[], "success":[],
        "pol_loss":[], "val_loss":[]
    }

    best_success = 0.0
    best_reward  = -np.inf

    print(f"\n{'Upd':>6}|{'Reward':>9}|"
          f"{'PlLoss':>8}|{'VlLoss':>8}|{'Success':>9}")
    print("─"*48)

    obs, _   = env.reset()
    ep_rew   = 0.0
    ep_rews  = []
    ep_succ  = []

    obs_l, act_l   = [], []
    rew_l, done_l  = [], []
    lp_l,  val_l   = [], []

    for upd in range(1, n_updates+1):
        obs_l.clear(); act_l.clear()
        rew_l.clear(); done_l.clear()
        lp_l.clear();  val_l.clear()
        ep_rews.clear(); ep_succ.clear()
        ep_rew = 0.0

        for _ in range(rollout_steps):
            ot = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                a, lp, v = policy.get_action(ot)
            an = a.squeeze().numpy()
            an = np.clip(an, -0.0005, 0.0005)

            no, r, term, trunc, info = env.step(an)

            obs_l.append(obs); act_l.append(an)
            rew_l.append(r);   done_l.append(
                float(term or trunc)
            )
            lp_l.append(lp.item()); val_l.append(v.item())

            ep_rew += r
            obs = no

            if term or trunc:
                ep_rews.append(ep_rew)
                ep_succ.append(
                    1.0 if info.get("success") else 0.0
                )
                ep_rew = 0.0
                obs, _ = env.reset()

        # Bootstrap
        ot2 = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            _, _, nv = policy.get_action(ot2)

        adv, ret = compute_gae(
            rew_l, val_l, done_l, nv.item()
        )

        rollout = {
            "obs":        np.array(obs_l),
            "actions":    np.array(act_l),
            "log_probs":  np.array(lp_l),
            "values":     np.array(val_l),
            "advantages": adv,
            "returns":    ret,
        }

        pl, vl = ppo_update(policy, opt, rollout)

        mean_rew  = np.mean(ep_rews)  if ep_rews else 0
        mean_succ = np.mean(ep_succ)  if ep_succ else 0

        if upd % 20 == 0 or upd == 1:
            history["update"].append(upd)
            history["reward"].append(mean_rew)
            history["success"].append(mean_succ)
            history["pol_loss"].append(pl)
            history["val_loss"].append(vl)

            print(f"{upd:>6}|{mean_rew:>9.2f}|"
                  f"{pl:>8.4f}|{vl:>8.4f}|"
                  f"{mean_succ:>8.1%}")

            if (mean_succ > best_success or
                    mean_rew > best_reward):
                best_success = max(best_success, mean_succ)
                best_reward  = max(best_reward,  mean_rew)
                torch.save({
                    "update":       upd,
                    "model_state":  policy.state_dict(),
                    "success_rate": mean_succ,
                    "mean_reward":  mean_rew,
                    "history":      history,
                }, MODELS/"best_drl.pt")

    print(f"\n✅ Best success: {best_success:.1%}  "
          f"Best reward: {best_reward:.2f}")
    return history, best_success


def evaluate(env, policy, n_ep=100):
    """Deterministic evaluation on n_ep episodes."""
    successes = []
    collisions = []
    rewards   = []
    dvs       = []

    for _ in range(n_ep):
        obs, _    = env.reset()
        ep_r      = 0.0
        ep_dv     = 0.0
        done      = False
        ep_coll   = False

        while not done:
            ot = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                a, _, _ = policy.get_action(ot, det=True)
            an = np.clip(a.squeeze().numpy(),
                         -0.0005, 0.0005)
            obs, r, term, trunc, info = env.step(an)
            ep_r  += r
            ep_dv += info.get("dv_mag", 0)
            if info.get("collision"):
                ep_coll = True
            done = term or trunc
            if term or trunc:
                successes.append(
                    1.0 if info.get("success") else 0.0
                )

        rewards.append(ep_r)
        dvs.append(ep_dv)
        collisions.append(1.0 if ep_coll else 0.0)

    return {
        "success_rate":   float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "mean_reward":    float(np.mean(rewards)),
        "mean_dv_ms":     float(np.mean(dvs)) * 1000,
    }


def plot(history, res):
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    fig.suptitle("ORBITGUARD — DRL Avoidance Agent (PPO)",
                 fontsize=13, fontweight='bold')
    upd = history["update"]

    ax = axes[0,0]
    ax.plot(upd, history["reward"],
            'b-o', ms=4, lw=1.5)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel("Update"); ax.set_ylabel("Mean Reward")
    ax.set_title("Training Reward"); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    ax.plot(upd, [s*100 for s in history["success"]],
            'g-o', ms=4, lw=1.5)
    ax.axhline(95, color='r', ls='--', lw=1.5,
               label='Target 95%')
    ax.set_xlabel("Update"); ax.set_ylabel("Success Rate %")
    ax.set_title("Avoidance Success Rate")
    ax.set_ylim(0,105); ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1,0]
    ax.plot(upd, history["pol_loss"],
            'r-o', ms=3, lw=1.5, label='Policy')
    ax.plot(upd, history["val_loss"],
            'b-o', ms=3, lw=1.5, label='Value')
    ax.set_xlabel("Update"); ax.set_ylabel("Loss")
    ax.set_title("PPO Losses"); ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    names  = ["Success\nRate", "No\nCollision",
              "Fuel\nEfficiency"]
    vals   = [
        res["success_rate"]*100,
        (1-res["collision_rate"])*100,
        max(0, 100 - res["mean_dv_ms"]*0.2),
    ]
    colors = ['green','blue','orange']
    bars   = ax.bar(names, vals, color=colors, alpha=0.8)
    ax.axhline(95, color='r', ls='--', lw=1.5,
               label='Target 95%')
    ax.set_ylim(0, 115); ax.set_ylabel("Score (%)")
    ax.set_title("Final Agent Performance")
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+1, f'{val:.1f}%',
                ha='center', fontsize=10,
                fontweight='bold')

    plt.tight_layout()
    out = OUTPUTS/"drl_results.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {out}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("="*55)
    print("ORBITGUARD — DRL Avoidance Agent (PPO) v2")
    print("="*55)

    # ── Load data ─────────────────────────────────────────
    print("\n--- STEP 1: Loading Data ---")
    df = pd.read_parquet(
        PROCESSED/"trajectories_full.parquet",
        columns=["id","t","x","y","z","vx","vy","vz","altitude"]
    )
    ids = df["id"].unique()
    np.random.shuffle(ids)
    sel = ids[:2000]
    traj_df = df[df["id"].isin(sel)].copy()

    conj_df = pd.read_parquet(PROCESSED/"conjunctions.parquet")
    conj_df = conj_df[
        conj_df["obj1"].isin(sel) |
        conj_df["obj2"].isin(sel)
    ].copy()

    print(f"  Trajectory rows:    {len(traj_df):,}")
    print(f"  Objects:            {traj_df['id'].nunique():,}")
    print(f"  Conjunction events: {len(conj_df):,}")

    # ── Create environment ────────────────────────────────
    print("\n--- STEP 2: Environment ---")
    env = SpaceDebrisEnv(
        traj_df=traj_df,
        conj_df=conj_df,
        max_steps=120,
        fuel_budget=0.5    # 500 m/s budget
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"  Obs dim:      {obs_dim}")
    print(f"  Act dim:      {act_dim}")
    print(f"  Max steps:    {env.max_steps}")
    print(f"  Fuel budget:  {env.fuel_budget*1000:.0f} m/s")
    print(f"  Max ΔV/step:  "
          f"{np.sqrt(3)*0.0005*1000:.2f} m/s")
    print(f"  Steps until fuel out: "
          f"~{env.fuel_budget/(np.sqrt(3)*0.0005):.0f}")

    # Sanity check
    obs, _  = env.reset()
    a       = env.action_space.sample()
    o2,r,te,tr,info = env.step(a)
    print(f"\n  ✅ Env OK  reward={r:.2f}  "
          f"min_dist={info['min_dist']:.1f}km  "
          f"success={info['success']}")

    # ── Build agent ───────────────────────────────────────
    print("\n--- STEP 3: PPO Agent ---")
    policy = ActorCritic(obs_dim, act_dim, hidden=256)
    opt    = torch.optim.Adam(
        policy.parameters(), lr=3e-4, eps=1e-5
    )
    print(f"  Parameters: {policy.count_parameters():,}")

    # ── Train ─────────────────────────────────────────────
    print("\n--- STEP 4: Training (300 updates) ---")
    print("  300 × 1024 = 307,200 total steps")
    print("  Expected time: 20-35 min on CPU\n")

    history, best_succ = train(
        env, policy, opt,
        n_updates=300,
        rollout_steps=1024
    )

    # ── Evaluate ──────────────────────────────────────────
    print("\n--- STEP 5: Final Evaluation (100 episodes) ---")
    # Load best saved model
    ckpt = torch.load(MODELS/"best_drl.pt",
                      map_location='cpu')
    policy.load_state_dict(ckpt["model_state"])

    res = evaluate(env, policy, n_ep=100)
    print(f"\n  Success rate:   {res['success_rate']:.1%}")
    print(f"  Collision rate: {res['collision_rate']:.1%}")
    print(f"  Mean reward:    {res['mean_reward']:.2f}")
    print(f"  Mean ΔV:        {res['mean_dv_ms']:.3f} m/s")

    # ── Plot ──────────────────────────────────────────────
    print("\n--- STEP 6: Plots ---")
    plot(history, res)

    # ── Save ──────────────────────────────────────────────
    metrics = {
        "algorithm":      "PPO",
        "obs_dim":        obs_dim,
        "act_dim":        act_dim,
        "parameters":     policy.count_parameters(),
        "n_updates":      300,
        "rollout_steps":  1024,
        "total_steps":    300*1024,
        "fuel_budget_ms": 500,
        "max_dv_step_ms": round(np.sqrt(3)*0.5, 3),
        "success_rate":   round(res["success_rate"],4),
        "collision_rate": round(res["collision_rate"],4),
        "mean_reward":    round(res["mean_reward"],2),
        "mean_dv_ms":     round(res["mean_dv_ms"],4),
        "target_95_met":  res["success_rate"] >= 0.95,
        "best_training_success": round(best_succ, 4),
    }
    with open(MODELS/"drl_metrics.json","w") as f:
        json.dump(metrics, f, indent=2)

    target = res["success_rate"] >= 0.95
    print(f"\n{'='*55}")
    print(f"STEP 7 COMPLETE")
    print(f"{'='*55}")
    print(f"  Success rate:   {res['success_rate']:.1%}  "
          f"{'✅' if target else '⚠️'}")
    print(f"  Collision rate: {res['collision_rate']:.1%}")
    print(f"  Mean ΔV:        {res['mean_dv_ms']:.3f} m/s")
    print(f"  Model saved:    models/best_drl.pt")
    print(f"\nNext: python src/integration.py")