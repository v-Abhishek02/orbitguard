"""
ORBITGUARD — Dataset Builder (Optimised)
Fixed: Realistic thresholds, capped samples, fast execution
Total runtime: ~15 minutes on MacBook
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sgp4.api import Satrec, jday
from scipy.spatial import KDTree
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

PROCESSED = Path("data/processed")
PROCESSED.mkdir(exist_ok=True)
R_EARTH = 6371.0
GM_KM   = 3.986004418e5   # km^3/s^2

# ── REALISTIC THRESHOLDS (based on NASA/ESA standards) ────────────────────────
# Do NOT use percentile-based thresholds — they produce unrealistic labels
# These are the actual operational thresholds used in real STM
HIGH_KM   = 5.0    # < 5 km   = HIGH risk (NASA action threshold)
MED_KM    = 20.0   # < 20 km  = MED risk  (monitor closely)
LOW_KM    = 50.0   # < 50 km  = LOW risk  (flagged)
SEARCH_KM = 50.0   # search radius

# ── SAMPLE CAPS (keeps runtime under 15 minutes) ──────────────────────────────
MAX_OBJECTS     = 5000   # use 5000 objects — enough density, manageable size
MAX_HIGH        = 2000   # cap HIGH risk sequences
MAX_PER_CLASS   = 4000   # cap MED and LOW sequences
SEQ_LEN         = 20     # sequence length


def load_catalogue():
    candidates = [
        "data/raw/leo_complete_catalogue.json",
        "data/raw/leo_debris_tle_full.json",
        "data/raw/leo_debris_tle.json",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            print(f"Loading: {path}")
            with open(p) as f:
                records = json.load(f)
            valid = [r for r in records
                     if r.get("TLE_LINE1") and r.get("TLE_LINE2")]
            print(f"  Total valid: {len(valid):,}")
            return valid
    raise FileNotFoundError("No catalogue file found")


def propagate_subset(records, n=5000):
    """
    Use a smart subset of objects for training.

    Strategy: sample objects from different altitude shells
    to get good coverage of all LEO zones.
    5000 objects × 1440 timesteps = 7.2M rows — manageable.
    """
    print(f"\nSelecting {n:,} objects from {len(records):,} available...")

    # Parse altitude for each record
    annotated = []
    for r in records:
        try:
            mm  = float(r.get("MEAN_MOTION", 0))
            if mm <= 0:
                continue
            T   = 86400.0 / mm
            a   = (3.986004418e14 * (T / (2*math.pi))**2)**(1/3)
            alt = a/1000 - R_EARTH
            if 150 < alt < 2100:   # valid LEO range
                r["_alt"] = alt
                annotated.append(r)
        except Exception:
            continue

    print(f"  Valid LEO objects: {len(annotated):,}")

    # Sample evenly across altitude bands for diversity
    bands = [
        (150,  450,  n//5,  "Very low LEO"),
        (450,  650,  n//5,  "Starlink zone"),
        (650,  900,  n//5,  "Mid LEO"),
        (900,  1200, n//5,  "Upper LEO"),
        (1200, 2100, n//5,  "High LEO"),
    ]

    selected = []
    seen     = set()

    for lo, hi, quota, name in bands:
        band_objs = [r for r in annotated
                     if lo <= r["_alt"] < hi
                     and r.get("NORAD_CAT_ID") not in seen]
        # Shuffle for randomness
        np.random.seed(42)
        np.random.shuffle(band_objs)
        take = band_objs[:quota]
        selected.extend(take)
        for r in take:
            seen.add(r.get("NORAD_CAT_ID"))
        print(f"  {name} ({lo}-{hi} km): {len(take):,} objects")

    # Fill remainder if any band was short
    remaining = [r for r in annotated
                 if r.get("NORAD_CAT_ID") not in seen]
    needed = n - len(selected)
    if needed > 0 and remaining:
        np.random.shuffle(remaining)
        extra = remaining[:needed]
        selected.extend(extra)
        print(f"  Extra fill: {len(extra):,} objects")

    print(f"  Total selected: {len(selected):,}")
    return selected


def propagate_all(records):
    """
    Propagate for 24 hours (not 48) at 1-minute steps.
    24h is enough for conjunction prediction.
    1-minute steps give finer resolution for close approaches.
    Runtime: ~2 minutes for 5000 objects.
    """
    print(f"\nSGP4 propagation: {len(records):,} objects × 1440 min...")

    all_rows = []
    failed   = 0

    for rec in tqdm(records, unit="obj"):
        norad_id = rec.get("NORAD_CAT_ID", "UNK")
        tle1     = rec["TLE_LINE1"]
        tle2     = rec["TLE_LINE2"]
        try:
            sat = Satrec.twoline2rv(tle1, tle2)
        except Exception:
            failed += 1
            continue

        for t in range(0, 1440, 1):   # 24 hours, 1-min step
            jd, fr = jday(2024, 1, 1, 0, t, 0)
            e, r, v = sat.sgp4(jd, fr)
            if e != 0:
                continue
            dist = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
            if dist < R_EARTH:
                break
            all_rows.append({
                "id": norad_id, "t": t,
                "x": r[0], "y": r[1], "z": r[2],
                "vx": v[0], "vy": v[1], "vz": v[2]
            })

    df = pd.DataFrame(all_rows)
    df["dist"]       = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    df["altitude"]   = df["dist"] - R_EARTH
    df["speed"]      = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)
    df["range_rate"] = (
        df["x"]*df["vx"] + df["y"]*df["vy"] + df["z"]*df["vz"]
    ) / df["dist"]
    df["energy"]     = df["speed"]**2/2 - GM_KM/df["dist"]

    # Remove physically impossible rows
    df = df[df["altitude"].between(150, 2500)].copy()

    print(f"  Objects: {df['id'].nunique():,}  Failed: {failed}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Altitude:{df['altitude'].min():.0f}–"
          f"{df['altitude'].max():.0f} km")

    out = PROCESSED / "trajectories_full.parquet"
    df.to_parquet(out, index=False)
    print(f"  Saved:   {out}  ({out.stat().st_size/1e6:.0f} MB)")
    return df


def find_conjunctions(df):
    """
    Find conjunction events with realistic NASA thresholds.
    Scan every 5th timestep (every 5 minutes).
    With 5000 objects this runs in ~3 minutes.
    """
    timesteps = sorted(df["t"].unique())[::5]
    print(f"\nConjunction scan:")
    print(f"  Objects:    {df['id'].nunique():,}")
    print(f"  Timesteps:  {len(timesteps)} (every 5 min)")
    print(f"  Thresholds: HIGH<{HIGH_KM}km  "
          f"MED<{MED_KM}km  LOW<{LOW_KM}km")
    print(f"  Search:     {SEARCH_KM} km radius")

    all_conj = []

    for t in tqdm(timesteps, unit="ts"):
        frame = df[df["t"] == t]
        if len(frame) < 2:
            continue
        coords = frame[["x","y","z"]].values
        ids    = frame["id"].values
        tree   = KDTree(coords)
        pairs  = tree.query_pairs(r=SEARCH_KM)

        for i, j in pairs:
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist < HIGH_KM:
                risk, rnum = "HIGH", 2
            elif dist < MED_KM:
                risk, rnum = "MED",  1
            else:
                risk, rnum = "LOW",  0
            all_conj.append({
                "t": t, "obj1": ids[i], "obj2": ids[j],
                "miss_km": round(dist, 4),
                "risk": risk, "risk_num": rnum,
            })

    conj_df = pd.DataFrame(all_conj) if all_conj else pd.DataFrame()
    print(f"\n  Total events: {len(conj_df):,}")
    for risk, rn in [("HIGH",2), ("MED",1), ("LOW",0)]:
        cnt = (conj_df["risk_num"] == rn).sum() if len(conj_df) > 0 else 0
        print(f"    {risk}: {cnt:,}")

    return conj_df


def build_sequences(df, conj_df):
    """
    Build training sequences with hard caps per class.
    Maximum total sequences: ~12,000 — builds in ~2 minutes.
    """
    FEATURES = ["x","y","z","vx","vy","vz",
                 "altitude","speed","range_rate"]

    HIGH_df = conj_df[conj_df["risk_num"] == 2]
    MED_df  = conj_df[conj_df["risk_num"] == 1]
    LOW_df  = conj_df[conj_df["risk_num"] == 0]

    high_n  = min(len(HIGH_df), MAX_HIGH)
    med_n   = min(len(MED_df),  MAX_PER_CLASS)
    low_n   = min(len(LOW_df),  MAX_PER_CLASS)

    print(f"\nSequence building:")
    print(f"  HIGH: {len(HIGH_df):,} → use {high_n:,}")
    print(f"  MED:  {len(MED_df):,}  → use {med_n:,}")
    print(f"  LOW:  {len(LOW_df):,}  → use {low_n:,}")
    print(f"  Total to build: {high_n+med_n+low_n:,}")
    print(f"  Estimated time: ~2-4 minutes")

    HIGH_s = HIGH_df.sample(high_n, random_state=42)
    MED_s  = MED_df.sample(med_n,  random_state=42)
    LOW_s  = LOW_df.sample(low_n,  random_state=42)

    work = pd.concat([HIGH_s, MED_s, LOW_s]).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    # Index trajectories
    print("\n  Indexing trajectories...")
    traj_idx = {}
    for obj_id, grp in df.groupby("id"):
        traj_idx[obj_id] = grp.set_index("t")

    X, y   = [], []
    skipped = 0

    print("  Building sequences...")
    for _, ev in tqdm(work.iterrows(),
                      total=len(work),
                      unit="seq"):
        o1   = ev["obj1"]
        o2   = ev["obj2"]
        t_ev = int(ev["t"])
        risk = int(ev["risk_num"])

        tr1 = traj_idx.get(o1)
        tr2 = traj_idx.get(o2)
        if tr1 is None or tr2 is None:
            skipped += 1
            continue

        times = list(range(max(0, t_ev - SEQ_LEN), t_ev))
        if len(times) < SEQ_LEN:
            skipped += 1
            continue

        s1 = (tr1.reindex(times)[FEATURES]
                 .ffill().fillna(0).values)
        s2 = (tr2.reindex(times)[FEATURES]
                 .ffill().fillna(0).values)

        if s1.shape[0] != SEQ_LEN or s2.shape[0] != SEQ_LEN:
            skipped += 1
            continue

        X.append(np.concatenate([s1, s2], axis=1))
        y.append(risk)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"\n  Built:   {len(X):,}")
    print(f"  Skipped: {skipped:,}")
    print(f"  X shape: {X.shape}")

    return X, y


if __name__ == "__main__":

    print("="*55)
    print("ORBITGUARD — Dataset Builder (Optimised)")
    print("Target runtime: ~15 minutes")
    print("="*55)

    # Step 1: Load
    records = load_catalogue()

    # Step 2: Select 5000 diverse objects
    subset = propagate_subset(records, n=MAX_OBJECTS)

    # Step 3: SGP4 (24h, 1-min step)
    df = propagate_all(subset)

    # Step 4: Conjunctions with real thresholds
    conj_df = find_conjunctions(df)

    if len(conj_df) == 0:
        print("\nNo conjunctions found with 5km threshold.")
        print("Check trajectory data.")
        exit(1)

    high_n = (conj_df["risk_num"] == 2).sum()
    if high_n < 10:
        print(f"\nOnly {high_n} HIGH risk events found.")
        print("Increasing search to 100km for training purposes...")
        HIGH_KM   = 20.0
        MED_KM    = 50.0
        LOW_KM    = 100.0
        SEARCH_KM = 100.0
        conj_df = find_conjunctions(df)

    conj_df.to_parquet(PROCESSED / "conjunctions.parquet", index=False)

    # Step 5: Sequences (capped)
    X, y = build_sequences(df, conj_df)

    # Save
    np.save(PROCESSED / "X_sequences.npy", X)
    np.save(PROCESSED / "y_labels.npy",    y)

    names = {0:"LOW", 1:"MED", 2:"HIGH"}
    config = {
        "n_objects":          int(df["id"].nunique()),
        "high_threshold_km":  HIGH_KM,
        "med_threshold_km":   MED_KM,
        "search_radius_km":   SEARCH_KM,
        "total_conjunctions": len(conj_df),
        "high_count":         int((conj_df["risk_num"]==2).sum()),
        "med_count":          int((conj_df["risk_num"]==1).sum()),
        "low_count":          int((conj_df["risk_num"]==0).sum()),
        "n_sequences":        len(X),
        "sequence_length":    SEQ_LEN,
        "n_features":         18,
    }
    with open(PROCESSED / "dataset_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*55}")
    print("DATASET COMPLETE")
    print(f"{'='*55}")
    for label in [2, 1, 0]:
        cnt = (y == label).sum()
        print(f"  {names[label]}: {cnt:,}  ({cnt/len(y)*100:.1f}%)")
    print(f"\n  X: {X.shape}")
    print(f"  Saved to data/processed/")
    print(f"\n✅ Now run: python src/cnn_lstm_fixed.py")