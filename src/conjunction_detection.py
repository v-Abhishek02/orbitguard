"""
ORBITGUARD — Stage 1: Conjunction Detection
File: src/conjunction_detection.py
Purpose: Find all close approach events between debris pairs using KD-Trees
         Output: labelled dataset with HIGH / MED / LOW risk events
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("data/processed")

# ── Risk thresholds (km) ──────────────────────────────────────────────────────
# These match real-world NASA/ESA conjunction alert thresholds
HIGH_THRESHOLD = 1.0    # < 1 km   = HIGH risk  (maneuver almost certain)
MED_THRESHOLD  = 5.0    # < 5 km   = MED risk   (monitor closely)
LOW_THRESHOLD  = 10.0   # < 10 km  = LOW risk   (flagged, watch)
SEARCH_RADIUS  = 10.0   # search radius for KD-Tree query


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD TRAJECTORY DATA
# ═════════════════════════════════════════════════════════════════════════════

def load_trajectories(filepath):
    """
    Load the full trajectory parquet file.
    Print memory usage so you understand the scale of your data.
    """
    print(f"Loading trajectories from {filepath}...")
    df = pd.read_parquet(filepath)

    mem_gb = df.memory_usage(deep=True).sum() / 1e9
    print(f"  Rows:           {len(df):,}")
    print(f"  Unique objects: {df['id'].nunique():,}")
    print(f"  Timesteps:      {df['t'].nunique():,}")
    print(f"  Memory usage:   {mem_gb:.2f} GB")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: KD-TREE CONJUNCTION SEARCH
# ═════════════════════════════════════════════════════════════════════════════

def find_conjunctions_at_timestep(frame, search_radius_km=10.0):
    """
    Find all pairs of objects within search_radius_km at ONE timestep.

    Why KD-Tree instead of nested loops:
    - Nested loops: O(n²) = 9050² = 81 million comparisons per timestep
    - KD-Tree:      O(n log n) = ~120,000 operations per timestep
    - Speedup: ~675x faster

    Returns list of conjunction dicts for this timestep.
    """
    coords = frame[["x", "y", "z"]].values   # shape: (n_objects, 3)
    ids    = frame["id"].values                # shape: (n_objects,)

    # Build KD-Tree from all object positions at this timestep
    tree = KDTree(coords)

    # Query all pairs within search radius — returns set of (i, j) pairs
    pairs = tree.query_pairs(r=search_radius_km)

    results = []
    for i, j in pairs:
        # Compute exact Euclidean distance between the pair
        dist = np.linalg.norm(coords[i] - coords[j])

        # Assign risk label based on miss distance
        if dist < HIGH_THRESHOLD:
            risk = "HIGH"
            risk_num = 2
        elif dist < MED_THRESHOLD:
            risk = "MED"
            risk_num = 1
        else:
            risk = "LOW"
            risk_num = 0

        results.append({
            "obj1":     ids[i],
            "obj2":     ids[j],
            "miss_km":  round(dist, 4),
            "risk":     risk,
            "risk_num": risk_num,   # numeric label for ML: 0=LOW,1=MED,2=HIGH
        })

    return results


def find_all_conjunctions(df, search_radius_km=10.0,
                           sample_timesteps=None):
    """
    Scan ALL timesteps and find all conjunction events.

    sample_timesteps: if set (e.g. 100), only scan that many
                      timesteps for quick testing.
    Full run: 2880 timesteps × 9050 objects ~ 20-40 minutes.
    """
    timesteps = sorted(df["t"].unique())

    if sample_timesteps:
        # Evenly sample timesteps across the full 48 hours
        step = max(1, len(timesteps) // sample_timesteps)
        timesteps = timesteps[::step][:sample_timesteps]
        print(f"  (Test mode: scanning {len(timesteps)} timesteps)")
    else:
        print(f"  (Full mode: scanning {len(timesteps)} timesteps)")

    all_conjunctions = []

    print(f"\nScanning {len(timesteps)} timesteps for conjunctions...")
    print(f"Search radius: {search_radius_km} km")
    print("Progress:")

    for t in tqdm(timesteps, unit="timestep"):
        # Get all object positions at this timestep
        frame = df[df["t"] == t]

        if len(frame) < 2:
            continue    # need at least 2 objects to have a conjunction

        # Find all close pairs at this timestep
        conjunctions = find_conjunctions_at_timestep(frame, search_radius_km)

        # Add timestep info to each conjunction
        for c in conjunctions:
            c["t"] = t

        all_conjunctions.extend(conjunctions)

    print(f"\nScan complete!")
    print(f"  Total conjunction events found: {len(all_conjunctions):,}")

    if len(all_conjunctions) == 0:
        print("  WARNING: Zero conjunctions found!")
        print("  Try increasing search radius or checking trajectory data.")
        return pd.DataFrame()

    conj_df = pd.DataFrame(all_conjunctions)
    return conj_df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: ANALYSE CONJUNCTION RESULTS
# ═════════════════════════════════════════════════════════════════════════════

def analyse_conjunctions(conj_df, traj_df):
    """
    Full statistical analysis of the conjunction events.
    These numbers go directly into your paper's Dataset section.
    """
    print("\n" + "="*60)
    print("CONJUNCTION ANALYSIS REPORT")
    print("="*60)

    total        = len(conj_df)
    high_count   = len(conj_df[conj_df["risk"] == "HIGH"])
    med_count    = len(conj_df[conj_df["risk"] == "MED"])
    low_count    = len(conj_df[conj_df["risk"] == "LOW"])

    total_rows   = len(traj_df)
    safe_rows    = total_rows - total

    print(f"Total trajectory rows:        {total_rows:,}")
    print(f"Total conjunction events:     {total:,}")
    print(f"  HIGH risk (< 1 km):         {high_count:,}")
    print(f"  MED  risk (1-5 km):         {med_count:,}")
    print(f"  LOW  risk (5-10 km):        {low_count:,}")
    print(f"  Safe (> 10 km):             {safe_rows:,}")

    print(f"\nClass imbalance ratios:")
    if high_count > 0:
        print(f"  HIGH : SAFE  =  1 : "
              f"{safe_rows // max(high_count,1):,}")
        print(f"  MED  : SAFE  =  1 : "
              f"{safe_rows // max(med_count,1):,}")

    print(f"\nMiss distance statistics (km):")
    print(f"  Min:    {conj_df['miss_km'].min():.4f}")
    print(f"  Max:    {conj_df['miss_km'].max():.4f}")
    print(f"  Mean:   {conj_df['miss_km'].mean():.4f}")
    print(f"  Median: {conj_df['miss_km'].median():.4f}")

    # Most dangerous pairs
    print(f"\nTop 10 closest approaches:")
    print("-"*50)
    top10 = conj_df.nsmallest(10, "miss_km")[
        ["obj1", "obj2", "miss_km", "risk", "t"]
    ]
    for _, row in top10.iterrows():
        hours = row["t"] / 60
        print(f"  {row['obj1']} ↔ {row['obj2']}"
              f"   {row['miss_km']:.4f} km"
              f"   [{row['risk']}]"
              f"   T+{hours:.1f}h")

    # Objects involved in most conjunctions
    all_ids = pd.concat([conj_df["obj1"], conj_df["obj2"]])
    top_objects = all_ids.value_counts().head(5)
    print(f"\nTop 5 most frequently involved objects:")
    for obj_id, count in top_objects.items():
        print(f"  NORAD {obj_id}: {count} conjunction events")

    print("="*60)

    # Return summary dict for saving
    return {
        "total_trajectory_rows":  total_rows,
        "total_conjunctions":     total,
        "high_risk_count":        high_count,
        "med_risk_count":         med_count,
        "low_risk_count":         low_count,
        "safe_rows":              safe_rows,
        "high_safe_ratio":        safe_rows // max(high_count, 1),
        "min_miss_km":            float(conj_df["miss_km"].min()),
        "max_miss_km":            float(conj_df["miss_km"].max()),
        "mean_miss_km":           float(conj_df["miss_km"].mean()),
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: BUILD ML TRAINING DATASET
# ═════════════════════════════════════════════════════════════════════════════

def build_training_sequences(traj_df, conj_df,
                              sequence_length=20,
                              features=None):
    """
    Build the final training dataset for your CNN+LSTM model.

    For each conjunction event, extract a sequence of the last
    `sequence_length` orbital state observations for BOTH objects
    involved. Label that sequence with the risk level.

    This is exactly the format your CNN+LSTM expects:
    - Input:  (sequence_length, n_features) per object
    - Output: risk label (0=LOW, 1=MED, 2=HIGH)

    sequence_length=20 means 20 minutes of orbital history
    before the conjunction event.
    """

    if features is None:
        features = ["x", "y", "z", "vx", "vy", "vz",
                    "altitude", "speed", "range_rate"]

    print(f"\nBuilding training sequences...")
    print(f"  Sequence length:  {sequence_length} timesteps")
    print(f"  Features:         {features}")
    print(f"  Conjunction events to process: {len(conj_df):,}")

    # Index trajectory data by object id for fast lookup
    print("  Indexing trajectory data by object...")
    traj_indexed = {}
    for obj_id, group in tqdm(
        traj_df.groupby("id"), desc="Indexing", unit="obj"
    ):
        traj_indexed[obj_id] = group.set_index("t")

    sequences   = []
    labels      = []
    metadata    = []
    skipped     = 0

    print("  Building sequences...")
    for _, event in tqdm(
        conj_df.iterrows(), total=len(conj_df),
        desc="Sequences", unit="event"
    ):
        obj1_id = event["obj1"]
        obj2_id = event["obj2"]
        t_event = event["t"]
        risk    = event["risk_num"]

        # Get trajectory data for both objects
        obj1_traj = traj_indexed.get(obj1_id)
        obj2_traj = traj_indexed.get(obj2_id)

        if obj1_traj is None or obj2_traj is None:
            skipped += 1
            continue

        # Extract the sequence_length timesteps BEFORE this conjunction
        # This is the "history" the model uses to predict risk
        t_start = t_event - sequence_length
        t_end   = t_event

        # Get sequence for object 1
        seq1_times = range(t_start, t_end)
        seq1 = obj1_traj.reindex(seq1_times)[features]

        # Get sequence for object 2
        seq2 = obj2_traj.reindex(seq1_times)[features]

        # Skip if too many missing values
        if seq1.isnull().sum().sum() > 5 or \
           seq2.isnull().sum().sum() > 5:
            skipped += 1
            continue

        # Fill any remaining NaN with forward fill then zero
        seq1 = seq1.ffill().fillna(0)
        seq2 = seq2.ffill().fillna(0)

        # Combine both objects' sequences:
        # Shape: (sequence_length, n_features * 2)
        # This gives the model both objects' states simultaneously
        combined = np.concatenate(
            [seq1.values, seq2.values], axis=1
        )

        sequences.append(combined)
        labels.append(risk)
        metadata.append({
            "obj1":     obj1_id,
            "obj2":     obj2_id,
            "t":        t_event,
            "miss_km":  event["miss_km"],
            "risk":     event["risk"],
        })

    print(f"\n  Sequences built:  {len(sequences):,}")
    print(f"  Skipped:          {skipped:,}")

    if len(sequences) == 0:
        print("  WARNING: No sequences built!")
        return None, None, None

    # Convert to numpy arrays
    X = np.array(sequences, dtype=np.float32)
    # Shape: (n_samples, sequence_length, n_features*2)
    y = np.array(labels, dtype=np.int64)
    # Shape: (n_samples,)

    print(f"\n  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Label distribution:")
    for label, name in [(0,"LOW"),(1,"MED"),(2,"HIGH")]:
        count = (y == label).sum()
        pct   = count / len(y) * 100
        print(f"    {name}: {count:,}  ({pct:.1f}%)")

    return X, y, metadata


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: VISUALISE RESULTS
# ═════════════════════════════════════════════════════════════════════════════

def visualise_conjunctions(conj_df, traj_df):
    """
    Create 3 visualisation plots:
    1. Miss distance distribution
    2. Risk level pie chart
    3. Conjunctions over time
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ORBITGUARD — Conjunction Analysis",
                 fontsize=14, fontweight='bold')

    # ── Plot 1: Miss distance histogram ───────────────────────────────────
    ax = axes[0]
    colors_map = {"HIGH": "#E53E3E", "MED": "#DD6B20", "LOW": "#3182CE"}
    for risk, color in colors_map.items():
        subset = conj_df[conj_df["risk"] == risk]["miss_km"]
        if len(subset) > 0:
            ax.hist(subset, bins=50, color=color,
                    alpha=0.7, label=risk, density=True)
    ax.axvline(x=HIGH_THRESHOLD, color='red',
               linestyle='--', alpha=0.8, label='HIGH threshold (1km)')
    ax.axvline(x=MED_THRESHOLD, color='orange',
               linestyle='--', alpha=0.8, label='MED threshold (5km)')
    ax.set_xlabel("Miss Distance (km)")
    ax.set_ylabel("Density")
    ax.set_title("Miss Distance Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Risk level counts (bar chart) ──────────────────────────────
    ax = axes[1]
    risk_counts = conj_df["risk"].value_counts()
    risk_order  = ["HIGH", "MED", "LOW"]
    risk_colors = ["#E53E3E", "#DD6B20", "#3182CE"]
    bars = ax.bar(
        [r for r in risk_order if r in risk_counts.index],
        [risk_counts.get(r, 0) for r in risk_order if r in risk_counts.index],
        color=[c for r, c in zip(risk_order, risk_colors)
               if r in risk_counts.index]
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h,
                f'{int(h):,}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Events")
    ax.set_title("Conjunction Events by Risk Level")
    ax.grid(True, alpha=0.3, axis='y')

    # ── Plot 3: Conjunctions over time ─────────────────────────────────────
    ax = axes[2]
    for risk, color in colors_map.items():
        subset = conj_df[conj_df["risk"] == risk]
        if len(subset) > 0:
            hours = subset["t"] / 60
            ax.scatter(hours, subset["miss_km"],
                       c=color, s=1, alpha=0.4, label=risk)
    ax.axhline(y=HIGH_THRESHOLD, color='red',
               linestyle='--', alpha=0.6)
    ax.axhline(y=MED_THRESHOLD, color='orange',
               linestyle='--', alpha=0.6)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Miss Distance (km)")
    ax.set_title("Conjunctions Over 48 Hours")
    patches = [mpatches.Patch(color=c, label=r)
               for r, c in colors_map.items()]
    ax.legend(handles=patches, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/conjunction_analysis.png",
                dpi=150, bbox_inches='tight')
    print("\nSaved: outputs/conjunction_analysis.png")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("="*60)
    print("ORBITGUARD — Conjunction Detection")
    print("="*60)

    # ── Load full trajectory data ─────────────────────────────────────────
    traj_df = load_trajectories(
        PROCESSED_DIR / "trajectories_full.parquet"
    )

    # ── PHASE 1: Quick test on 100 timesteps ──────────────────────────────
    print("\n--- PHASE 1: Quick conjunction test (100 timesteps) ---")
    conj_test = find_all_conjunctions(
        traj_df,
        search_radius_km=10.0,
        sample_timesteps=100
    )

    if len(conj_test) > 0:
        print("\nTest found conjunctions — proceeding to full scan")
        analyse_conjunctions(conj_test, traj_df)
    else:
        print("WARNING: No conjunctions found in test sample")
        print("This may mean your debris objects are well-separated")
        print("Try increasing search_radius_km to 50.0")

    # ── PHASE 2: Full conjunction scan ────────────────────────────────────
    print("\n--- PHASE 2: Full conjunction scan (all 2880 timesteps) ---")
    print("This will take 20-40 minutes. Go get a coffee.")

    answer = input("Run full conjunction scan now? (yes/no): ").strip().lower()

    if answer == "yes":
        conj_full = find_all_conjunctions(
            traj_df,
            search_radius_km=10.0,
            sample_timesteps=None    # scan ALL timesteps
        )

        if len(conj_full) == 0:
            print("No conjunctions found. Try radius 50km:")
            conj_full = find_all_conjunctions(
                traj_df,
                search_radius_km=50.0,
                sample_timesteps=None
            )

        # Save raw conjunction events
        conj_full.to_parquet(
            PROCESSED_DIR / "conjunctions.parquet",
            index=False
        )
        print(f"\nSaved: data/processed/conjunctions.parquet")
        conj_file = PROCESSED_DIR / 'conjunctions.parquet'
        print(f"Size: {conj_file.stat().st_size / 1e6:.2f} MB")

        # Full analysis
        summary = analyse_conjunctions(conj_full, traj_df)

        # Save summary stats as JSON
        with open("data/processed/dataset_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("Saved: data/processed/dataset_summary.json")

        # Visualise
        visualise_conjunctions(conj_full, traj_df)

        # ── PHASE 3: Build ML training sequences ──────────────────────────
        print("\n--- PHASE 3: Building ML training sequences ---")
        X, y, meta = build_training_sequences(
            traj_df,
            conj_full,
            sequence_length=20,
        )

        if X is not None:
            # Save training data
            np.save(PROCESSED_DIR / "X_sequences.npy", X)
            np.save(PROCESSED_DIR / "y_labels.npy",    y)

            with open(PROCESSED_DIR / "metadata.json", "w") as f:
                json.dump(meta[:100], f, indent=2)

            print(f"\n✅ Training data saved!")
            print(f"   X_sequences.npy:  shape {X.shape}")
            print(f"   y_labels.npy:     shape {y.shape}")
            print(f"\nStage 1 COMPLETE — your dataset is ready!")
            print(f"These files feed directly into your"
                  f" CNN+LSTM model in Step 5.")

    print("\n✅ Step 4 Complete!")