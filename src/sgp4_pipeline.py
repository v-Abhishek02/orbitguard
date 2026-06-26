"""
ORBITGUARD — Stage 1: SGP4 Orbital Propagation
File: src/sgp4_pipeline.py
Purpose: Convert TLE data into real X,Y,Z orbital trajectories using SGP4
"""

import json
import math
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sgp4.api import Satrec, jday
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
GM        = 3.986004418e14   # Earth gravitational parameter (m^3/s^2)
R_EARTH   = 6371.0           # Earth radius (km)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD AND VALIDATE RAW TLE DATA
# ═════════════════════════════════════════════════════════════════════════════

def load_tle_data(filepath):
    """
    Load the JSON file downloaded from Space-Track.
    Validate every record has both TLE lines before using it.
    """
    print(f"Loading TLE data from {filepath}...")
    with open(filepath) as f:
        raw = json.load(f)

    print(f"  Total records loaded:    {len(raw)}")

    # Keep only records with both TLE lines
    valid = [r for r in raw
             if r.get("TLE_LINE1") and r.get("TLE_LINE2")]

    invalid = len(raw) - len(valid)
    print(f"  Valid (have both lines): {len(valid)}")
    print(f"  Invalid (missing lines): {invalid}")

    return valid


def parse_epoch_to_datetime(epoch_str):
    """
    Convert Space-Track epoch string to year, month, day, hour, min, sec.
    Space-Track format: '2024-01-15T06:30:00'
    """
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(epoch_str)
        return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    except Exception:
        # fallback to Jan 1 2024 noon if parsing fails
        return 2024, 1, 1, 12, 0, 0


def compute_altitude_from_mean_motion(mean_motion_revday):
    """
    Convert mean motion (rev/day) to approximate altitude (km).
    Uses Kepler's Third Law.
    This is used for filtering and understanding your data.
    """
    if mean_motion_revday <= 0:
        return None
    T_seconds = 86400.0 / mean_motion_revday      # orbital period in seconds
    a = (GM * (T_seconds / (2 * math.pi)) ** 2) ** (1/3)  # semi-major axis m
    altitude_km = (a / 1000.0) - R_EARTH
    return round(altitude_km, 1)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: SGP4 PROPAGATION
# ═════════════════════════════════════════════════════════════════════════════

def propagate_single_object(norad_id, tle1, tle2,
                             start_year=2024, start_month=1,
                             start_day=1, duration_minutes=2880,
                             step_minutes=1):
    """
    Propagate one TLE object forward in time using SGP4.

    Returns a list of dicts, one per timestep, each containing:
        id, t (minutes from start), x, y, z (km), vx, vy, vz (km/s)

    duration_minutes = 2880 = 48 hours of trajectory
    step_minutes     = 1    = one position per minute
    """
    try:
        # Create satellite object from TLE strings
        sat = Satrec.twoline2rv(tle1, tle2)
    except Exception as e:
        return []   # bad TLE — skip this object

    rows = []
    error_count = 0

    for t in range(0, duration_minutes, step_minutes):
        # Convert elapsed minutes to Julian Date
        # t is total minutes elapsed from start time
        total_minutes = (start_day - 1) * 1440 + t   # minutes from Jan 1
        hour    = total_minutes // 60
        minute  = total_minutes % 60

        jd, fr = jday(start_year, start_month, 1, hour, minute, 0)

        # Run SGP4 propagation
        # e = error code (0 = success, non-zero = problem)
        # r = position vector [x, y, z] in km (TEME frame)
        # v = velocity vector [vx, vy, vz] in km/s (TEME frame)
        e, r, v = sat.sgp4(jd, fr)

        if e != 0:
            # SGP4 error — object may have decayed or bad elements
            error_count += 1
            if error_count > 10:
                break   # too many errors — skip rest of this object
            continue

        # Compute distance from Earth centre (should be > R_EARTH)
        dist = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        if dist < R_EARTH:
            break   # object has re-entered — stop propagating

        rows.append({
            "id":  norad_id,
            "t":   t,           # minutes from start
            "x":   r[0],        # km, TEME frame
            "y":   r[1],        # km, TEME frame
            "z":   r[2],        # km, TEME frame
            "vx":  v[0],        # km/s
            "vy":  v[1],        # km/s
            "vz":  v[2],        # km/s
        })

    return rows


def propagate_all_objects(tle_records,
                           duration_minutes=2880,
                           step_minutes=1,
                           max_objects=None):
    """
    Propagate ALL debris objects and collect into one large DataFrame.

    For 20,000 objects × 2880 timesteps = ~57 million rows.
    This takes 15-25 minutes on a MacBook — be patient.

    max_objects: set a number (e.g. 500) for quick testing first.
    """

    if max_objects:
        tle_records = tle_records[:max_objects]
        print(f"  (Testing mode: processing only {max_objects} objects)")

    print(f"\nPropagating {len(tle_records)} objects × "
          f"{duration_minutes} timesteps...")
    print(f"Estimated time: "
          f"{'~2 min (test)' if max_objects else '~20 min (full)'}")
    print("Progress bar:")

    all_rows = []
    failed   = 0

    for record in tqdm(tle_records, unit="obj"):
        norad_id = record.get("NORAD_CAT_ID", "UNKNOWN")
        tle1     = record["TLE_LINE1"]
        tle2     = record["TLE_LINE2"]

        rows = propagate_single_object(
            norad_id, tle1, tle2,
            duration_minutes=duration_minutes,
            step_minutes=step_minutes
        )

        if len(rows) == 0:
            failed += 1
        else:
            all_rows.extend(rows)

    print(f"\nPropagation complete!")
    print(f"  Objects propagated:  {len(tle_records) - failed}")
    print(f"  Objects failed/bad:  {failed}")
    print(f"  Total rows created:  {len(all_rows):,}")

    df = pd.DataFrame(all_rows)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: COMPUTE ORBITAL FEATURES
# ═════════════════════════════════════════════════════════════════════════════

def add_orbital_features(df):
    """
    Add derived features to each position row.
    These will become input features for your CNN+LSTM detector.

    Features added:
    - altitude:   height above Earth surface (km)
    - speed:      total velocity magnitude (km/s)
    - range_rate: rate of change of distance from Earth centre (km/s)
    """
    print("\nComputing orbital features...")

    # Distance from Earth centre (km)
    df["dist"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)

    # Altitude above Earth surface (km)
    df["altitude"] = df["dist"] - R_EARTH

    # Total speed (km/s)
    df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)

    # Range rate — how fast altitude is changing (km/s)
    # Positive = moving away from Earth, Negative = falling toward Earth
    df["range_rate"] = (
        df["x"] * df["vx"] +
        df["y"] * df["vy"] +
        df["z"] * df["vz"]
    ) / df["dist"]

    # Specific orbital energy (km^2/s^2) — negative means bound orbit
    GM_km = GM / 1e9   # convert to km^3/s^2
    df["energy"] = (df["speed"]**2 / 2) - (GM_km / df["dist"])

    print(f"  Altitude range:  "
          f"{df['altitude'].min():.1f} — {df['altitude'].max():.1f} km")
    print(f"  Speed range:     "
          f"{df['speed'].min():.3f} — {df['speed'].max():.3f} km/s")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: ANALYSE THE DATASET
# ═════════════════════════════════════════════════════════════════════════════

def analyse_dataset(df):
    """
    Print a full statistical analysis of your trajectory dataset.
    This is your Exploratory Data Analysis (EDA).
    Write these numbers down — they go in your paper.
    """
    print("\n" + "="*60)
    print("DATASET ANALYSIS REPORT")
    print("="*60)

    n_objects = df["id"].nunique()
    n_rows    = len(df)
    n_times   = df["t"].nunique()

    print(f"Unique debris objects:   {n_objects:,}")
    print(f"Total trajectory rows:   {n_rows:,}")
    print(f"Timesteps per object:    {n_times}")
    print(f"Time coverage:           {n_times} minutes = "
          f"{n_times/60:.1f} hours")

    print(f"\nAltitude statistics (km):")
    print(f"  Min:    {df['altitude'].min():.1f}")
    print(f"  Max:    {df['altitude'].max():.1f}")
    print(f"  Mean:   {df['altitude'].mean():.1f}")
    print(f"  Median: {df['altitude'].median():.1f}")

    print(f"\nSpeed statistics (km/s):")
    print(f"  Min:    {df['speed'].min():.3f}")
    print(f"  Max:    {df['speed'].max():.3f}")
    print(f"  Mean:   {df['speed'].mean():.3f}")

    # Altitude zone breakdown
    leo_low  = df[df["altitude"] < 600]["id"].nunique()
    leo_mid  = df[(df["altitude"] >= 600) &
                  (df["altitude"] < 1200)]["id"].nunique()
    leo_high = df[df["altitude"] >= 1200]["id"].nunique()

    print(f"\nAltitude zones (unique objects):")
    print(f"  LEO Low  (200-600 km):   {leo_low:,}")
    print(f"  LEO Mid  (600-1200 km):  {leo_mid:,}")
    print(f"  LEO High (1200-2000 km): {leo_high:,}")

    print("="*60)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: VISUALISE ONE ORBIT
# ═════════════════════════════════════════════════════════════════════════════

def visualise_single_orbit(df, norad_id=None):
    """
    Plot the 3D orbit of one debris object.
    This is your first visual proof that SGP4 is working correctly.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Pick first object if none specified
    if norad_id is None:
        norad_id = df["id"].iloc[0]

    obj_df = df[df["id"] == norad_id].copy()

    print(f"\nPlotting orbit for object: {norad_id}")
    print(f"  Timesteps: {len(obj_df)}")
    print(f"  Altitude range: "
          f"{obj_df['altitude'].min():.1f} — "
          f"{obj_df['altitude'].max():.1f} km")

    fig = plt.figure(figsize=(12, 5))

    # ── Plot 1: 3D orbit ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(121, projection='3d')

    # Draw Earth sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    xe = R_EARTH * np.outer(np.cos(u), np.sin(v))
    ye = R_EARTH * np.outer(np.sin(u), np.sin(v))
    ze = R_EARTH * np.outer(np.ones(50), np.cos(v))
    ax1.plot_surface(xe, ye, ze, color='royalblue',
                     alpha=0.4, linewidth=0)

    # Draw orbit
    ax1.plot(obj_df["x"], obj_df["y"], obj_df["z"],
             'r-', linewidth=0.8, alpha=0.8)

    # Mark start point
    ax1.scatter(obj_df["x"].iloc[0],
                obj_df["y"].iloc[0],
                obj_df["z"].iloc[0],
                color='green', s=50, zorder=5, label='Start')

    ax1.set_title(f"Orbit: NORAD {norad_id}")
    ax1.set_xlabel("X (km)")
    ax1.set_ylabel("Y (km)")
    ax1.set_zlabel("Z (km)")
    ax1.legend()

    # ── Plot 2: Altitude over time ────────────────────────────────────────
    ax2 = fig.add_subplot(122)
    ax2.plot(obj_df["t"] / 60,   # convert minutes to hours
             obj_df["altitude"],
             'b-', linewidth=1.2)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Altitude (km)")
    ax2.set_title(f"Altitude over 48 hours: NORAD {norad_id}")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=obj_df["altitude"].mean(),
                color='r', linestyle='--', alpha=0.5,
                label=f'Mean: {obj_df["altitude"].mean():.1f} km')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("outputs/sample_orbit.png", dpi=150, bbox_inches='tight')
    print("  Saved plot: outputs/sample_orbit.png")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("="*60)
    print("ORBITGUARD — SGP4 Trajectory Propagation")
    print("="*60)

    # ── Load TLE data ─────────────────────────────────────────────────────
    tle_records = load_tle_data(RAW_DIR / "leo_debris_tle.json")

    # ── PHASE 1: Quick test with 200 objects first ─────────────────────────
    # This runs in ~1 minute and confirms everything works
    # before committing to the full 20,000 object run
    print("\n--- PHASE 1: Quick test (200 objects) ---")
    df_test = propagate_all_objects(
        tle_records,
        duration_minutes=180,    # 3 hours only for test
        step_minutes=1,
        max_objects=200
    )

    df_test = add_orbital_features(df_test)
    analyse_dataset(df_test)
    visualise_single_orbit(df_test)

    # Save test result
    df_test.to_parquet(
        PROCESSED_DIR / "trajectories_test_200.parquet",
        index=False
    )
    test_file = PROCESSED_DIR / "trajectories_test_200.parquet"
    print(f"\nTest data saved: {test_file}")
    print(f"File size: {test_file.stat().st_size / 1e6:.2f} MB")
    
    # ── PHASE 2: Ask user before running full dataset ──────────────────────
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE — Test run successful!")
    print("="*60)
    print("\nReady for FULL dataset propagation:")
    print(f"  Objects: ~{len(tle_records):,}")
    print(f"  Duration: 48 hours per object")
    print(f"  Expected rows: ~{len(tle_records)*2880:,}")
    print(f"  Estimated time: 15-25 minutes")
    print(f"  Output size: ~200-400 MB")

    answer = input("\nRun full propagation now? (yes/no): ").strip().lower()

    if answer == "yes":
        print("\n--- PHASE 2: Full propagation ---")
        df_full = propagate_all_objects(
            tle_records,
            duration_minutes=2880,   # 48 hours
            step_minutes=1,
            max_objects=None         # ALL objects
        )

        df_full = add_orbital_features(df_full)
        analyse_dataset(df_full)

        # Save full dataset
        df_full.to_parquet(
            PROCESSED_DIR / "trajectories_full.parquet",
            index=False
        )

        size = (PROCESSED_DIR/"trajectories_full.parquet").stat().st_size
        print(f"\n✅ Full trajectory dataset saved!")
        print(f"   File: data/processed/trajectories_full.parquet")
        print(f"   Size: {size/1e6:.1f} MB")
        print(f"   Rows: {len(df_full):,}")

    else:
        print("\nSkipped full run. You can run it later.")
        print("The test file is saved and ready for Step 4 testing.")

    print("\n✅ Step 3 Complete!")