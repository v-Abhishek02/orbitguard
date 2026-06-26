"""
ORBITGUARD — Stage 1: Data Collection
File: src/download_data.py
Purpose: Connect to Space-Track.org API and download real LEO debris TLE data
"""

import os
import json
import time
import requests
from dotenv import load_dotenv
from pathlib import Path

# ── Load credentials from .env file ──────────────────────────────────────────
load_dotenv()
USERNAME = os.getenv("SPACETRACK_USER")
PASSWORD = os.getenv("SPACETRACK_PASS")

# ── Space-Track API base URL ──────────────────────────────────────────────────
BASE_URL = "https://www.space-track.org"
LOGIN_URL = BASE_URL + "/ajaxauth/login"

# ── Output path ───────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def login(username, password):
    """
    Log in to Space-Track and return an authenticated session.
    The session object keeps the login cookie for all future requests.
    """
    print("Connecting to Space-Track.org...")
    session = requests.Session()

    payload = {
        "identity": username,
        "password": password
    }

    response = session.post(LOGIN_URL, data=payload)

    # Check login succeeded
    if "Failed" in response.text or response.status_code != 200:
        print("LOGIN FAILED — check your .env credentials")
        print(f"Response: {response.text[:200]}")
        return None

    print("Login successful!")
    return session


def fetch_leo_debris(session):
    """
    Download all LEO debris objects with recent TLE data.
    
    Query explanation:
    - OBJECT_TYPE/DEBRIS      → only debris objects (not active satellites)
    - MEAN_MOTION/>11         → mean motion > 11 rev/day = orbital period < 130 min = LEO
    - EPOCH/>now-30           → TLEs updated in last 30 days (reasonably fresh)
    - format/json             → get data as JSON
    """
    print("\nFetching LEO debris TLE data...")
    print("This may take 1-2 minutes depending on your internet speed...")

    query_url = (
        BASE_URL
        + "/basicspacedata/query"
        + "/class/gp"
        + "/OBJECT_TYPE/DEBRIS"
        + "/MEAN_MOTION/>11"
        + "/EPOCH/>now-30"
        + "/format/json"
    )

    response = session.get(query_url)

    if response.status_code != 200:
        print(f"Query failed with status: {response.status_code}")
        print(f"Response: {response.text[:300]}")
        return None

    data = response.json()
    print(f"Downloaded {len(data)} debris objects!")
    return data


def fetch_sample_active_satellites(session):
    """
    Download a small sample of active satellites for comparison.
    Useful for testing and understanding normal vs debris orbits.
    """
    print("\nFetching sample active satellites...")

    query_url = (
        BASE_URL
        + "/basicspacedata/query"
        + "/class/gp"
        + "/OBJECT_TYPE/PAYLOAD"
        + "/MEAN_MOTION/>11"
        + "/EPOCH/>now-7"
        + "/DECAY_DATE/null-val"   # only objects that have NOT re-entered
        + "/orderby/NORAD_CAT_ID asc"
        + "/limit/500"             # just 500 for now
        + "/format/json"
    )

    response = session.get(query_url)

    if response.status_code != 200:
        print(f"Query failed: {response.status_code}")
        return None

    data = response.json()
    print(f"Downloaded {len(data)} active satellites!")
    return data


def save_data(data, filename):
    """Save downloaded data as JSON to data/raw/"""
    filepath = RAW_DATA_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Saved: {filepath}  ({size_mb:.2f} MB)")
    return filepath


def inspect_record(record):
    """Print all fields of one TLE record so you understand the data."""
    print("\n" + "="*60)
    print("SAMPLE RECORD — one debris object from Space-Track")
    print("="*60)
    for key, value in record.items():
        print(f"  {key:<30} {value}")
    print("="*60)


def extract_tle_fields(record):
    """
    Extract just the fields we need for SGP4 propagation.
    Shows you exactly what your pipeline will use.
    """
    return {
        "norad_id":    record.get("NORAD_CAT_ID"),
        "name":        record.get("OBJECT_NAME"),
        "object_type": record.get("OBJECT_TYPE"),
        "epoch":       record.get("EPOCH"),
        "inclination": float(record.get("INCLINATION", 0)),
        "eccentricity":float(record.get("ECCENTRICITY", 0)),
        "mean_motion": float(record.get("MEAN_MOTION", 0)),
        "bstar":       float(record.get("BSTAR", 0)),
        "tle_line1":   record.get("TLE_LINE1"),
        "tle_line2":   record.get("TLE_LINE2"),
    }


def run_data_quality_check(data):
    """
    Check the downloaded data for quality issues.
    Always do this before building models.
    """
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)

    total = len(data)
    print(f"Total records:          {total}")

    # Check for missing TLE lines
    missing_tle1 = sum(1 for r in data if not r.get("TLE_LINE1"))
    missing_tle2 = sum(1 for r in data if not r.get("TLE_LINE2"))
    print(f"Missing TLE_LINE1:      {missing_tle1}")
    print(f"Missing TLE_LINE2:      {missing_tle2}")

    # Altitude distribution
    mean_motions = [float(r.get("MEAN_MOTION", 0)) for r in data
                    if r.get("MEAN_MOTION")]

    # Convert mean motion to approximate altitude
    import math
    GM = 3.986004418e14  # m^3/s^2
    altitudes = []
    for mm in mean_motions:
        if mm > 0:
            T = 86400 / mm          # period in seconds
            a = (GM * (T/(2*math.pi))**2) ** (1/3)  # semi-major axis in m
            alt = (a / 1000) - 6371  # altitude in km
            altitudes.append(alt)

    if altitudes:
        print(f"Altitude range:         {min(altitudes):.0f} km — {max(altitudes):.0f} km")
        print(f"Average altitude:       {sum(altitudes)/len(altitudes):.0f} km")

    # Inclination distribution
    incls = [float(r.get("INCLINATION", 0)) for r in data if r.get("INCLINATION")]
    if incls:
        print(f"Inclination range:      {min(incls):.1f}° — {max(incls):.1f}°")

    # Object name check
    named = sum(1 for r in data if r.get("OBJECT_NAME") and
                r.get("OBJECT_NAME") != "TBA - TO BE ASSIGNED")
    print(f"Named objects:          {named} / {total}")
    print(f"Unnamed/TBA:            {total - named} / {total}")
    print("="*60)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("="*60)
    print("ORBITGUARD — Stage 1: Data Download")
    print("="*60)

    # Check credentials loaded
    if not USERNAME or not PASSWORD:
        print("ERROR: .env file not found or credentials missing")
        print("Make sure .env exists in your project root with:")
        print("  SPACETRACK_USER=your@email.com")
        print("  SPACETRACK_PASS=yourpassword")
        exit(1)

    print(f"Using account: {USERNAME}")

    # Step 1: Login
    session = login(USERNAME, PASSWORD)
    if session is None:
        exit(1)

    # Step 2: Download debris TLE data
    debris_data = fetch_leo_debris(session)
    if debris_data is None:
        print("Failed to fetch debris data")
        exit(1)

    # Step 3: Save raw debris data
    debris_file = save_data(debris_data, "leo_debris_tle.json")

    # Step 4: Show one sample record so you understand the structure
    if debris_data:
        inspect_record(debris_data[0])

    # Step 5: Download sample active satellites
    time.sleep(2)  # be polite to the API
    # sat_data = fetch_active_satellites(session)
    sat_data = fetch_sample_active_satellites(session)
    if sat_data:
        save_data(sat_data, "active_satellites_sample.json")

    # Step 6: Data quality check
    run_data_quality_check(debris_data)

    # Step 7: Show extracted fields for first 3 records
    print("\nEXTRACTED FIELDS FOR FIRST 3 DEBRIS OBJECTS:")
    print("-"*60)
    for i, record in enumerate(debris_data[:3]):
        extracted = extract_tle_fields(record)
        print(f"\nObject {i+1}:")
        for k, v in extracted.items():
            print(f"  {k:<20} {v}")

    print("\n✅ Stage 1 Data Download Complete!")
    print(f"   Debris objects downloaded: {len(debris_data)}")
    print(f"   Saved to: data/raw/leo_debris_tle.json")