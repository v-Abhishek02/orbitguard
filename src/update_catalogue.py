"""
ORBITGUARD — Live Catalogue Updater
Run this weekly to incorporate newly launched objects.
New objects appear on Space-Track within hours of launch.
"""

import os
import json
import requests
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

load_dotenv()
BASE     = "https://www.space-track.org"
RAW_DIR  = Path("data/raw")


def update_catalogue():
    """
    Download only NEW or UPDATED objects since last run.
    Merges with existing catalogue.
    """
    print(f"Updating catalogue: {datetime.now():%Y-%m-%d %H:%M}")

    s = requests.Session()
    s.post(BASE + "/ajaxauth/login",
           data={"identity": os.getenv("SPACETRACK_USER"),
                 "password": os.getenv("SPACETRACK_PASS")})

    # Load existing catalogue
    cat_path = RAW_DIR / "leo_complete_catalogue.json"
    with open(cat_path) as f:
        existing = json.load(f)
    existing_ids = {r["NORAD_CAT_ID"] for r in existing}
    print(f"  Existing objects: {len(existing):,}")

    # Fetch objects updated in last 7 days
    new_data = []
    for obj_type in ["DEBRIS", "ROCKET BODY", "PAYLOAD"]:
        url = (BASE + "/basicspacedata/query/class/gp"
               + f"/OBJECT_TYPE/{obj_type}"
               + "/MEAN_MOTION/>11"
               + "/EPOCH/>now-7"           # last 7 days
               + "/DECAY_DATE/null-val"
               + "/format/json")
        resp = s.get(url)
        if resp.status_code == 200:
            new_data.extend(resp.json())

    # Find truly new objects (not in existing catalogue)
    brand_new = [r for r in new_data
                 if r.get("NORAD_CAT_ID") not in existing_ids
                 and r.get("TLE_LINE1")
                 and r.get("TLE_LINE2")]

    # Update existing objects with fresh TLEs
    existing_map = {r["NORAD_CAT_ID"]: r for r in existing}
    updated = 0
    for r in new_data:
        oid = r.get("NORAD_CAT_ID")
        if oid in existing_map and r.get("TLE_LINE1"):
            existing_map[oid]["TLE_LINE1"] = r["TLE_LINE1"]
            existing_map[oid]["TLE_LINE2"] = r["TLE_LINE2"]
            existing_map[oid]["EPOCH"]     = r.get("EPOCH","")
            updated += 1

    # Add new objects
    merged = list(existing_map.values()) + brand_new

    print(f"  New objects added:  {len(brand_new):,}")
    print(f"  TLEs updated:       {updated:,}")
    print(f"  Total after update: {len(merged):,}")

    # Save updated catalogue
    with open(cat_path, "w") as f:
        json.dump(merged, f, indent=2)

    # Log the update
    log_path = RAW_DIR / "update_log.json"
    log = []
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
    log.append({
        "date":         datetime.now().isoformat(),
        "new_objects":  len(brand_new),
        "tles_updated": updated,
        "total":        len(merged),
    })
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"  ✅ Catalogue updated!")
    return len(brand_new), updated


if __name__ == "__main__":
    new, updated = update_catalogue()
    print(f"\nRun build_maximum_dataset.py to incorporate changes")