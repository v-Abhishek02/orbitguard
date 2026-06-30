"""
ORBITGUARD — Complete LEO Catalogue Download
Strategy: Download ALL object types in LEO for maximum density
This is what real STM systems do — track everything
"""

import os
import json
import time
import requests
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timezone

load_dotenv()
USERNAME = os.getenv("SPACETRACK_USER")
PASSWORD = os.getenv("SPACETRACK_PASS")

BASE_URL  = "https://www.space-track.org"
LOGIN_URL = BASE_URL + "/ajaxauth/login"
RAW_DIR   = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def login(username, password):
    print("Logging in to Space-Track...")
    session  = requests.Session()
    response = session.post(
        LOGIN_URL,
        data={"identity": username, "password": password}
    )
    if "Failed" in response.text or response.status_code != 200:
        print("LOGIN FAILED — check credentials")
        return None
    print("✅ Login successful!")
    return session


def fetch_by_object_type(session, object_type, label):
    """Fetch all LEO objects of a specific type."""
    print(f"\nFetching {label}...")
    url = (BASE_URL
           + "/basicspacedata/query"
           + "/class/gp"
           + f"/OBJECT_TYPE/{object_type}"
           + "/MEAN_MOTION/>11.25"   # LEO only (period < 95 min)
           + "/DECAY_DATE/null-val"  # still in orbit
           + "/format/json")
    try:
        response = session.get(url, timeout=120)
        if response.status_code != 200:
            print(f"  Failed: {response.status_code}")
            return []
        data = response.json()
        print(f"  Got {len(data):,} objects")
        return data
    except Exception as e:
        print(f"  Error: {e}")
        return []


def fetch_complete_catalogue(session):
    """
    Download all LEO tracked objects across all categories.
    
    Categories:
    - DEBRIS:   Fragmentation debris, spent hardware pieces
    - ROCKET BODY: Spent upper stages (large, dangerous)
    - PAYLOAD:  Active + dead satellites
    
    Why include all types:
    - Rocket bodies are LARGEST objects — highest collision risk
    - Dead payloads behave exactly like debris
    - Real STM systems track all of these together
    - More objects = denser field = more conjunction events
    - Your paper contribution: first system to use complete catalogue
    """
    all_objects = []
    seen_ids    = set()

    object_types = [
        ("DEBRIS",      "Fragmentation Debris"),
        ("ROCKET BODY", "Spent Rocket Bodies"),
        ("PAYLOAD",     "Satellites (active + dead)"),
    ]

    for obj_type, label in object_types:
        data = fetch_by_object_type(session, obj_type, label)
        new  = 0
        for obj in data:
            obj_id = obj.get("NORAD_CAT_ID")
            if obj_id not in seen_ids:
                # Tag each object with its type for analysis
                obj["CATEGORY"] = obj_type
                seen_ids.add(obj_id)
                all_objects.append(obj)
                new += 1
        print(f"  Added {new:,} unique objects "
              f"(running total: {len(all_objects):,})")
        time.sleep(1)  # rate limit courtesy

    return all_objects


def fetch_starlink_and_megaconstellations(session):
    """
    Starlink, OneWeb, etc. dramatically increase LEO density.
    These are the objects most at risk from debris.
    Including them gives realistic conjunction scenarios.
    """
    print("\nFetching mega-constellation satellites...")
    constellations = []
    seen_ids       = set()

    queries = [
        # Starlink constellation (550 km shell)
        ("/basicspacedata/query/class/gp"
         "/OBJECT_NAME/STARLINK~~"
         "/DECAY_DATE/null-val/format/json",
         "Starlink"),
        # OneWeb constellation
        ("/basicspacedata/query/class/gp"
         "/OBJECT_NAME/ONEWEB~~"
         "/DECAY_DATE/null-val/format/json",
         "OneWeb"),
    ]

    for query, name in queries:
        try:
            print(f"  Fetching {name}...")
            response = session.get(BASE_URL + query, timeout=60)
            if response.status_code == 200:
                data = response.json()
                for obj in data:
                    obj_id = obj.get("NORAD_CAT_ID")
                    if obj_id not in seen_ids:
                        obj["CATEGORY"] = "MEGA_CONSTELLATION"
                        seen_ids.add(obj_id)
                        constellations.append(obj)
                print(f"  Got {len(data):,} {name} satellites")
            time.sleep(1)
        except Exception as e:
            print(f"  {name} failed: {e}")

    return constellations


def quality_report(data):
    """Full quality analysis of downloaded catalogue."""
    print(f"\n{'='*55}")
    print("COMPLETE CATALOGUE QUALITY REPORT")
    print(f"{'='*55}")

    total    = len(data)
    has_tle  = [r for r in data
                if r.get("TLE_LINE1") and r.get("TLE_LINE2")]

    print(f"Total objects:          {total:,}")
    print(f"With valid TLE lines:   {len(has_tle):,}")

    # Category breakdown
    from collections import Counter
    categories = Counter(r.get("CATEGORY", "UNKNOWN") for r in has_tle)
    print(f"\nBy category:")
    for cat, cnt in categories.most_common():
        pct = cnt / len(has_tle) * 100
        print(f"  {cat:<20} {cnt:>6,}  ({pct:.1f}%)")

    # Altitude distribution
    import math
    GM = 3.986004418e14
    R_EARTH = 6371.0
    alts = []
    for r in has_tle:
        try:
            mm  = float(r.get("MEAN_MOTION", 0))
            if mm > 0:
                T   = 86400.0 / mm
                a   = (GM * (T / (2 * math.pi)) ** 2) ** (1/3)
                alt = a / 1000 - R_EARTH
                alts.append(alt)
        except Exception:
            pass

    if alts:
        alts.sort()
        print(f"\nAltitude distribution:")
        print(f"  Range:   {min(alts):.0f} — {max(alts):.0f} km")
        print(f"  Median:  {alts[len(alts)//2]:.0f} km")

        bands = [
            (200,  400,  "Very Low LEO"),
            (400,  600,  "Low LEO (Starlink zone)"),
            (600,  900,  "Mid LEO"),
            (900,  1200, "Upper LEO"),
            (1200, 2000, "High LEO"),
        ]
        for lo, hi, name in bands:
            cnt = sum(1 for a in alts if lo <= a < hi)
            print(f"  {name:<25} ({lo}-{hi} km): {cnt:,}")

    # TLE age
    now = datetime.now(timezone.utc)
    ages = []
    for r in has_tle:
        try:
            ep = datetime.fromisoformat(
                r.get("EPOCH","").replace("Z","+00:00")
            )
            if ep.tzinfo is None:
                ep = ep.replace(tzinfo=timezone.utc)
            ages.append((now - ep).days)
        except Exception:
            pass

    if ages:
        ages.sort()
        print(f"\nTLE freshness:")
        print(f"  Freshest: {min(ages)} days old")
        print(f"  Median:   {ages[len(ages)//2]} days old")
        fresh = sum(1 for a in ages if a <= 30)
        print(f"  < 30 days: {fresh:,} ({fresh/len(ages)*100:.1f}%)")

    print(f"{'='*55}")
    return has_tle


if __name__ == "__main__":

    print("="*55)
    print("ORBITGUARD — Complete LEO Catalogue Download")
    print("Strategy: ALL object types for maximum coverage")
    print("="*55)

    session = login(USERNAME, PASSWORD)
    if not session:
        exit(1)

    # Download all LEO object types
    catalogue = fetch_complete_catalogue(session)

    # Add mega-constellation satellites
    mega = fetch_starlink_and_megaconstellations(session)
    seen = {r["NORAD_CAT_ID"] for r in catalogue}
    for obj in mega:
        if obj["NORAD_CAT_ID"] not in seen:
            catalogue.append(obj)
            seen.add(obj["NORAD_CAT_ID"])

    print(f"\nTotal unique LEO objects: {len(catalogue):,}")

    # Quality report
    valid = quality_report(catalogue)

    # Save
    out = RAW_DIR / "leo_complete_catalogue.json"
    with open(out, "w") as f:
        json.dump(valid, f, indent=2)

    size = out.stat().st_size / 1e6
    print(f"\n✅ Saved: {out}")
    print(f"   Objects: {len(valid):,}")
    print(f"   Size:    {size:.1f} MB")

    if len(valid) >= 20000:
        print(f"\n🎯 TARGET ACHIEVED: {len(valid):,} objects!")
    elif len(valid) >= 15000:
        print(f"\n✅ Good dataset: {len(valid):,} objects")
        print(f"   This is sufficient for strong results")
    else:
        print(f"\n⚠️  {len(valid):,} objects is what Space-Track")
        print(f"   has available for LEO right now.")
        print(f"   This is still a strong research dataset.")