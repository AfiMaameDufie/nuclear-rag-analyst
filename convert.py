"""
convert.py — Convert global_nuclear_energy_intelligence_1965_2025.csv
into one JSON file per row, stored in data/.

Usage:
    python convert.py
"""

import csv
import json
import os
import math
from pathlib import Path

CSV_PATH = Path("global_nuclear_energy_intelligence_1965_2025.csv")
OUT_DIR = Path("data")
YEAR_FROM = 2015  # only export this year onwards to keep ingestion fast


def clean(value: str):
    """Return a float, int, bool, or string; skip NaN/empty."""
    if value == "" or value is None:
        return None
    # booleans stored as 0/1
    if value in ("0", "1"):
        try:
            return int(value)
        except ValueError:
            pass
    try:
        f = float(value)
        if math.isnan(f):
            return None
        # return int if it is a whole number
        if f == int(f) and abs(f) < 1e12:
            return int(f)
        return round(f, 6)
    except ValueError:
        return value.strip()


def main():
    OUT_DIR.mkdir(exist_ok=True)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            doc = {k: clean(v) for k, v in row.items()}

            # Skip rows outside the target year range
            try:
                if int(doc.get("year", 0)) < YEAR_FROM:
                    continue
            except (TypeError, ValueError):
                pass

            country = (doc.get("country") or "unknown").replace("/", "-").replace(" ", "_")
            year = doc.get("year", "unknown")
            filename = OUT_DIR / f"{country}_{year}.json"

            with open(filename, "w", encoding="utf-8") as out:
                json.dump(doc, out, ensure_ascii=False, indent=2)
            count += 1

    print(f"✅  Wrote {count} JSON files to {OUT_DIR}/")


if __name__ == "__main__":
    main()
