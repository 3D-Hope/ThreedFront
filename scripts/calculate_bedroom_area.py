"""Compute the distribution of bedroom floor areas (in square meters).

This script loads the 3D-FRONT dataset configuration from a results.pkl
(ThreedFrontResults) file and computes the area of each room in the split
using the floor plan boundary points (fpbpn). It prints summary statistics
(mean, median, std, percentiles) and can optionally save all areas to CSV.

Usage:
    python scripts/calculate_bedroom_area.py /path/to/sampled_scenes_results.pkl \
        [--splits test] [--output_csv bedroom_areas.csv]

Notes:
- Area is computed on the X-Z plane using the polygon shoelace formula.
- Coordinates are assumed to be in meters; area is reported in m^2.
"""
import argparse
import csv
import pickle
import numpy as np
from pathlib import Path

from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults


def polygon_area_shoelace(points: np.ndarray) -> float:
    """Compute area of a simple polygon via the shoelace formula.
    points: array of shape [N, 2] ordered along boundary (closed or open)
    returns a positive scalar area
    """
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    # Roll for vectorized summation of x_i*y_{i+1} - x_{i+1}*y_i
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


def main():
    parser = argparse.ArgumentParser(
        description="Compute room area distribution (m^2) for 3D-FRONT bedrooms"
    )
    parser.add_argument(
        "result_file",
        help="Path to a pickled result file (ThreedFrontResults object)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Dataset splits to load (default: config.validation.splits or ['test'])"
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional CSV file to save per-room areas"
    )
    args = parser.parse_args()

    # Load saved results to access dataset config
    result_path = Path(args.result_file)
    with result_path.open("rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)

    # Load dataset
    config = threed_front_results.config
    splits = args.splits or config.get("validation", {}).get("splits", ["test"])
    raw_dataset = get_raw_dataset(
        config["data"],
        split=splits,
        include_room_mask=True,
    )

    areas_m2 = []
    missing = 0

    for i in range(len(raw_dataset)):
        room_params = raw_dataset.get_room_params(i)
        if "fpbpn" not in room_params:
            missing += 1
            continue
        # Use boundary points projected on XZ plane
        boundary = room_params["fpbpn"][:, :2].astype(np.float64)
        area = polygon_area_shoelace(boundary)
        areas_m2.append(area)

    areas = np.array(areas_m2, dtype=np.float64)

    if areas.size == 0:
        print("No areas computed (missing fpbpn in all rooms). Aborting.")
        return

    # Summary statistics
    mean = float(np.mean(areas))
    median = float(np.median(areas))
    std = float(np.std(areas))
    min_v = float(np.min(areas))
    max_v = float(np.max(areas))
    p25, p75 = float(np.percentile(areas, 25)), float(np.percentile(areas, 75))
    p90, p95 = float(np.percentile(areas, 90)), float(np.percentile(areas, 95))

    print("\nRoom area distribution (m^2)")
    print(f"  Split(s): {splits}")
    print(f"  Rooms counted: {len(areas)} (missing fpbpn: {missing})")
    print(f"  Mean:   {mean:.2f} m^2")
    print(f"  Median: {median:.2f} m^2")
    print(f"  Std:    {std:.2f} m^2")
    print(f"  Min:    {min_v:.2f} m^2  |  Max: {max_v:.2f} m^2")
    print(f"  P25:    {p25:.2f}  |  P75: {p75:.2f}")
    print(f"  P90:    {p90:.2f}  |  P95: {p95:.2f}")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["area_m2"])  # header
            for a in areas:
                writer.writerow([f"{a:.6f}"])
        print(f"\nSaved per-room areas to {args.output_csv}")


if __name__ == "__main__":
    main()
