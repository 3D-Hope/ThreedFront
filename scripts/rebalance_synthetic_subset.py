"""
Rebalance synthesized scenes to better match ground-truth class distribution.
- Loads a ThreedFrontResults pickle (result_file)
- Computes GT distribution over train+val(if present)+test (excluding 'end')
- Computes per-scene class counts via argmax (one class per object)
- Greedily selects a subset of scenes to match GT counts up to target_size
- Saves a new ThreedFrontResults pickle with the selected subset
"""

import argparse
import os
import pickle
import numpy as np
from typing import List

from threed_front.evaluation import ThreedFrontResults


def to_prob(x: np.ndarray) -> np.ndarray:
    return (x + 1) / 2 if (x.min() < 0) else x


def compute_gt_counts(results: ThreedFrontResults) -> np.ndarray:
    train_ds = results.train_dataset
    test_ds = results.test_dataset
    n_object_types = results.n_object_types

    ds_list = [train_ds, test_ds]
    val_ds = getattr(results, "val_dataset", None)
    if val_ds is not None:
        ds_list = [train_ds, val_ds, test_ds]

    gt_counts = np.zeros(n_object_types, dtype=np.float64)
    for ds in ds_list:
        for i in range(len(ds)):
            L = ds.get_room_params(i)["class_labels"]
            L = to_prob(L)
            has_end = (L.shape[1] > n_object_types)
            L_use = L[:, :n_object_types] if has_end else L
            if has_end:
                valid = (L[:, -1] != 1)
                L_use = L_use[valid]
            gt_counts += L_use.sum(0)
    return gt_counts


def scene_class_counts(layout: dict, n_object_types: int) -> np.ndarray:
    L = layout["class_labels"]  # [N, C(+1?)]
    has_end = (L.shape[1] > n_object_types)
    L_use = L[:, :n_object_types] if has_end else L
    valid_mask = (L_use.sum(axis=-1) > 1e-6)
    L_use = L_use[valid_mask]
    if L_use.shape[0] == 0:
        return np.zeros(n_object_types, dtype=np.int64)
    cls_idx = L_use.argmax(axis=-1)
    return np.bincount(cls_idx, minlength=n_object_types).astype(np.int64)


def greedy_rebalance(target_counts: np.ndarray,
                     per_scene_counts: List[np.ndarray],
                     max_scenes: int) -> List[int]:
    """Deficit-weighted selection. Stops when no positive improvement remains."""
    selected: List[int] = []
    current = np.zeros_like(target_counts, dtype=np.int64)
    remaining = set(range(len(per_scene_counts)))

    while len(selected) < max_scenes and remaining:
        deficit = (target_counts - current).clip(min=0).astype(np.float64)
        if deficit.sum() == 0:
            break
        deficit_norm = deficit / (deficit.sum() + 1e-9)

        best_idx = -1
        best_score = -1.0
        for i in remaining:
            sc = per_scene_counts[i].astype(np.float64)
            if sc.sum() == 0:
                continue
            sc_norm = sc / (sc.sum() + 1e-9)
            # Higher score if scene fills current deficits
            score = float((deficit_norm * sc_norm).sum())
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0 or best_score <= 0:
            break  # no positive help left

        selected.append(best_idx)
        current += per_scene_counts[best_idx]
        remaining.remove(best_idx)

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", help="Path to ThreedFrontResults pickle")
    parser.add_argument("output_file", help="Path to save balanced results pickle")
    parser.add_argument("--target_size", type=int, default=5000,
                        help="Number of scenes to select (default: 5000)")
    args = parser.parse_args()

    with open(args.result_file, "rb") as f:
        results: ThreedFrontResults = pickle.load(f)
    assert isinstance(results, ThreedFrontResults)

    n_object_types = results.n_object_types

    # GT distribution
    gt_counts = compute_gt_counts(results)
    if gt_counts.sum() == 0:
        raise RuntimeError("GT counts sum to zero. Check datasets in results.")
    gt_freq = gt_counts / gt_counts.sum()

    # Target counts for subset
    target_counts = np.round(gt_freq * args.target_size).astype(np.int64)

    # Build per-scene counts
    per_scene_counts = []
    for _, layout in results:
        per_scene_counts.append(scene_class_counts(layout, n_object_types))

    # Greedy selection (deficit-weighted)
    selected_indices = greedy_rebalance(target_counts, per_scene_counts, args.target_size)

    # If under target, top-up by resampling from already selected (does not drift histogram)
    if 0 < len(selected_indices) < args.target_size:
        # Repeat selected indices to reach target size
        rep = (args.target_size + len(selected_indices) - 1) // len(selected_indices)
        selected_indices = (selected_indices * rep)[:args.target_size]

    if len(selected_indices) == 0:
        # Fallback: select the top-N scenes with highest total objects
        totals = [c.sum() for c in per_scene_counts]
        order = np.argsort(totals)[::-1][:args.target_size]
        selected_indices = list(order)

    # Build new ThreedFrontResults with subset
    sel_scene_indices = [results._scene_indices[i] for i in selected_indices]
    sel_layouts = [results._predicted_layouts[i] for i in selected_indices]
    balanced = ThreedFrontResults(results.train_dataset,
                                  results.test_dataset,
                                  results.config,
                                  sel_scene_indices,
                                  sel_layouts)

    with open(args.output_file, "wb") as f:
        pickle.dump(balanced, f)
    print(f"Saved balanced results to {os.path.abspath(args.output_file)} with {len(sel_scene_indices)} scenes")


if __name__ == "__main__":
    main()
