"""Script to compute the KL-divergence between the object categories of real 
and synthetic scenes."""
import argparse
import os
import sys
import pickle
import numpy as np

from threed_front.evaluation.utils import collect_cooccurrence
from threed_front.evaluation import ThreedFrontResults


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the KL-divergence between the object category "
                     "distributions of real and synthesized scenes")
    )
    parser.add_argument(
        "result_file",
        help="Path to a pickled result (ThreedFrontResults object)"
    )
    parser.add_argument(
        "--output_directory",
        default=None,
        help="Output directory to store kl-divergence and other stats (default: None)"
    )

    args = parser.parse_args(argv)

    # Load saved results
    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    print("Received {} synthesized scenes".format(len(threed_front_results)))

    # Datasets and taxonomy
    train_ds = threed_front_results.train_dataset
    test_ds = threed_front_results.test_dataset
    classes = np.array(test_ds.object_types)
    n_object_types = threed_front_results.n_object_types  # exclude 'end'

    # Predicted layouts
    pred_layouts = [threed_front_results[i][1] for i in range(len(threed_front_results))]

    def to_prob(x):
        # Map [-1,1] -> [0,1] if needed, else no-op
        return (x + 1) / 2 if (x.min() < 0) else x

    # Aggregate GT over train + val (if present) + test, exclude 'end' rows
    gt_counts = np.zeros(n_object_types, dtype=np.float64)
    gt_total = 0
    gt_scenes = []

    ds_list = [train_ds, test_ds]
    val_ds = getattr(threed_front_results, "val_dataset", None)
    if val_ds is not None:
        ds_list = [train_ds, val_ds, test_ds]

    for ds in ds_list:
        for i in range(len(ds)):
            L = ds.get_room_params(i)["class_labels"]  # [N, C(+1?)]
            L = to_prob(L)
            has_end = (L.shape[1] > n_object_types)
            L_use = L[:, :n_object_types] if has_end else L
            if has_end:
                valid = (L[:, -1] != 1)
                L_use = L_use[valid]
            gt_counts += L_use.sum(0)
            gt_total += L_use.shape[0]
            gt_scenes.append({"class_labels": L_use})

    # Aggregate SYN via per-object argmax (sampling), exclude clearly empty rows
    syn_counts = np.zeros(n_object_types, dtype=np.float64)
    syn_total = 0
    syn_scenes = []

    for layout in pred_layouts:
        L = layout["class_labels"]  # [N, C(+1?)]
        has_end = (L.shape[1] > n_object_types)
        L_use = L[:, :n_object_types] if has_end else L
        valid_mask = (L_use.sum(axis=-1) > 1e-6)
        L_use = L_use[valid_mask]
        if L_use.shape[0] == 0:
            continue
        cls_idx = L_use.argmax(axis=-1)
        syn_counts += np.bincount(cls_idx, minlength=n_object_types)
        syn_total += len(cls_idx)
        syn_scenes.append({"class_labels": L_use})

    # Remap to PhyScene class order intersection BEFORE computing frequencies and KL
    physcene_order = [
        'wardrobe','nightstand','double_bed','single_bed','kids_bed','desk',
        'dressing_table','table','chair','armchair','bookshelf','cabinet',
        'tv_stand','stool','pendant_lamp','ceiling_lamp','floor_lamp',
        'chest','shelf','coffee_table','dining_table','dining_chair'
    ]
    idx_map = {c: i for i, c in enumerate(classes.tolist())}
    common = [c for c in physcene_order if c in idx_map]
    if len(common) > 0:
        remap_idx = [idx_map[c] for c in common]
        classes_eval = np.array(common)
        gt_counts_eval = gt_counts[remap_idx]
        syn_counts_eval = syn_counts[remap_idx]
    else:
        classes_eval = classes
        gt_counts_eval = gt_counts
        syn_counts_eval = syn_counts

    gt_freq = gt_counts_eval / max(gt_counts_eval.sum(), 1)
    syn_freq = syn_counts_eval / max(syn_counts_eval.sum(), 1)

    def categorical_kl(p, q):
        return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()

    # Debug sums
    print(f"[Ashok] sum of gt_freq: {gt_freq.sum()}, sum of syn_freq: {syn_freq.sum()}")

    kl_divergence = categorical_kl(gt_freq, syn_freq)

    # Print results
    for c, gt_cp, syn_cp in zip(classes_eval, gt_counts_eval, syn_counts_eval):
        print("[{:>18}]: target: {:.4f} / synth: {:.4f}" \
              .format(c, gt_cp/gt_counts_eval.sum(), syn_cp/syn_counts_eval.sum()))
    print("object category kl divergence: {}".format(kl_divergence))

    if args.output_directory is not None:
        # Label co-ocurrences
        # Slice scenes' class_labels to remapped indices to avoid out-of-bounds
        if len(common) > 0:
            gt_scenes_mapped = [{"class_labels": d["class_labels"][:, remap_idx]} for d in gt_scenes]
            syn_scenes_mapped = [{"class_labels": d["class_labels"][:, remap_idx]} for d in syn_scenes]
        else:
            gt_scenes_mapped = gt_scenes
            syn_scenes_mapped = syn_scenes

        gt_cooccurrences = collect_cooccurrence(gt_scenes_mapped, len(classes_eval))
        syn_cooccurrences = collect_cooccurrence(syn_scenes_mapped, len(classes_eval))

        path_to_stats = os.path.join(args.output_directory, "stats.npz")
        np.savez(path_to_stats,
                kl_divergence=kl_divergence, classes=classes_eval,
                gt_classes=gt_counts_eval, syn_classes=syn_counts_eval,
                gt_cooccur=gt_cooccurrences, syn_cooccur=syn_cooccurrences)
        print("Saved stats to {}".format(path_to_stats))

if __name__ == "__main__":
    main(sys.argv[1:])
    
# python scripts/evaluate_kl_divergence_object_category.py  /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/balanced_results.pkl --output_dir ./kl_tmps