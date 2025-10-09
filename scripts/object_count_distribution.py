#!/usr/bin/env python3
"""
Calculate object count distribution for 3D-FRONT dataset (bedroom scenes)
to use for penalty term |n_objects - expected_count|²
"""

import os
import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

from threed_front.datasets import get_raw_dataset
from threed_front.datasets.threed_front import CachedThreedFront, ThreedFront


def plot_distribution(counts, expected_count):
    """
    Plot the distribution of object counts
    """
    plt.figure(figsize=(12, 6))
    count_hist = Counter(counts)
    
    # Plot histogram
    keys = sorted(count_hist.keys())
    values = [count_hist[k] for k in keys]
    
    bars = plt.bar(keys, values, alpha=0.7)
    plt.axvline(expected_count, color='r', linestyle='--', label=f'Expected count: {expected_count:.2f}')
    
    # Annotate bars with counts
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')
    
    plt.xlabel('Number of Objects')
    plt.ylabel('Frequency')
    plt.title('Object Count Distribution')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.savefig('object_count_distribution.png')
    print("Plot saved as 'object_count_distribution.png'")


def main():
    """
    Main function to calculate object count distribution for 3D-FRONT dataset
    """
    # Hardcoded configuration for bedroom dataset
    path_to_dataset_files = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/dataset_files"
    path_to_processed_data = "/mnt/sv-share/MiData/"
    dataset_directory = os.path.join(path_to_processed_data, "bedroom")
    dataset_type = "cached_threedfront"
    encoding_type = "cached_diffusion_cosin_angle_wocm_no_prm"
    annotation_file = os.path.join(path_to_dataset_files, "bedroom_threed_front_splits_original.csv")
    room_type = "bedroom"
    
    # Set up the config dictionary - note that get_raw_dataset expects top-level keys
    config = {
        "path_to_processed_data": path_to_processed_data,
        "path_to_dataset_files": path_to_dataset_files,
        "dataset_directory": dataset_directory,
        "dataset_type": dataset_type,
        "encoding_type": encoding_type,
        "annotation_file": annotation_file,
        "room_layout_size": "64,64",
        "room_type": room_type,
        "augmentations": ["fixed_rotations"],
        "train_stats": "dataset_stats.txt"
    }
    
    print("Loading training dataset...")
    # Get the dataset specifically for the train split
    dataset = get_raw_dataset(config, split=["train"])

    # Calculate object counts per scene
    print(f"Calculating object counts for {len(dataset)} scenes...")
    object_counts = []
    
    for i in tqdm(range(len(dataset))):
        # Get room parameters which contain translations, sizes, etc.
        room_params = dataset.get_room_params(i)
        if "translations" in room_params:
            object_counts.append(len(room_params["translations"]))
        else:
            # Fallback: try to get scene object
            scene = dataset[i]
            if hasattr(scene, 'translations'):
                object_counts.append(len(scene.translations))
            else:
                print(f"Warning: Could not get object count for scene {i}")
    
    if len(object_counts) == 0:
        print("Error: No object counts could be extracted from the dataset!")
        return
    
    # Calculate statistics
    expected_count = np.mean(object_counts)
    median_count = np.median(object_counts)
    std_count = np.std(object_counts)
    min_count = np.min(object_counts)
    max_count = np.max(object_counts)
    
    # Print results
    print("\n--- Object Count Distribution Statistics ---")
    print(f"Total scenes: {len(object_counts)}")
    print(f"Mean object count: {expected_count:.2f}")
    print(f"Median object count: {median_count}")
    print(f"Std deviation: {std_count:.2f}")
    print(f"Min count: {min_count}")
    print(f"Max count: {max_count}")
    
    # Print histogram
    count_hist = Counter(object_counts)
    print("\nObject count histogram:")
    for n, freq in sorted(count_hist.items()):
        print(f"  {n} objects: {freq} scenes ({freq/len(object_counts)*100:.2f}%)")
    
    # Calculate penalty for different object counts
    print("\nExample penalty values (|n_objects - expected_count|²):")
    for n in range(min_count, max_count + 1, max(1, int((max_count-min_count)/10))):
        penalty = (n - expected_count) ** 2
        print(f"  {n} objects: penalty = {penalty:.2f}")
    
    # Plot the distribution
    print("\nPlotting distribution...")
    plot_distribution(object_counts, expected_count)
    
    # Save the distribution data to file
    results = {
        "object_counts": object_counts,
        "count_histogram": dict(count_hist),
        "expected_count": expected_count,
        "median_count": median_count,
        "std_deviation": std_count,
        "min_count": min_count,
        "max_count": max_count
    }
    
    with open('object_count_distribution.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Distribution data saved to 'object_count_distribution.pkl'")
    
    # Print penalty formula for use in models
    print("\nUse this penalty term in your model:")
    print(f"penalty = |n_objects - {expected_count:.2f}|²")


if __name__ == "__main__":
    main()