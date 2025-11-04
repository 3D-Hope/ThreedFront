#!/usr/bin/env python3
"""
Check how many bounding boxes are axis-aligned vs rotated in 3D-FRONT dataset
"""

import os
import csv
import numpy as np
from tqdm import tqdm


def load_scene_ids_from_csv(csv_path, split="train"):
    """Load scene IDs from the CSV file for a specific split"""
    scene_ids = []
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                scene_id, scene_split = parts
                if scene_split == split:
                    scene_ids.append(scene_id)
    return scene_ids


def main():
    """
    Check axis-aligned vs rotated bounding boxes in the 3D-FRONT bedroom dataset
    """
    # Hardcoded configuration for bedroom dataset
    path_to_dataset_files = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/dataset_files"
    path_to_processed_data = "/mnt/sv-share/MiData/"
    dataset_directory = os.path.join(path_to_processed_data, "bedroom")
    annotation_file = os.path.join(path_to_dataset_files, "bedroom_threed_front_splits_original.csv")
    
    print("Loading scene IDs from CSV...")
    scene_ids = load_scene_ids_from_csv(annotation_file, split="train")
    print(f"Found {len(scene_ids)} training scenes")
    
    # Find all scene directories
    all_dirs = sorted(os.listdir(dataset_directory))
    scene_dirs = [d for d in all_dirs if any(d.endswith(scene_id) for scene_id in scene_ids)]
    print(f"Found {len(scene_dirs)} scene directories")
    
    # Analyze angles
    print(f"Analyzing {len(scene_dirs)} scenes...")
    
    total_objects = 0
    axis_aligned_count = 0  # angles close to 0, π/2, π, 3π/2
    rotated_count = 0
    
    angle_threshold = np.deg2rad(5)  # 5 degree tolerance
    all_angles = []
    
    for scene_dir in tqdm(scene_dirs):
        # Load the .npz file for this scene
        npz_path = os.path.join(dataset_directory, scene_dir, "boxes.npz")
        if not os.path.exists(npz_path):
            continue
        
        data = np.load(npz_path)
        print(f"[Ashok] all keys in the npz file: {data.files}")
        angles = data["angles"].flatten()  # shape (N, 1) -> (N,)
        all_angles.extend(angles.tolist())
        
        for angle in angles:
            total_objects += 1
            
            # Normalize angle to [0, 2π)
            normalized_angle = angle % (2 * np.pi)
            
            # Check if close to 0, π/2, π, or 3π/2
            axis_aligned_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
            is_axis_aligned = any(
                abs(normalized_angle - target) < angle_threshold or
                abs(normalized_angle - target - 2*np.pi) < angle_threshold
                for target in axis_aligned_angles
            )
            
            if is_axis_aligned:
                axis_aligned_count += 1
            else:
                rotated_count += 1
    
    # Calculate statistics
    all_angles = np.array(all_angles)
    
    print("\n" + "="*60)
    print("BOUNDING BOX ROTATION ANALYSIS")
    print("="*60)
    print(f"Total objects: {total_objects:,}")
    print(f"Axis-aligned (±5°): {axis_aligned_count:,} ({axis_aligned_count/total_objects*100:.2f}%)")
    print(f"Rotated: {rotated_count:,} ({rotated_count/total_objects*100:.2f}%)")
    
    print(f"\nAngle Statistics:")
    print(f"  Min angle: {np.min(all_angles):.4f} rad ({np.rad2deg(np.min(all_angles)):.2f}°)")
    print(f"  Max angle: {np.max(all_angles):.4f} rad ({np.rad2deg(np.max(all_angles)):.2f}°)")
    print(f"  Mean angle: {np.mean(all_angles):.4f} rad ({np.rad2deg(np.mean(all_angles)):.2f}°)")
    print(f"  Std angle: {np.std(all_angles):.4f} rad ({np.rad2deg(np.std(all_angles)):.2f}°)")
    
    # Check distribution of angles
    print(f"\nAngle Distribution (in degrees):")
    bins = np.linspace(-180, 180, 37)  # 10-degree bins
    hist, bin_edges = np.histogram(np.rad2deg(all_angles), bins=bins)
    
    # Show top 10 bins
    top_bins = np.argsort(hist)[-10:][::-1]
    for idx in top_bins:
        bin_center = (bin_edges[idx] + bin_edges[idx+1]) / 2
        print(f"  {bin_edges[idx]:6.1f}° to {bin_edges[idx+1]:6.1f}°: {hist[idx]:6,} objects ({hist[idx]/total_objects*100:5.2f}%)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
