#!/usr/bin/env python3
"""
Analyze gravity violations in ground truth 3D-FRONT bedroom scenes.

Gravity violation is defined as the sum of distances from ground (y=0) to the 
minimum y point of each bounding box (excluding ceiling objects).

Y-axis is up, sizes are already half-extents (no need to divide by 2).
Violation = sum of abs(y_min) for all non-ceiling objects per scene.
"""

import os
import numpy as np
from tqdm import tqdm


# Ceiling objects that should NOT follow gravity (their indices)
CEILING_OBJECT_INDICES = {
    3,   # ceiling_lamp
    11,  # pendant_lamp
}


def load_scene_ids_from_csv(csv_path, splits=["train", "val"]):
    """Load scene IDs from the CSV file for specific splits"""
    scene_ids = []
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                scene_id, scene_split = parts
                if scene_split in splits:
                    scene_ids.append(scene_id)
    return scene_ids


def main():
    """
    Analyze gravity violations in the 3D-FRONT bedroom dataset.
    """
    # Hardcoded configuration for bedroom dataset
    path_to_dataset_files = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/dataset_files"
    path_to_processed_data = "/mnt/sv-share/MiData/"
    dataset_directory = os.path.join(path_to_processed_data, "bedroom")
    annotation_file = os.path.join(path_to_dataset_files, "bedroom_threed_front_splits_original.csv")
    
    print("Loading scene IDs from CSV (train + val splits)...")
    scene_ids = load_scene_ids_from_csv(annotation_file, splits=["train", "val"])
    print(f"Found {len(scene_ids)} training+validation scenes")
    
    # Find all scene directories
    all_dirs = sorted(os.listdir(dataset_directory))
    scene_dirs = [d for d in all_dirs if any(d.endswith(scene_id) for scene_id in scene_ids)]
    print(f"Found {len(scene_dirs)} scene directories")
    
    # Analyze gravity violations
    print(f"Analyzing {len(scene_dirs)} scenes...")
    
    violations_per_scene = []
    all_object_violations = []
    num_objects_per_scene = []
    
    for scene_dir in tqdm(scene_dirs):
        # Load the .npz file for this scene
        npz_path = os.path.join(dataset_directory, scene_dir, "boxes.npz")
        if not os.path.exists(npz_path):
            continue
        
        data = np.load(npz_path)
        
        # translations: [x, y, z] center of bbox - shape (N, 3)
        # sizes: [sx/2, sy/2, sz/2] HALF extents - shape (N, 3)
        # class_labels: object type indices - shape (N,)
        translations = data["translations"]  # (N, 3)
        sizes = data["sizes"]  # (N, 3) - already half-extents
        class_labels = data["class_labels"].flatten()  # (N,)
        
        scene_violation = 0.0
        scene_object_count = 0
        
        num_objects = len(translations)
        for i in range(num_objects):
            # Skip ceiling objects
            if class_labels[i] in CEILING_OBJECT_INDICES:
                continue
            
            # Get y center and y half-extent
            y_center = translations[i, 1]
            y_half = sizes[i, 1]  # Already half-extent, no need to divide by 2
            
            # Calculate minimum y point of the bounding box
            y_min = y_center - y_half
            
            # Gravity violation is the absolute distance from ground (y=0)
            violation = abs(y_min)
            
            scene_violation += violation
            all_object_violations.append(violation)
            scene_object_count += 1
        
        violations_per_scene.append(scene_violation)
        num_objects_per_scene.append(scene_object_count)
    
    # Convert to numpy arrays
    violations_per_scene = np.array(violations_per_scene)
    all_object_violations = np.array(all_object_violations)
    num_objects_per_scene = np.array(num_objects_per_scene)
    
    # Print statistics
    print("\n" + "="*70)
    print("GRAVITY VIOLATION STATISTICS - TRAIN+VAL SET")
    print("="*70)
    print(f"\nScene-Level Statistics:")
    print(f"  Total scenes analyzed:        {len(violations_per_scene):,}")
    print(f"  Mean objects per scene:       {np.mean(num_objects_per_scene):.2f}")
    
    print(f"\nTotal Violation per Scene (sum of all object violations):")
    print(f"  Mean:                         {np.mean(violations_per_scene):.6f} m  ({np.mean(violations_per_scene)*1000:.2f} mm)")
    print(f"  Std Dev:                      {np.std(violations_per_scene):.6f} m  ({np.std(violations_per_scene)*1000:.2f} mm)")
    print(f"  Minimum:                      {np.min(violations_per_scene):.6f} m  ({np.min(violations_per_scene)*1000:.2f} mm)")
    print(f"  Maximum:                      {np.max(violations_per_scene):.6f} m  ({np.max(violations_per_scene)*1000:.2f} mm)")
    print(f"  Median:                       {np.median(violations_per_scene):.6f} m  ({np.median(violations_per_scene)*1000:.2f} mm)")
    print(f"  25th percentile:              {np.percentile(violations_per_scene, 25):.6f} m  ({np.percentile(violations_per_scene, 25)*1000:.2f} mm)")
    print(f"  75th percentile:              {np.percentile(violations_per_scene, 75):.6f} m  ({np.percentile(violations_per_scene, 75)*1000:.2f} mm)")
    print(f"  90th percentile:              {np.percentile(violations_per_scene, 90):.6f} m  ({np.percentile(violations_per_scene, 90)*1000:.2f} mm)")
    print(f"  95th percentile:              {np.percentile(violations_per_scene, 95):.6f} m  ({np.percentile(violations_per_scene, 95)*1000:.2f} mm)")
    print(f"  99th percentile:              {np.percentile(violations_per_scene, 99):.6f} m  ({np.percentile(violations_per_scene, 99)*1000:.2f} mm)")
    
    print(f"\nPer-Object Violation Statistics:")
    print(f"  Total objects analyzed:       {len(all_object_violations):,}")
    print(f"  Mean violation per object:    {np.mean(all_object_violations):.6f} m  ({np.mean(all_object_violations)*1000:.2f} mm)")
    print(f"  Std Dev per object:           {np.std(all_object_violations):.6f} m  ({np.std(all_object_violations)*1000:.2f} mm)")
    print(f"  Median per object:            {np.median(all_object_violations):.6f} m  ({np.median(all_object_violations)*1000:.2f} mm)")
    print(f"  Min violation (single obj):   {np.min(all_object_violations):.6f} m  ({np.min(all_object_violations)*1000:.2f} mm)")
    print(f"  Max violation (single obj):   {np.max(all_object_violations):.6f} m  ({np.max(all_object_violations)*1000:.2f} mm)")
    
    print(f"\nInterpretation:")
    print(f"  Average scene has ~{np.mean(num_objects_per_scene):.1f} objects")
    print(f"  Each object is on average ~{np.mean(all_object_violations)*1000:.2f} mm off the ground")
    print(f"  Total scene violation is on average ~{np.mean(violations_per_scene)*1000:.2f} mm (sum across all objects)")
    
    # Distribution analysis
    print(f"\nDistribution of Scene Violations:")
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(violations_per_scene, p)
        print(f"  {p:2d}% of scenes have violation ≤ {val:.6f} m ({val*1000:.2f} mm)")
    
    # Analyze per-object distribution
    print(f"\nDistribution of Per-Object Violations:")
    for p in percentiles:
        val = np.percentile(all_object_violations, p)
        print(f"  {p:2d}% of objects have violation ≤ {val:.6f} m ({val*1000:.2f} mm)")
    
    # Count violations by magnitude
    print(f"\nViolation Magnitude Breakdown (per object):")
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]  # meters
    for i, thresh in enumerate(thresholds):
        count = np.sum(all_object_violations <= thresh)
        pct = count / len(all_object_violations) * 100
        print(f"  ≤ {thresh*1000:5.1f} mm: {count:6,} objects ({pct:5.2f}%)")
    
    print("\n" + "="*70)
    
    # Save detailed results
    output_file = "gravity_violations_analysis.txt"
    with open(output_file, 'w') as f:
        f.write("Scene-level violations (meters):\n")
        for i, (v, n) in enumerate(zip(violations_per_scene, num_objects_per_scene)):
            f.write(f"{i},{v:.6f},{n}\n")
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
