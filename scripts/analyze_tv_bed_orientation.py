"""Analyze TV stand and bed orientations in the 3D-FRONT dataset.

This script loads bedroom scenes from boxes.npz files and analyzes whether
TV stands and beds are facing each other or in the same direction.

Usage:
    python scripts/analyze_tv_bed_orientation.py /path/to/bedroom/dataset
"""
import argparse
import os
import numpy as np
from pathlib import Path
from collections import defaultdict


# Label mapping from the dataset
idx_to_labels = {
    0: "armchair",
    1: "bookshelf",
    2: "cabinet",
    3: "ceiling_lamp",
    4: "chair",
    5: "children_cabinet",
    6: "coffee_table",
    7: "desk",
    8: "double_bed",
    9: "dressing_chair",
    10: "dressing_table",
    11: "kids_bed",
    12: "nightstand",
    13: "pendant_lamp",
    14: "shelf",
    15: "single_bed",
    16: "sofa",
    17: "stool",
    18: "table",
    19: "tv_stand",
    20: "wardrobe",
}

labels_to_idx = {v: k for k, v in idx_to_labels.items()}


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def get_facing_direction(angle):
    """Get unit vector representing the facing direction given rotation angle.
    
    In 3D-FRONT, angle is rotation around Y axis (vertical).
    We assume objects face along their local +Z axis initially (forward direction).
    After rotation by angle around Y, the facing direction becomes:
    facing = [sin(angle), 0, cos(angle)]
    """
    # Ensure angle is a scalar
    if isinstance(angle, np.ndarray):
        angle = float(angle.item())
    return np.array([np.sin(angle), 0, np.cos(angle)])


def angle_between_vectors(v1, v2):
    """Compute angle between two vectors in radians."""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(cos_angle)


def are_facing_each_other(pos1, angle1, pos2, angle2, threshold_deg=45):
    """Check if two objects are facing each other.
    
    Args:
        pos1, pos2: 3D positions (translations) of the two objects
        angle1, angle2: rotation angles around Y axis
        threshold_deg: angle threshold in degrees to consider "facing"
    
    Returns:
        dict with analysis results
    """
    # Get facing directions
    dir1 = get_facing_direction(angle1)
    dir2 = get_facing_direction(angle2)
    
    # Vector from object 1 to object 2
    vec_1_to_2 = pos2 - pos1
    vec_1_to_2[1] = 0  # Project to XZ plane (ignore height)
    vec_1_to_2_norm = vec_1_to_2 / (np.linalg.norm(vec_1_to_2) + 1e-8)
    
    # Vector from object 2 to object 1
    vec_2_to_1 = -vec_1_to_2_norm
    
    # Check if obj1 is facing towards obj2
    angle_1_to_2 = angle_between_vectors(dir1, vec_1_to_2_norm)
    
    # Check if obj2 is facing towards obj1
    angle_2_to_1 = angle_between_vectors(dir2, vec_2_to_1)
    
    threshold_rad = np.deg2rad(threshold_deg)
    obj1_facing_obj2 = angle_1_to_2 < threshold_rad
    obj2_facing_obj1 = angle_2_to_1 < threshold_rad
    
    # Angle between facing directions (0 means same direction, pi means opposite)
    angle_between_dirs = angle_between_vectors(dir1, dir2)
    
    # Convert all numpy types to Python native types for clean printing
    return {
        'distance': float(np.linalg.norm(vec_1_to_2)),
        'angle_1_to_2_deg': float(np.rad2deg(angle_1_to_2)),
        'angle_2_to_1_deg': float(np.rad2deg(angle_2_to_1)),
        'obj1_facing_obj2': bool(obj1_facing_obj2),
        'obj2_facing_obj1': bool(obj2_facing_obj1),
        'facing_each_other': bool(obj1_facing_obj2 and obj2_facing_obj1),
        'angle_between_dirs_deg': float(np.rad2deg(angle_between_dirs)),
        'same_direction': bool(angle_between_dirs < np.deg2rad(45)),
        'opposite_direction': bool(angle_between_dirs > np.deg2rad(135)),
        'angle1_rad': float(angle1) if isinstance(angle1, np.ndarray) else float(angle1),
        'angle2_rad': float(angle2) if isinstance(angle2, np.ndarray) else float(angle2),
        'angle1_deg': float(np.rad2deg(angle1)),
        'angle2_deg': float(np.rad2deg(angle2)),
    }


def analyze_scene(npz_path):
    """Analyze a single scene for TV stand and bed orientations."""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return None
    
    # Extract data
    class_labels = data['class_labels']  # Shape: (N, num_classes)
    translations = data['translations']   # Shape: (N, 3)
    angles = data['angles']              # Shape: (N,)
    
    # Get class indices (argmax over one-hot encoding)
    class_indices = np.argmax(class_labels, axis=1)
    
    # Find TV stands and beds
    tv_stand_idx_label = labels_to_idx.get('tv_stand')
    bed_indices_labels = [
        labels_to_idx.get('double_bed'),
        labels_to_idx.get('single_bed'),
        labels_to_idx.get('kids_bed')
    ]
    
    # Find objects
    tv_stands = [(i, translations[i], angles[i]) 
                 for i in range(len(class_indices)) 
                 if class_indices[i] == tv_stand_idx_label]
    
    beds = [(i, translations[i], angles[i]) 
            for i in range(len(class_indices)) 
            if class_indices[i] in bed_indices_labels]
    
    if not tv_stands or not beds:
        return None
    
    results = []
    for tv_idx, tv_pos, tv_angle in tv_stands:
        for bed_idx, bed_pos, bed_angle in beds:
            bed_label = idx_to_labels[class_indices[bed_idx]]
            analysis = are_facing_each_other(tv_pos, tv_angle, bed_pos, bed_angle)
            analysis['scene_path'] = str(npz_path)
            analysis['tv_idx'] = tv_idx
            analysis['bed_idx'] = bed_idx
            analysis['bed_type'] = bed_label
            results.append(analysis)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TV stand and bed orientations in 3D-FRONT dataset"
    )
    parser.add_argument(
        "dataset_directory",
        help="Path to the bedroom dataset directory containing scene subdirectories"
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to analyze (default: all)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each scene"
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_directory)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist")
        return
    
    # Find all boxes.npz files
    scene_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    
    if args.max_scenes:
        scene_dirs = scene_dirs[:args.max_scenes]
    
    print(f"Analyzing {len(scene_dirs)} scenes from {dataset_dir}")
    print("=" * 80)
    
    all_results = []
    scenes_with_tv_and_bed = 0
    table_data = []  # Store data for table printing
    
    for scene_dir in scene_dirs:
        npz_path = scene_dir / "boxes.npz"
        if not npz_path.exists():
            continue
        
        scene_results = analyze_scene(npz_path)
        if scene_results:
            scenes_with_tv_and_bed += 1
            all_results.extend(scene_results)
            
            # Store for table
            for res in scene_results:
                table_data.append({
                    'scene_name': scene_dir.name,
                    'bed_type': res['bed_type'],
                    'tv_angle_rad': res['angle1_rad'],
                    'tv_angle_deg': res['angle1_deg'],
                    'bed_angle_rad': res['angle2_rad'],
                    'bed_angle_deg': res['angle2_deg'],
                    'angle_diff_rad': abs(normalize_angle(res['angle1_rad'] - res['angle2_rad'])),
                    'facing_each_other': res['facing_each_other'],
                    'same_direction': res['same_direction'],
                    'opposite_direction': res['opposite_direction'],
                })
            
            if args.verbose:
                print(f"\nScene: {scene_dir.name}")
                for res in scene_results:
                    print(f"  TV-{res['bed_type']} pair:")
                    print(f"    Distance: {res['distance']:.2f}m")
                    print(f"    TV angle: {res['angle1_deg']:.1f}° ({res['angle1_rad']:.3f} rad)")
                    print(f"    Bed angle: {res['angle2_deg']:.1f}° ({res['angle2_rad']:.3f} rad)")
                    print(f"    Angle between facing dirs: {res['angle_between_dirs_deg']:.1f}°")
                    print(f"    TV facing bed: {res['obj1_facing_obj2']}")
                    print(f"    Bed facing TV: {res['obj2_facing_obj1']}")
                    print(f"    Facing each other: {res['facing_each_other']}")
                    print(f"    Same direction: {res['same_direction']}")
    
    if not all_results:
        print("\nNo scenes found with both TV stand and bed!")
        return
    
    # Print table of angles
    print(f"\n{'=' * 120}")
    print(f"TABLE: TV Stand and Bed Angles (First 50 scenes)")
    print(f"{'=' * 120}")
    print(f"{'Scene':<50} {'Bed Type':<15} {'TV Angle (rad)':<15} {'TV Angle (°)':<15} {'Bed Angle (rad)':<15} {'Bed Angle (°)':<15}")
    print(f"{'-' * 120}")
    
    for i, row in enumerate(table_data[:50]):
        print(f"{row['scene_name']:<50} {row['bed_type']:<15} "
              f"{row['tv_angle_rad']:>14.3f} {row['tv_angle_deg']:>14.1f} "
              f"{row['bed_angle_rad']:>14.3f} {row['bed_angle_deg']:>14.1f}")
    
    if len(table_data) > 50:
        print(f"... ({len(table_data) - 50} more scenes not shown)")
    
    print(f"{'=' * 120}")
    
    # Aggregate statistics
    print(f"\n{'=' * 80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    print(f"Total scenes analyzed: {len(scene_dirs)}")
    print(f"Scenes with TV and bed: {scenes_with_tv_and_bed}")
    print(f"Total TV-bed pairs: {len(all_results)}")
    print()
    
    # Count different configurations
    facing_each_other = sum(1 for r in all_results if r['facing_each_other'])
    tv_facing_bed = sum(1 for r in all_results if r['obj1_facing_obj2'])
    bed_facing_tv = sum(1 for r in all_results if r['obj2_facing_obj1'])
    same_direction = sum(1 for r in all_results if r['same_direction'])
    opposite_direction = sum(1 for r in all_results if r['opposite_direction'])
    
    print(f"Orientation Analysis:")
    print(f"  Facing each other: {facing_each_other} ({100*facing_each_other/len(all_results):.1f}%)")
    print(f"  TV facing bed (but not mutual): {tv_facing_bed} ({100*tv_facing_bed/len(all_results):.1f}%)")
    print(f"  Bed facing TV (but not mutual): {bed_facing_tv} ({100*bed_facing_tv/len(all_results):.1f}%)")
    print(f"  Same direction (±45°): {same_direction} ({100*same_direction/len(all_results):.1f}%)")
    print(f"  Opposite direction (±45°): {opposite_direction} ({100*opposite_direction/len(all_results):.1f}%)")
    print()
    
    # Angle statistics
    tv_angles = [r['angle1_rad'] for r in all_results]
    bed_angles = [r['angle2_rad'] for r in all_results]
    angle_diffs = [abs(normalize_angle(r['angle1_rad'] - r['angle2_rad'])) 
                   for r in all_results]
    dir_angles = [r['angle_between_dirs_deg'] for r in all_results]
    
    print(f"Angle Statistics:")
    print(f"  TV angles (rad):")
    print(f"    Mean: {np.mean(tv_angles):.3f}, Std: {np.std(tv_angles):.3f}")
    print(f"    Min: {np.min(tv_angles):.3f}, Max: {np.max(tv_angles):.3f}")
    print(f"  Bed angles (rad):")
    print(f"    Mean: {np.mean(bed_angles):.3f}, Std: {np.std(bed_angles):.3f}")
    print(f"    Min: {np.min(bed_angles):.3f}, Max: {np.max(bed_angles):.3f}")
    print(f"  Angle difference |TV - Bed| (rad):")
    print(f"    Mean: {np.mean(angle_diffs):.3f}, Std: {np.std(angle_diffs):.3f}")
    print(f"  Angle between facing directions (deg):")
    print(f"    Mean: {np.mean(dir_angles):.1f}°, Std: {np.std(dir_angles):.1f}°")
    print(f"    Min: {np.min(dir_angles):.1f}°, Max: {np.max(dir_angles):.1f}°")
    print()
    
    # Check if most angles are close to 0
    tv_near_zero = sum(1 for a in tv_angles if abs(a) < 0.1)
    bed_near_zero = sum(1 for a in bed_angles if abs(a) < 0.1)
    both_near_zero = sum(1 for r in all_results if abs(r['angle1_rad']) < 0.1 and abs(r['angle2_rad']) < 0.1)
    
    print(f"Near-zero angles (|angle| < 0.1 rad ≈ 5.7°):")
    print(f"  TV stands: {tv_near_zero}/{len(all_results)} ({100*tv_near_zero/len(all_results):.1f}%)")
    print(f"  Beds: {bed_near_zero}/{len(all_results)} ({100*bed_near_zero/len(all_results):.1f}%)")
    print(f"  Both: {both_near_zero}/{len(all_results)} ({100*both_near_zero/len(all_results):.1f}%)")
    print()
    
    # Distance statistics
    distances = [r['distance'] for r in all_results]
    print(f"Distance Statistics (meters):")
    print(f"  Mean: {np.mean(distances):.2f}m, Std: {np.std(distances):.2f}m")
    print(f"  Min: {np.min(distances):.2f}m, Max: {np.max(distances):.2f}m")
    print()
    
    # Show some examples
    print(f"{'=' * 80}")
    print("EXAMPLE SCENES:")
    print(f"{'=' * 80}")
    
    # Find examples of different configurations
    if same_direction > 0:
        print("\nExample of SAME DIRECTION configuration:")
        same_dir_example = next(r for r in all_results if r['same_direction'])
        print(f"  Scene: {Path(same_dir_example['scene_path']).parent.name}")
        print(f"  TV angle: {same_dir_example['angle1_deg']:.1f}° ({same_dir_example['angle1_rad']:.3f} rad)")
        print(f"  Bed angle: {same_dir_example['angle2_deg']:.1f}° ({same_dir_example['angle2_rad']:.3f} rad)")
        print(f"  Angle between facing directions: {same_dir_example['angle_between_dirs_deg']:.1f}°")
    
    if facing_each_other > 0:
        print("\nExample of FACING EACH OTHER configuration:")
        facing_example = next(r for r in all_results if r['facing_each_other'])
        print(f"  Scene: {Path(facing_example['scene_path']).parent.name}")
        print(f"  TV angle: {facing_example['angle1_deg']:.1f}° ({facing_example['angle1_rad']:.3f} rad)")
        print(f"  Bed angle: {facing_example['angle2_deg']:.1f}° ({facing_example['angle2_rad']:.3f} rad)")
        print(f"  Angle between facing directions: {facing_example['angle_between_dirs_deg']:.1f}°")
    
    if opposite_direction > 0:
        print("\nExample of OPPOSITE DIRECTION configuration:")
        opp_example = next(r for r in all_results if r['opposite_direction'])
        print(f"  Scene: {Path(opp_example['scene_path']).parent.name}")
        print(f"  TV angle: {opp_example['angle1_deg']:.1f}° ({opp_example['angle1_rad']:.3f} rad)")
        print(f"  Bed angle: {opp_example['angle2_deg']:.1f}° ({opp_example['angle2_rad']:.3f} rad)")
        print(f"  Angle between facing directions: {opp_example['angle_between_dirs_deg']:.1f}°")


if __name__ == "__main__":
    main()
