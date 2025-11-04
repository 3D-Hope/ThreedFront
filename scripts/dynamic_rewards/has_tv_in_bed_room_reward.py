import pickle
import sys
import numpy as np
import argparse


def calculate_metrics(threed_front_results, ideal_distance=3.0, distance_range=(2.0, 4.0), angle_threshold=0.7):
    """
    Calculate constraint satisfaction metrics for TV-bed layouts.
    
    Args:
        threed_front_results: Object with _predicted_layouts attribute
        ideal_distance: Ideal TV-bed distance in meters (default: 3.0)
        distance_range: Acceptable distance range as (min, max) tuple
        angle_threshold: Minimum cosine similarity for good viewing angle
    
    Returns:
        dict: Metrics including rates and detailed statistics
    """
    
    # Class label mapping
    class_labels = {
        0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'ceiling_lamp', 
        4: 'chair', 5: 'children_cabinet', 6: 'coffee_table', 7: 'desk', 
        8: 'double_bed', 9: 'dressing_chair', 10: 'dressing_table', 
        11: 'kids_bed', 12: 'nightstand', 13: 'pendant_lamp', 14: 'shelf', 
        15: 'single_bed', 16: 'sofa', 17: 'stool', 18: 'table', 
        19: 'tv_stand', 20: 'wardrobe'
    }
    
    # Identify TV and bed indices
    tv_idx = 19  # tv_stand
    bed_indices = [8, 15, 11]  # double_bed, single_bed, kids_bed
    
    if not hasattr(threed_front_results, '_predicted_layouts'):
        print("Error: Object does not have _predicted_layouts attribute")
        return None
    
    layouts = threed_front_results._predicted_layouts
    total_scenes = len(layouts)
    
    # Tracking variables
    scenes_with_tv = 0
    scenes_with_bed = 0
    scenes_with_both = 0
    
    distances = []
    distances_in_range = 0
    
    viewing_angles = []
    good_viewing_angles = 0
    
    print(f"\nAnalyzing {total_scenes} scenes...")
    print("=" * 60)
    
    for scene_idx, layout in enumerate(layouts):
        class_labels_one_hot = layout['class_labels']
        translations = layout['translations']  # positions
        angles = layout['angles']  # theta angles in radians (z-axis rotation)
        
        # Get object indices from one-hot encoding
        object_indices = np.argmax(class_labels_one_hot, axis=1)
        
        # Find TV and bed in this scene
        tv_mask = (object_indices == tv_idx)
        bed_mask = np.isin(object_indices, bed_indices)
        
        has_tv = np.any(tv_mask)
        has_bed = np.any(bed_mask)
        
        if has_tv:
            scenes_with_tv += 1
        if has_bed:
            scenes_with_bed += 1
        
        # Only analyze scenes with both TV and bed
        if has_tv and has_bed:
            scenes_with_both += 1
            
            # Get positions
            tv_pos = translations[tv_mask][0]
            bed_pos = translations[bed_mask][0]
            
            # Calculate distance (in XZ plane, y is up)
            distance_vec = tv_pos - bed_pos
            distance_2d = np.sqrt(distance_vec[0]**2 + distance_vec[2]**2)
            distances.append(distance_2d)
            
            # Check if distance is in acceptable range
            if distance_range[0] <= distance_2d <= distance_range[1]:
                distances_in_range += 1
            
            # Calculate viewing angle alignment
            bed_angle = angles[bed_mask][0][0]  # Get the angle value
            
            # Convert angle to direction vector (cos, sin) in XZ plane
            bed_direction = np.array([np.cos(bed_angle), np.sin(bed_angle)])
            
            # Direction from bed to TV in XZ plane
            dir_bed_to_tv = np.array([distance_vec[0], distance_vec[2]])
            dir_bed_to_tv_norm = dir_bed_to_tv / (np.linalg.norm(dir_bed_to_tv) + 1e-6)
            
            # Cosine similarity (dot product of normalized vectors)
            alignment = np.dot(bed_direction, dir_bed_to_tv_norm)
            # Clamp to [0, 1] - only positive alignment counts
            alignment = np.clip(alignment, 0, 1)
            
            viewing_angles.append(alignment)
            
            if alignment >= angle_threshold:
                good_viewing_angles += 1
    
    # Calculate rates and statistics
    tv_presence_rate = (scenes_with_tv / total_scenes) * 100
    bed_presence_rate = (scenes_with_bed / total_scenes) * 100
    both_present_rate = (scenes_with_both / total_scenes) * 100
    
    # Distance metrics (only for scenes with both objects)
    if distances:
        distance_in_range_rate = (distances_in_range / len(distances)) * 100
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
    else:
        distance_in_range_rate = 0.0
        mean_distance = std_distance = min_distance = max_distance = 0.0
    
    # Viewing angle metrics (only for scenes with both objects)
    if viewing_angles:
        good_angle_rate = (good_viewing_angles / len(viewing_angles)) * 100
        mean_angle = np.mean(viewing_angles)
        std_angle = np.std(viewing_angles)
        min_angle = np.min(viewing_angles)
        max_angle = np.max(viewing_angles)
    else:
        good_angle_rate = 0.0
        mean_angle = std_angle = min_angle = max_angle = 0.0
    
    # Compile results
    metrics = {
        'total_scenes': total_scenes,
        'tv_presence_rate': tv_presence_rate,
        'bed_presence_rate': bed_presence_rate,
        'both_present_rate': both_present_rate,
        'scenes_with_both': scenes_with_both,
        'distance_in_range_rate': distance_in_range_rate,
        'distance_stats': {
            'mean': mean_distance,
            'std': std_distance,
            'min': min_distance,
            'max': max_distance,
            'ideal': ideal_distance,
            'acceptable_range': distance_range
        },
        'good_angle_rate': good_angle_rate,
        'angle_stats': {
            'mean': mean_angle,
            'std': std_angle,
            'min': min_angle,
            'max': max_angle,
            'threshold': angle_threshold
        }
    }
    
    return metrics


def print_metrics(metrics):
    """Pretty print the metrics."""
    if metrics is None:
        return
    
    print("\n" + "=" * 60)
    print("CONSTRAINT SATISFACTION METRICS")
    print("=" * 60)
    
    print(f"\nTotal Scenes Analyzed: {metrics['total_scenes']}")
    print(f"Scenes with TV: {metrics['total_scenes'] * metrics['tv_presence_rate'] / 100:.0f} ({metrics['tv_presence_rate']:.1f}%)")
    print(f"Scenes with Bed: {metrics['total_scenes'] * metrics['bed_presence_rate'] / 100:.0f} ({metrics['bed_presence_rate']:.1f}%)")
    print(f"Scenes with Both: {metrics['scenes_with_both']} ({metrics['both_present_rate']:.1f}%)")
    
    print("\n" + "-" * 60)
    print("1. TV PRESENCE RATE")
    print("-" * 60)
    print(f"   Rate: {metrics['tv_presence_rate']:.2f}%")
    
    print("\n" + "-" * 60)
    print("2. TV-BED DISTANCE")
    print("-" * 60)
    print(f"   Scenes in Acceptable Range ({metrics['distance_stats']['acceptable_range'][0]:.1f}m - {metrics['distance_stats']['acceptable_range'][1]:.1f}m): {metrics['distance_in_range_rate']:.2f}%")
    if metrics['scenes_with_both'] > 0:
        print(f"   Mean Distance: {metrics['distance_stats']['mean']:.3f}m (ideal: {metrics['distance_stats']['ideal']:.1f}m)")
        print(f"   Std Dev: {metrics['distance_stats']['std']:.3f}m")
        print(f"   Range: [{metrics['distance_stats']['min']:.3f}m, {metrics['distance_stats']['max']:.3f}m]")
    else:
        print("   No scenes with both TV and bed to analyze")
    
    print("\n" + "-" * 60)
    print("3. VIEWING ANGLE ALIGNMENT")
    print("-" * 60)
    print(f"   Scenes with Good Alignment (≥{metrics['angle_stats']['threshold']:.1f}): {metrics['good_angle_rate']:.2f}%")
    if metrics['scenes_with_both'] > 0:
        print(f"   Mean Alignment: {metrics['angle_stats']['mean']:.3f}")
        print(f"   Std Dev: {metrics['angle_stats']['std']:.3f}")
        print(f"   Range: [{metrics['angle_stats']['min']:.3f}, {metrics['angle_stats']['max']:.3f}]")
    else:
        print("   No scenes with both TV and bed to analyze")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ TV Presence: {metrics['tv_presence_rate']:.1f}%")
    print(f"✓ Optimal Distance: {metrics['distance_in_range_rate']:.1f}%")
    print(f"✓ Good Viewing Angle: {metrics['good_angle_rate']:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate constraint satisfaction metrics for TV-bed layout generation."
    )
    parser.add_argument(
        "result_file", 
        type=str, 
        help="Path to the ThreedFrontResults pickle file"
    )
    parser.add_argument(
        "--ideal-distance",
        type=float,
        default=3.0,
        help="Ideal TV-bed distance in meters (default: 3.0)"
    )
    parser.add_argument(
        "--distance-min",
        type=float,
        default=2.0,
        help="Minimum acceptable TV-bed distance in meters (default: 2.0)"
    )
    parser.add_argument(
        "--distance-max",
        type=float,
        default=4.0,
        help="Maximum acceptable TV-bed distance in meters (default: 4.0)"
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=0.7,
        help="Minimum cosine similarity for good viewing angle (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    # Load the pickle file
    print(f"\nLoading results from: {args.result_file}")
    try:
        with open(args.result_file, "rb") as f:
            threed_front_results = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {args.result_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    # Calculate metrics
    distance_range = (args.distance_min, args.distance_max)
    metrics = calculate_metrics(
        threed_front_results,
        ideal_distance=args.ideal_distance,
        distance_range=distance_range,
        angle_threshold=args.angle_threshold
    )
    
    # Print results
    if metrics:
        print_metrics(metrics)