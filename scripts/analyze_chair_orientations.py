"""
Analyze Chair Orientations in 3D-FRONT Dataset
==============================================

This script analyzes what chairs are facing in the dataset to determine if
"chairs facing walls" is a dataset artifact or a generation problem.

For each scene with chairs, we check:
1. What direction is the chair facing?
2. Is it facing toward a wall/room boundary?
3. Is it facing toward another object (desk, table, etc.)?
4. What's the distribution of chair orientations?

Usage:
    python analyze_chair_orientations.py /path/to/bedroom/dataset [--max_scenes N]
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


idx_to_labels = {
    0: "armchair", 1: "bookshelf", 2: "cabinet", 3: "ceiling_lamp",
    4: "chair", 5: "children_cabinet", 6: "coffee_table", 7: "desk",
    8: "double_bed", 9: "dressing_chair", 10: "dressing_table",
    11: "kids_bed", 12: "nightstand", 13: "pendant_lamp", 14: "shelf",
    15: "single_bed", 16: "sofa", 17: "stool", 18: "table",
    19: "tv_stand", 20: "wardrobe",
}

labels_to_idx = {v: k for k, v in idx_to_labels.items()}

# Chair types
CHAIR_TYPES = ["chair", "armchair", "dressing_chair", "stool"]
CHAIR_INDICES = [labels_to_idx[ct] for ct in CHAIR_TYPES if ct in labels_to_idx]

# Objects chairs typically face
FACING_OBJECTS = ["desk", "dressing_table", "table"]
FACING_INDICES = [labels_to_idx[obj] for obj in FACING_OBJECTS if obj in labels_to_idx]


def get_room_bounds_from_floor_plan(floor_plan_vertices):
    """Get approximate room boundaries from floor plan vertices."""
    if floor_plan_vertices is None or len(floor_plan_vertices) == 0:
        return None
    
    xz_coords = floor_plan_vertices[:, [0, 2]]
    min_x, min_z = xz_coords.min(axis=0)
    max_x, max_z = xz_coords.max(axis=0)
    
    return {
        'min_x': min_x, 'max_x': max_x,
        'min_z': min_z, 'max_z': max_z,
        'center': np.array([(min_x + max_x) / 2, 0, (min_z + max_z) / 2])
    }


def distance_to_nearest_wall(pos, room_bounds):
    """Calculate distance from position to nearest wall."""
    if room_bounds is None:
        return float('inf')
    
    x, z = pos[0], pos[2]
    
    dist_to_walls = [
        abs(x - room_bounds['min_x']),
        abs(x - room_bounds['max_x']),
        abs(z - room_bounds['min_z']),
        abs(z - room_bounds['max_z'])
    ]
    
    return min(dist_to_walls)


def is_facing_wall(pos, angle, room_bounds, threshold_distance=0.5, threshold_angle=45):
    """
    Check if chair is facing toward a wall.
    
    Returns:
        (bool, dict): (True if facing wall, info dict)
    """
    if room_bounds is None:
        return False, {}
    
    # Chair facing direction
    facing = np.array([np.sin(angle), np.cos(angle)])
    
    x, z = pos[0], pos[2]
    
    # Directions to each wall (normalized)
    walls = {
        'left': (np.array([-1, 0]), abs(x - room_bounds['min_x'])),
        'right': (np.array([1, 0]), abs(x - room_bounds['max_x'])),
        'back': (np.array([0, -1]), abs(z - room_bounds['min_z'])),
        'front': (np.array([0, 1]), abs(z - room_bounds['max_z']))
    }
    
    # Check each wall
    facing_wall = False
    wall_info = {}
    
    for wall_name, (wall_dir, wall_dist) in walls.items():
        # Angle between facing and wall direction
        dot_product = np.dot(facing, wall_dir)
        angle_to_wall = np.rad2deg(np.arccos(np.clip(dot_product, -1, 1)))
        
        if angle_to_wall < threshold_angle and wall_dist < threshold_distance:
            facing_wall = True
            wall_info = {
                'wall': wall_name,
                'distance': wall_dist,
                'angle': angle_to_wall
            }
            break
    
    return facing_wall, wall_info


def find_facing_object(chair_pos, chair_angle, all_positions, all_indices, all_angles,
                       threshold_angle=60, max_distance=2.0):
    """
    Find what object the chair is likely facing.
    
    Returns:
        (object_label, distance, alignment_angle) or (None, None, None)
    """
    facing_chair = np.array([np.sin(chair_angle), np.cos(chair_angle)])
    
    best_match = None
    best_score = -1
    
    for i, (obj_pos, obj_idx) in enumerate(zip(all_positions, all_indices)):
        # Skip the chair itself
        if np.allclose(obj_pos, chair_pos):
            continue
        
        # Direction from chair to object
        dir_to_obj = obj_pos - chair_pos
        distance = np.linalg.norm(dir_to_obj[[0, 2]])
        
        if distance > max_distance:
            continue
        
        dir_to_obj_xz = dir_to_obj[[0, 2]]
        dir_to_obj_xz = dir_to_obj_xz / (np.linalg.norm(dir_to_obj_xz) + 1e-8)
        
        # Angle between chair facing and direction to object
        dot_product = np.dot(facing_chair, dir_to_obj_xz)
        angle_deg = np.rad2deg(np.arccos(np.clip(dot_product, -1, 1)))
        
        if angle_deg < threshold_angle:
            # Score based on angle and distance (closer and more aligned = better)
            score = (1 - angle_deg / threshold_angle) * (1 - distance / max_distance)
            
            if score > best_score:
                best_score = score
                best_match = (idx_to_labels[obj_idx], distance, angle_deg)
    
    return best_match if best_match else (None, None, None)


def analyze_scene(npz_path):
    """Analyze chair orientations in a single scene."""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return None
    
    class_labels = data['class_labels']
    translations = data['translations']
    angles = data['angles'].squeeze()
    
    # Get room bounds if available
    room_bounds = None
    if 'floor_plan_vertices' in data:
        room_bounds = get_room_bounds_from_floor_plan(data['floor_plan_vertices'])
    
    class_indices = np.argmax(class_labels, axis=1)
    
    # Find all chairs
    chair_mask = np.isin(class_indices, CHAIR_INDICES)
    if not chair_mask.any():
        return None
    
    results = []
    
    for chair_idx in np.where(chair_mask)[0]:
        chair_pos = translations[chair_idx]
        chair_angle = float(angles[chair_idx])
        chair_type = idx_to_labels[class_indices[chair_idx]]
        
        # Check if facing wall
        facing_wall, wall_info = is_facing_wall(chair_pos, chair_angle, room_bounds)
        
        # Find what object it's facing
        facing_obj, obj_dist, obj_angle = find_facing_object(
            chair_pos, chair_angle, translations, class_indices, angles
        )
        
        # Distance to nearest wall
        dist_to_wall = distance_to_nearest_wall(chair_pos, room_bounds)
        
        results.append({
            'chair_type': chair_type,
            'position': chair_pos,
            'angle_deg': np.rad2deg(chair_angle),
            'angle_rad': chair_angle,
            'facing_wall': facing_wall,
            'wall_info': wall_info,
            'facing_object': facing_obj,
            'object_distance': obj_dist,
            'object_angle': obj_angle,
            'distance_to_wall': dist_to_wall,
            'scene_path': str(npz_path)
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze chair orientations in 3D-FRONT dataset"
    )
    parser.add_argument(
        "dataset_directory",
        help="Path to the dataset directory containing scene subdirectories"
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to analyze"
    )
    parser.add_argument(
        "--save_plot",
        action="store_true",
        help="Save orientation distribution plot"
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_directory)
    scene_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    
    if args.max_scenes:
        scene_dirs = scene_dirs[:args.max_scenes]
    
    print(f"Analyzing chair orientations in {len(scene_dirs)} scenes...")
    print("=" * 100)
    print()
    
    all_results = []
    scenes_with_chairs = 0
    
    for scene_dir in scene_dirs:
        npz_path = scene_dir / "boxes.npz"
        if not npz_path.exists():
            continue
        
        scene_results = analyze_scene(npz_path)
        if scene_results:
            scenes_with_chairs += 1
            all_results.extend(scene_results)
    
    if not all_results:
        print("No chairs found in any scenes!")
        return
    
    # Aggregate statistics
    print(f"SUMMARY STATISTICS")
    print("=" * 100)
    print(f"Total scenes analyzed: {len(scene_dirs)}")
    print(f"Scenes with chairs: {scenes_with_chairs}")
    print(f"Total chairs found: {len(all_results)}")
    print()
    
    # Chair types distribution
    chair_types = Counter([r['chair_type'] for r in all_results])
    print(f"Chair Types:")
    for chair_type, count in chair_types.most_common():
        print(f"  {chair_type}: {count} ({100*count/len(all_results):.1f}%)")
    print()
    
    # Facing wall statistics
    facing_wall_count = sum(1 for r in all_results if r['facing_wall'])
    print(f"Chairs Facing Walls:")
    print(f"  Count: {facing_wall_count} / {len(all_results)} ({100*facing_wall_count/len(all_results):.1f}%)")
    print()
    
    # Facing objects statistics
    facing_objects = Counter([r['facing_object'] for r in all_results if r['facing_object'] is not None])
    print(f"Chairs Facing Objects:")
    if facing_objects:
        for obj, count in facing_objects.most_common():
            print(f"  {obj}: {count} ({100*count/len(all_results):.1f}%)")
    else:
        print(f"  No clear object associations found")
    print()
    
    # Not facing anything
    no_clear_target = sum(1 for r in all_results if not r['facing_wall'] and r['facing_object'] is None)
    print(f"Chairs Not Clearly Facing Anything:")
    print(f"  Count: {no_clear_target} / {len(all_results)} ({100*no_clear_target/len(all_results):.1f}%)")
    print()
    
    # Angle distribution
    angles = [r['angle_deg'] for r in all_results]
    print(f"Orientation Angles:")
    print(f"  Mean: {np.mean(angles):.1f}°")
    print(f"  Std: {np.std(angles):.1f}°")
    print(f"  Min: {np.min(angles):.1f}°  |  Max: {np.max(angles):.1f}°")
    print()
    
    # Distance to walls
    distances = [r['distance_to_wall'] for r in all_results]
    print(f"Distance to Nearest Wall:")
    print(f"  Mean: {np.mean(distances):.2f}m")
    print(f"  Std: {np.std(distances):.2f}m")
    print(f"  Min: {np.min(distances):.2f}m  |  Max: {np.max(distances):.2f}m")
    print()
    
    # Example chairs facing walls
    print("=" * 100)
    print("EXAMPLES OF CHAIRS FACING WALLS (first 10):")
    print("=" * 100)
    
    wall_facers = [r for r in all_results if r['facing_wall']][:10]
    if wall_facers:
        print(f"{'Scene':<50} {'Type':<15} {'Angle':<10} {'Wall':<10} {'Dist'}")
        print("-" * 100)
        for r in wall_facers:
            scene_name = Path(r['scene_path']).parent.name
            wall_name = r['wall_info'].get('wall', 'N/A')
            wall_dist = r['wall_info'].get('distance', 0)
            print(f"{scene_name:<50} {r['chair_type']:<15} {r['angle_deg']:>8.1f}° {wall_name:<10} {wall_dist:.2f}m")
    else:
        print("No chairs facing walls found!")
    print()
    
    # Plot angle distribution
    if args.save_plot:
        plt.figure(figsize=(12, 6))
        
        # Histogram of angles
        plt.subplot(1, 2, 1)
        plt.hist(angles, bins=36, edgecolor='black', alpha=0.7)
        plt.xlabel('Orientation Angle (degrees)')
        plt.ylabel('Count')
        plt.title('Chair Orientation Distribution')
        plt.grid(True, alpha=0.3)
        
        # Polar histogram
        plt.subplot(1, 2, 2, projection='polar')
        angles_rad = np.deg2rad(angles)
        plt.hist(angles_rad, bins=36, alpha=0.7)
        plt.title('Chair Orientations (Polar)')
        
        plt.tight_layout()
        plt.savefig('chair_orientation_distribution.png', dpi=150, bbox_inches='tight')
        print(f"Saved plot to chair_orientation_distribution.png")
    
    # Conclusion
    print("=" * 100)
    print("CONCLUSION:")
    print("=" * 100)
    
    if facing_wall_count / len(all_results) > 0.2:
        print(f"⚠️  {100*facing_wall_count/len(all_results):.1f}% of chairs face walls in the dataset!")
        print(f"   This IS a dataset artifact. If your generated scenes have similar rates,")
        print(f"   it's likely the model learned this from the data.")
    else:
        print(f"✓  Only {100*facing_wall_count/len(all_results):.1f}% of chairs face walls in the dataset.")
        print(f"   If your generated scenes have much higher rates, it's a generation problem.")
    print()
    
    if sum(facing_objects.values()) / len(all_results) > 0.5:
        print(f"✓  {100*sum(facing_objects.values())/len(all_results):.1f}% of chairs face objects (good!)")
        print(f"   Most common: {facing_objects.most_common(3)}")
    else:
        print(f"⚠️  Only {100*sum(facing_objects.values())/len(all_results):.1f}% face specific objects.")
    

if __name__ == "__main__":
    main()
