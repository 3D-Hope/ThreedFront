#!/usr/bin/env python3
"""
Analyze free floor space distribution in 3D-FRONT dataset for navigability metrics.
Excludes ceiling objects (pendant_lamp, ceiling_lamp) and calculates:
- Free space ratio (free area / total floor area)
- Minimum free space area
- Statistics for navigability constraints
"""

import os
import csv
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
import matplotlib.pyplot as plt


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


def get_bbox_xz_polygon(translation, size, angle):
    """
    Get XZ-plane polygon for a bounding box.
    Args:
        translation: [x, y, z] center position
        size: [sx, sy, sz] HALF-sizes (half-extents from center)
        angle: rotation angle around Y axis
    Returns:
        Shapely Polygon representing the bbox footprint in XZ plane
    """
    # Create corners of the bbox in local coordinates (XZ plane)
    x, y, z = translation
    sx, sy, sz = size
    
    # Sizes are already half-dimensions (half-extents)
    half_x, half_z = sx/2, sz/2
    
    # Corners in local frame (before rotation)
    corners = np.array([
        [-half_x, -half_z],
        [half_x, -half_z],
        [half_x, half_z],
        [-half_x, half_z]
    ])
    
    # Rotation matrix for Y-axis rotation (in XZ plane)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Rotate and translate
    rotated_corners = corners @ rotation.T
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += z
    
    return Polygon(rotated_corners)


def get_floor_polygon(floor_corners):
    """
    Convert floor ordered corners to a Shapely polygon (XZ plane).
    Args:
        floor_corners: Nx2 array of ordered floor corners in XZ plane
    """
    return Polygon(floor_corners)


def is_ceiling_object(class_label):
    """Check if object is a ceiling-mounted object to exclude"""
    ceiling_objects = ['pendant_lamp', 'ceiling_lamp', 'chandelier']
    return class_label in ceiling_objects


def analyze_scene_free_space(data, class_labels_list):
    """
    Analyze free space in a single scene.
    Returns dict with metrics.
    """
    # Get floor polygon from ordered corners
    floor_corners = data["floor_plan_ordered_corners"]  # Nx2 array (XZ coordinates)
    floor_polygon = get_floor_polygon(floor_corners)
    total_floor_area = floor_polygon.area
    
    # Get all bounding boxes (excluding ceiling objects)
    translations = data["translations"]
    sizes = data["sizes"]
    angles = data["angles"].flatten()
    class_labels = data["class_labels"]
    
    # Create list of object footprint polygons
    object_polygons = []
    for i in range(len(translations)):
        # Decode class label
        class_idx = np.argmax(class_labels[i])
        class_name = class_labels_list[class_idx] if class_idx < len(class_labels_list) else "unknown"
        
        # Skip ceiling objects
        if is_ceiling_object(class_name):
            continue
        
        # Create polygon for this object's footprint
        poly = get_bbox_xz_polygon(translations[i], sizes[i], angles[i])
        object_polygons.append(poly)
    
    # Union all object polygons to get total occupied area
    if len(object_polygons) > 0:
        occupied_union = unary_union(object_polygons)
        occupied_area = occupied_union.area
    else:
        occupied_area = 0.0
    
    # Calculate free space
    free_area = total_floor_area - occupied_area
    free_ratio = free_area / total_floor_area if total_floor_area > 0 else 0.0
    
    # Calculate free space polygon (useful for more detailed analysis)
    if len(object_polygons) > 0:
        try:
            free_space_polygon = floor_polygon.difference(occupied_union)
            # Get largest contiguous free space
            if isinstance(free_space_polygon, MultiPolygon):
                largest_free_area = max([p.area for p in free_space_polygon.geoms])
            else:
                largest_free_area = free_space_polygon.area
        except:
            largest_free_area = free_area
    else:
        largest_free_area = free_area
    
    return {
        "total_floor_area": total_floor_area,
        "occupied_area": occupied_area,
        "free_area": free_area,
        "free_ratio": free_ratio,
        "largest_free_area": largest_free_area,
        "num_objects": len(object_polygons),
        "num_total_objects": len(translations)
    }


def plot_distributions(stats_train, stats_val):
    """Plot free space distributions"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Free ratio distribution
    axes[0, 0].hist(stats_train['free_ratios'], bins=50, alpha=0.7, label='Train', color='blue')
    axes[0, 0].hist(stats_val['free_ratios'], bins=50, alpha=0.7, label='Val', color='orange')
    axes[0, 0].set_xlabel('Free Space Ratio')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Free Space Ratio Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Total floor area
    axes[0, 1].hist(stats_train['floor_areas'], bins=50, alpha=0.7, label='Train', color='blue')
    axes[0, 1].hist(stats_val['floor_areas'], bins=50, alpha=0.7, label='Val', color='orange')
    axes[0, 1].set_xlabel('Total Floor Area (m²)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Total Floor Area Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Free area
    axes[0, 2].hist(stats_train['free_areas'], bins=50, alpha=0.7, label='Train', color='blue')
    axes[0, 2].hist(stats_val['free_areas'], bins=50, alpha=0.7, label='Val', color='orange')
    axes[0, 2].set_xlabel('Free Area (m²)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Free Area Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Occupied area
    axes[1, 0].hist(stats_train['occupied_areas'], bins=50, alpha=0.7, label='Train', color='blue')
    axes[1, 0].hist(stats_val['occupied_areas'], bins=50, alpha=0.7, label='Val', color='orange')
    axes[1, 0].set_xlabel('Occupied Area (m²)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Occupied Area Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Largest contiguous free space
    axes[1, 1].hist(stats_train['largest_free_areas'], bins=50, alpha=0.7, label='Train', color='blue')
    axes[1, 1].hist(stats_val['largest_free_areas'], bins=50, alpha=0.7, label='Val', color='orange')
    axes[1, 1].set_xlabel('Largest Contiguous Free Area (m²)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Largest Contiguous Free Area Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Free ratio vs floor area scatter
    axes[1, 2].scatter(stats_train['floor_areas'], stats_train['free_ratios'], 
                       alpha=0.3, s=10, label='Train', color='blue')
    axes[1, 2].scatter(stats_val['floor_areas'], stats_val['free_ratios'], 
                       alpha=0.3, s=10, label='Val', color='orange')
    axes[1, 2].set_xlabel('Total Floor Area (m²)')
    axes[1, 2].set_ylabel('Free Space Ratio')
    axes[1, 2].set_title('Free Ratio vs Floor Area')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('free_space_analysis.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'free_space_analysis.png'")


def main():
    """Main analysis function"""
    # Configuration
    path_to_dataset_files = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/dataset_files"
    path_to_processed_data = "/mnt/sv-share/MiData/"
    dataset_directory = os.path.join(path_to_processed_data, "bedroom")
    annotation_file = os.path.join(path_to_dataset_files, "bedroom_threed_front_splits_original.csv")
    
    # Load class labels (bedroom furniture types)
    # This is a typical list for bedroom scenes
    class_labels_list = [
        'wardrobe', 'nightstand', 'double_bed', 'single_bed', 'kids_bed',
        'desk', 'dressing_table', 'table', 'chair', 'armchair',
        'bookshelf', 'drawer', 'tv_stand', 'stool', 'pendant_lamp',
        'ceiling_lamp', 'floor_lamp', 'cabinet', 'chest', 'lounge_chair',
        'shelving', 'round_end_table', 'corner_side_table', 'side_table',
        'coffee_table', 'dining_table', 'dining_chair'
    ]
    
    # Analyze both train and val splits
    for split in ['train', 'val']:
        print(f"\n{'='*60}")
        print(f"Analyzing {split.upper()} set")
        print(f"{'='*60}")
        
        scene_ids = load_scene_ids_from_csv(annotation_file, split=split)
        print(f"Found {len(scene_ids)} {split} scenes")
        
        # Find matching directories
        all_dirs = sorted(os.listdir(dataset_directory))
        scene_dirs = [d for d in all_dirs if any(d.endswith(scene_id) for scene_id in scene_ids)]
        print(f"Found {len(scene_dirs)} scene directories")
        
        # Collect statistics
        free_ratios = []
        floor_areas = []
        free_areas = []
        occupied_areas = []
        largest_free_areas = []
        num_objects_list = []
        
        for scene_dir in tqdm(scene_dirs, desc=f"Processing {split} scenes"):
            npz_path = os.path.join(dataset_directory, scene_dir, "boxes.npz")
            if not os.path.exists(npz_path):
                continue
            
            try:
                data = np.load(npz_path)
                metrics = analyze_scene_free_space(data, class_labels_list)
                
                free_ratios.append(metrics['free_ratio'])
                floor_areas.append(metrics['total_floor_area'])
                free_areas.append(metrics['free_area'])
                occupied_areas.append(metrics['occupied_area'])
                largest_free_areas.append(metrics['largest_free_area'])
                num_objects_list.append(metrics['num_objects'])
                
            except Exception as e:
                print(f"Error processing {scene_dir}: {e}")
                continue
        
        # Convert to numpy arrays
        free_ratios = np.array(free_ratios)
        floor_areas = np.array(floor_areas)
        free_areas = np.array(free_areas)
        occupied_areas = np.array(occupied_areas)
        largest_free_areas = np.array(largest_free_areas)
        
        # Print statistics
        print(f"\n{split.upper()} SET STATISTICS:")
        print(f"{'─'*60}")
        print(f"Number of scenes analyzed: {len(free_ratios)}")
        
        print(f"\nFree Space Ratio:")
        print(f"  Mean: {np.mean(free_ratios):.3f}")
        print(f"  Median: {np.median(free_ratios):.3f}")
        print(f"  Std: {np.std(free_ratios):.3f}")
        print(f"  Min: {np.min(free_ratios):.3f}")
        print(f"  Max: {np.max(free_ratios):.3f}")
        print(f"  25th percentile: {np.percentile(free_ratios, 25):.3f}")
        print(f"  75th percentile: {np.percentile(free_ratios, 75):.3f}")
        
        print(f"\nTotal Floor Area (m²):")
        print(f"  Mean: {np.mean(floor_areas):.2f}")
        print(f"  Median: {np.median(floor_areas):.2f}")
        print(f"  Std: {np.std(floor_areas):.2f}")
        print(f"  Min: {np.min(floor_areas):.2f}")
        print(f"  Max: {np.max(floor_areas):.2f}")
        
        print(f"\nFree Area (m²):")
        print(f"  Mean: {np.mean(free_areas):.2f}")
        print(f"  Median: {np.median(free_areas):.2f}")
        print(f"  Std: {np.std(free_areas):.2f}")
        print(f"  Min: {np.min(free_areas):.2f}")
        print(f"  Max: {np.max(free_areas):.2f}")
        
        print(f"\nOccupied Area (m²):")
        print(f"  Mean: {np.mean(occupied_areas):.2f}")
        print(f"  Median: {np.median(occupied_areas):.2f}")
        print(f"  Std: {np.std(occupied_areas):.2f}")
        
        print(f"\nLargest Contiguous Free Area (m²):")
        print(f"  Mean: {np.mean(largest_free_areas):.2f}")
        print(f"  Median: {np.median(largest_free_areas):.2f}")
        print(f"  Min: {np.min(largest_free_areas):.2f}")
        
        print(f"\nNavigability Thresholds:")
        print(f"  Scenes with >30% free space: {np.sum(free_ratios > 0.3)} ({np.sum(free_ratios > 0.3)/len(free_ratios)*100:.1f}%)")
        print(f"  Scenes with >40% free space: {np.sum(free_ratios > 0.4)} ({np.sum(free_ratios > 0.4)/len(free_ratios)*100:.1f}%)")
        print(f"  Scenes with >50% free space: {np.sum(free_ratios > 0.5)} ({np.sum(free_ratios > 0.5)/len(free_ratios)*100:.1f}%)")
        print(f"  Scenes with <20% free space: {np.sum(free_ratios < 0.2)} ({np.sum(free_ratios < 0.2)/len(free_ratios)*100:.1f}%)")
        
        print(f"\nMinimum Free Space for Navigability:")
        print(f"  Scenes with >2 m² free: {np.sum(free_areas > 2.0)} ({np.sum(free_areas > 2.0)/len(free_areas)*100:.1f}%)")
        print(f"  Scenes with >3 m² free: {np.sum(free_areas > 3.0)} ({np.sum(free_areas > 3.0)/len(free_areas)*100:.1f}%)")
        print(f"  Scenes with >5 m² free: {np.sum(free_areas > 5.0)} ({np.sum(free_areas > 5.0)/len(free_areas)*100:.1f}%)")
        
        # Store for plotting
        if split == 'train':
            stats_train = {
                'free_ratios': free_ratios,
                'floor_areas': floor_areas,
                'free_areas': free_areas,
                'occupied_areas': occupied_areas,
                'largest_free_areas': largest_free_areas
            }
        else:
            stats_val = {
                'free_ratios': free_ratios,
                'floor_areas': floor_areas,
                'free_areas': free_areas,
                'occupied_areas': occupied_areas,
                'largest_free_areas': largest_free_areas
            }
    
    # Plot distributions
    print(f"\n{'='*60}")
    print("Generating plots...")
    plot_distributions(stats_train, stats_val)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
