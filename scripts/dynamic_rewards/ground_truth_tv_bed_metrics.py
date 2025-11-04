
import os
import numpy as np
from tqdm import tqdm

def load_scene_ids_from_csv(csv_path, split="test"):
    scene_ids = []
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                scene_id, scene_split = parts
                if scene_split == split:
                    scene_ids.append(scene_id)
    return scene_ids

def calculate_metrics(layouts, ideal_distance=3.0, distance_range=(2.0, 4.0), angle_threshold=0.7):
    tv_idx = 19
    bed_indices = [8, 15, 11]
    total_scenes = len(layouts)
    scenes_with_tv = 0
    scenes_with_bed = 0
    scenes_with_both = 0
    distances = []
    distances_in_range = 0
    viewing_angles = []
    good_viewing_angles = 0
    for layout in layouts:
        class_labels_one_hot = layout['class_labels']
        translations = layout['translations']
        angles = layout['angles']
        object_indices = np.argmax(class_labels_one_hot, axis=1)
        tv_mask = (object_indices == tv_idx)
        bed_mask = np.isin(object_indices, bed_indices)
        has_tv = np.any(tv_mask)
        has_bed = np.any(bed_mask)
        if has_tv:
            scenes_with_tv += 1
        if has_bed:
            scenes_with_bed += 1
        if has_tv and has_bed:
            scenes_with_both += 1
            tv_pos = translations[tv_mask][0]
            bed_pos = translations[bed_mask][0]
            distance_vec = tv_pos - bed_pos
            distance_2d = np.sqrt(distance_vec[0]**2 + distance_vec[2]**2)
            distances.append(distance_2d)
            if distance_range[0] <= distance_2d <= distance_range[1]:
                distances_in_range += 1
            bed_angle = angles[bed_mask][0][0]
            bed_direction = np.array([np.cos(bed_angle), np.sin(bed_angle)])
            dir_bed_to_tv = np.array([distance_vec[0], distance_vec[2]])
            dir_bed_to_tv_norm = dir_bed_to_tv / (np.linalg.norm(dir_bed_to_tv) + 1e-6)
            alignment = np.dot(bed_direction, dir_bed_to_tv_norm)
            alignment = np.clip(alignment, 0, 1)
            viewing_angles.append(alignment)
            if alignment >= angle_threshold:
                good_viewing_angles += 1
    tv_presence_rate = (scenes_with_tv / total_scenes) * 100 if total_scenes else 0.0
    bed_presence_rate = (scenes_with_bed / total_scenes) * 100 if total_scenes else 0.0
    both_present_rate = (scenes_with_both / total_scenes) * 100 if total_scenes else 0.0
    if distances:
        distance_in_range_rate = (distances_in_range / len(distances)) * 100
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
    else:
        distance_in_range_rate = 0.0
        mean_distance = std_distance = min_distance = max_distance = 0.0
    if viewing_angles:
        good_angle_rate = (good_viewing_angles / len(viewing_angles)) * 100
        mean_angle = np.mean(viewing_angles)
        std_angle = np.std(viewing_angles)
        min_angle = np.min(viewing_angles)
        max_angle = np.max(viewing_angles)
    else:
        good_angle_rate = 0.0
        mean_angle = std_angle = min_angle = max_angle = 0.0
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
    if metrics is None:
        return
    print("\n" + "=" * 60)
    print("GROUND TRUTH CONSTRAINT SATISFACTION METRICS")
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

def main():
    # Hardcoded config for ground truth bedroom dataset
    path_to_dataset_files = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/dataset_files"
    path_to_processed_data = "/mnt/sv-share/MiData/"
    dataset_directory = os.path.join(path_to_processed_data, "bedroom")
    annotation_file = os.path.join(path_to_dataset_files, "bedroom_threed_front_splits_original.csv")

    print("Loading scene IDs from CSV...")
    scene_ids = load_scene_ids_from_csv(annotation_file, split="test")
    print(f"Found {len(scene_ids)} test scenes")

    # Find all scene directories
    all_dirs = sorted(os.listdir(dataset_directory))
    scene_dirs = [d for d in all_dirs if any(d.endswith(scene_id) for scene_id in scene_ids)]
    print(f"Found {len(scene_dirs)} scene directories")

    layouts = []
    for scene_dir in tqdm(scene_dirs):
        npz_path = os.path.join(dataset_directory, scene_dir, "boxes.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        layout = {
            'class_labels': data['class_labels'],
            'translations': data['translations'],
            'angles': data['angles']
        }
        layouts.append(layout)

    metrics = calculate_metrics(layouts)
    print_metrics(metrics)

if __name__ == "__main__":
    main()
