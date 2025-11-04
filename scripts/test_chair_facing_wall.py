import os
import numpy as np
from tqdm import tqdm

    scene_ids = []
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                scene_id, scene_split = parts
                if scene_split == split:
                    scene_ids.append(scene_id)
    return scene_ids

def get_idx_to_labels():
    # You may need to adjust this mapping to match your dataset
    return {
        0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'ceiling_lamp',
        4: 'chair', 5: 'children_cabinet', 6: 'coffee_table', 7: 'desk',
        8: 'double_bed', 9: 'dressing_chair', 10: 'dressing_table',
        11: 'kids_bed', 12: 'nightstand', 13: 'pendant_lamp', 14: 'shelf',
        15: 'single_bed', 16: 'sofa', 17: 'stool', 18: 'table',
        19: 'tv_stand', 20: 'wardrobe', 21: 'dining_chair', 22: 'office_chair',
        23: 'loveseat', 24: 'bench'
    }

def main():
    # Config
    path_to_dataset_files = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/dataset_files"
    path_to_processed_data = "/mnt/sv-share/MiData/"
    dataset_directory = os.path.join(path_to_processed_data, "bedroom")
    annotation_file = os.path.join(path_to_dataset_files, "bedroom_threed_front_splits_original.csv")

    print("Loading scene IDs from CSV...")
    scene_ids = load_scene_ids_from_csv(annotation_file, split="test")
    print(f"Found {len(scene_ids)} test scenes")

    all_dirs = sorted(os.listdir(dataset_directory))
    scene_dirs = [d for d in all_dirs if any(d.endswith(scene_id) for scene_id in scene_ids)]
    print(f"Found {len(scene_dirs)} scene directories")

    idx_to_labels = get_idx_to_labels()
    SEATING_OBJECTS = {
        "chair", "armchair", "sofa", "dining_chair", "office_chair", "loveseat", "bench"
    }
    seating_indices = [idx for idx, label in idx_to_labels.items() if label in SEATING_OBJECTS]

    total_seats = 0
    violation_count = 0
    violation_details = []

    for scene_dir in tqdm(scene_dirs):
        npz_path = os.path.join(dataset_directory, scene_dir, "boxes.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        class_labels = data['class_labels']
        translations = data['translations']
        angles = data['angles']
        sizes = data['sizes'] if 'sizes' in data else np.ones_like(translations)
        object_indices = np.argmax(class_labels, axis=1)
        is_empty = np.zeros(object_indices.shape, dtype=bool)

        # Dummy floor polygon: bounding box of all objects (for demo; replace with real floor if available)
        min_x = np.min(translations[:,0]) - 0.5
        max_x = np.max(translations[:,0]) + 0.5
        min_z = np.min(translations[:,2]) - 0.5
        max_z = np.max(translations[:,2]) + 0.5
        floor_poly = np.array([
            [min_x, min_z], [max_x, min_z], [max_x, max_z], [min_x, max_z]
        ])

        # Prepare batch of 1
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
        }

        reward = compute_seating_accessibility_reward(parsed_scene, floor_polygons, idx_to_labels)
        # For each seat, check if penalty was applied (reward < 0)
        for n, idx in enumerate(object_indices):
            if idx in seating_indices:
                total_seats += 1
                # If penalty for this scene, count as violation (since reward is per scene, not per seat)
                if reward[0].item() < 0:
                    violation_count += 1
                    violation_details.append({
                        'scene': scene_dir,
                        'seat_idx': int(n),
                        'seat_label': idx_to_labels.get(int(idx), str(idx)),
                        'reward': float(reward[0].item())
                    })

    print("\n====================")
    print(f"Total seating objects checked: {total_seats}")
    print(f"Number of scenes with violation: {violation_count}")
    print(f"Violation rate: {violation_count/total_seats*100:.2f}%")
    print("====================")
    print("Sample violations:")
    for v in violation_details[:10]:
        print(f"Scene: {v['scene']} | Seat idx: {v['seat_idx']} | Label: {v['seat_label']} | Reward: {v['reward']:.4f}")

if __name__ == "__main__":
    main()
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon

# Seating set
SEATING_OBJECTS = {
    "chair",
    "armchair",
    "sofa",
    "dining_chair",
    "office_chair",
    "loveseat",
    "bench",
}

def compute_seating_accessibility_reward(parsed_scene, floor_polygons, idx_to_labels, **kwargs):
    """
    Reward that penalizes seating objects that face a wall too closely (so they cannot be seated).
    Signature matches compute_wall_proximity_reward:
      - parsed_scene: dict with 'positions' (B,N,3), 'orientations' (B,N,2), 'sizes' (B,N,3), 'object_indices' (B,N), 'is_empty' (B,N)
      - floor_polygons: list length B of (M,2) arrays or tensors with [x,z] vertices
      - idx_to_labels: dict mapping index -> label string
    Returns:
      - reward: (B,) tensor (negative penalties per scene)
    """
    positions = parsed_scene["positions"]
    orientations = parsed_scene["orientations"]
    sizes = parsed_scene["sizes"]
    object_indices = parsed_scene["object_indices"]
    is_empty = parsed_scene["is_empty"]

    batch_size, num_objects = positions.shape[0], positions.shape[1]
    device = positions.device

    # Find which indices correspond to seating objects
    seating_indices = [int(idx) for idx, label in idx_to_labels.items() if label in SEATING_OBJECTS]

    should_seating_check = torch.zeros_like(is_empty, dtype=torch.bool)
    for idx in seating_indices:
        should_seating_check |= (object_indices == idx)

    valid_mask = ~is_empty & should_seating_check

    # Parameters (tweakable)
    min_clearance = kwargs.get("min_clearance", 0.3)   # meters: below this => cannot sit (hard penalty)
    eps = 1e-6

    penalties = torch.zeros(batch_size, device=device)

    for b in range(batch_size):
        if floor_polygons[b] is None or len(floor_polygons[b]) == 0:
            continue

        poly = floor_polygons[b]
        if not isinstance(poly, torch.Tensor):
            poly = torch.tensor(poly, dtype=torch.float32, device=device)
        else:
            poly = poly.to(device)

        num_edges = poly.shape[0]

        for n in range(num_objects):
            if not valid_mask[b, n]:
                continue

            pos_x = positions[b, n, 0]
            pos_z = positions[b, n, 2]
            size_x = sizes[b, n, 0]
            size_z = sizes[b, n, 2]

            cos_theta = orientations[b, n, 0]
            sin_theta = orientations[b, n, 1]

            # compute forward direction (rounded to 4 cardinal directions for simplicity)
            angle_rad = torch.atan2(sin_theta, cos_theta)
            angle_deg = (angle_rad * 180 / torch.pi).item()
            rounded_angle = round(angle_deg / 90) * 90
            normalized_angle = ((rounded_angle % 360) + 360) % 360

            # For seating we consider the seat-front point (the side occupant faces)
            # Forward = +Z when normalized_angle == 0
            if normalized_angle == 0:
                # forward +Z => seat front at z + size_z
                front_x = pos_x
                front_z = pos_z + size_z
                ray_dx = 0.0
                ray_dz = 1.0
            elif normalized_angle == 90:
                # forward +X => seat front at x + size_x
                front_x = pos_x + size_x
                front_z = pos_z
                ray_dx = 1.0
                ray_dz = 0.0
            elif normalized_angle == 180:
                # forward -Z => seat front at z - size_z
                front_x = pos_x
                front_z = pos_z - size_z
                ray_dx = 0.0
                ray_dz = -1.0
            else:  # 270
                # forward -X => seat front at x - size_x
                front_x = pos_x - size_x
                front_z = pos_z
                ray_dx = -1.0
                ray_dz = 0.0

            # Cast ray from front point along forward direction; find nearest wall intersection distance t
            min_t = float('inf')
            for i in range(num_edges):
                p1 = poly[i]
                p2 = poly[(i + 1) % num_edges]
                edge_x = (p2[0] - p1[0]).item()
                edge_z = (p2[1] - p1[1]).item()

                denom = ray_dx * edge_z - ray_dz * edge_x
                if abs(denom) < 1e-8:
                    continue

                diff_x = p1[0].item() - front_x.item()
                diff_z = p1[1].item() - front_z.item()

                t = (diff_x * edge_z - diff_z * edge_x) / denom
                s = (diff_x * ray_dz - diff_z * ray_dx) / denom

                if t > 0 and 0 <= s <= 1:
                    if t < min_t:
                        min_t = t

            # If ray hits a wall (seat front faces wall), apply penalty depending on how close it is
            if min_t < float('inf'):
                # Hard violation: too close to wall to sit
                if min_t + eps < min_clearance:
                    # quadratic penalty of violation magnitude
                    violation = (min_clearance - min_t)
                    penalties[b] += (violation ** 2)  # positive penalty; reward will be negative sum
                    # print(f"less than clearance {min_t=}, {penalties[b]}")


            # If no wall intersection along forward ray -> no penalty (seat faces open space)

    # reward is negative of penalties (higher penalty -> more negative)
    reward = -penalties
    return reward