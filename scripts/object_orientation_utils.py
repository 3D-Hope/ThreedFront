"""
Object Orientation Utilities for 3D Scene Layout
=================================================

This module provides utilities for computing and evaluating object orientations
in 3D scenes, particularly for determining if objects are facing each other.

Coordinate System Convention (3D-FRONT):
- Y-axis: vertical (up)
- X-axis: horizontal (right)
- Z-axis: horizontal (forward/backward)
- Rotation angle: around Y-axis
- Facing direction after rotation by θ: [sin(θ), 0, cos(θ)]
- In XZ plane (2D): [sin(θ), cos(θ)]

Usage:
    from object_orientation_utils import compute_facing_reward, angle_to_direction_2d
    
    reward = compute_facing_reward(
        obj1_pos=[0, 0.5, 0],
        obj1_angle=np.pi,
        obj2_pos=[3, 0.5, 0],
        obj2_angle=0.0
    )
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Union, Optional


def normalize_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Normalize angle to [-π, π] range.
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle in [-π, π]
    """
    if isinstance(angle, np.ndarray):
        result = angle.copy()
        result = np.mod(result + np.pi, 2 * np.pi) - np.pi
        return result
    else:
        return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def angle_to_direction_2d(angle: float, convention: str = 'sin_cos') -> np.ndarray:
    """Convert rotation angle to 2D direction vector in XZ plane.
    
    Args:
        angle: Rotation angle around Y-axis in radians
        convention: Direction vector convention
            - 'sin_cos': [sin(θ), cos(θ)] (3D-FRONT default)
            - 'cos_sin': [cos(θ), sin(θ)]
            
    Returns:
        2D direction vector [x, z] representing facing direction
        
    Examples:
        >>> angle_to_direction_2d(0.0)  # Facing +Z
        array([0., 1.])
        >>> angle_to_direction_2d(np.pi/2)  # Facing +X
        array([1., 0.])
        >>> angle_to_direction_2d(np.pi)  # Facing -Z
        array([0., -1.])
    """
    if convention == 'sin_cos':
        return np.array([np.sin(angle), np.cos(angle)])
    elif convention == 'cos_sin':
        return np.array([np.cos(angle), np.sin(angle)])
    else:
        raise ValueError(f"Unknown convention: {convention}")


def compute_direction_between_positions(
    pos1: np.ndarray, 
    pos2: np.ndarray, 
    normalize: bool = True
) -> np.ndarray:
    """Compute direction vector from pos1 to pos2 in XZ plane.
    
    Args:
        pos1: 3D position [x, y, z]
        pos2: 3D position [x, y, z]
        normalize: Whether to normalize to unit vector
        
    Returns:
        2D direction vector [x, z] from pos1 to pos2
    """
    # Project to XZ plane (ignore Y/height)
    dir_vec = np.array([pos2[0] - pos1[0], pos2[2] - pos1[2]])
    
    if normalize:
        norm = np.linalg.norm(dir_vec)
        if norm > 1e-6:
            dir_vec = dir_vec / norm
    
    return dir_vec


def compute_angle_between_vectors(
    v1: np.ndarray, 
    v2: np.ndarray, 
    in_degrees: bool = False
) -> float:
    """Compute angle between two 2D vectors.
    
    Args:
        v1: First vector [x, z]
        v2: Second vector [x, z]
        in_degrees: Return angle in degrees instead of radians
        
    Returns:
        Angle between vectors in [0, π] (or [0, 180°])
        
    Examples:
        >>> v1 = np.array([1, 0])
        >>> v2 = np.array([0, 1])
        >>> compute_angle_between_vectors(v1, v2, in_degrees=True)
        90.0
    """
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    if in_degrees:
        return np.rad2deg(angle)
    return angle


def compute_alignment_score(
    v1: np.ndarray,
    v2: np.ndarray,
    scoring: str = 'cosine_normalized'
) -> float:
    """Compute alignment score between two direction vectors.
    
    Args:
        v1: First direction vector [x, z]
        v2: Second direction vector [x, z]
        scoring: Scoring method
            - 'cosine_normalized': (cos + 1) / 2, maps [-1,1] to [0,1]
            - 'cosine_positive': max(0, cos), clamps negative to 0
            - 'angle_based': 1 - angle/π, uses angle between vectors
            
    Returns:
        Alignment score in [0, 1], where:
        - 1.0: vectors pointing same direction (perfect alignment)
        - 0.5: vectors perpendicular (neutral)
        - 0.0: vectors pointing opposite directions (worst)
        
    Examples:
        >>> v1 = np.array([1, 0])
        >>> v2 = np.array([1, 0])
        >>> compute_alignment_score(v1, v2)  # Same direction
        1.0
        >>> v2 = np.array([-1, 0])
        >>> compute_alignment_score(v1, v2)  # Opposite
        0.0
    """
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    
    if scoring == 'cosine_normalized':
        cos_sim = np.dot(v1_norm, v2_norm)
        return (cos_sim + 1.0) / 2.0
    
    elif scoring == 'cosine_positive':
        cos_sim = np.dot(v1_norm, v2_norm)
        return max(0.0, cos_sim)
    
    elif scoring == 'angle_based':
        angle = compute_angle_between_vectors(v1, v2)
        return 1.0 - angle / np.pi
    
    else:
        raise ValueError(f"Unknown scoring method: {scoring}")


def are_objects_facing_each_other(
    pos1: np.ndarray,
    angle1: float,
    pos2: np.ndarray,
    angle2: float,
    threshold_degrees: float = 45.0,
    convention: str = 'sin_cos'
) -> Tuple[bool, dict]:
    """Determine if two objects are facing each other.
    
    Args:
        pos1: Position of object 1 [x, y, z]
        angle1: Rotation angle of object 1 (radians)
        pos2: Position of object 2 [x, y, z]
        angle2: Rotation angle of object 2 (radians)
        threshold_degrees: Maximum angle deviation to consider "facing"
        convention: Direction vector convention
        
    Returns:
        Tuple of (are_facing, info_dict) where info_dict contains:
        - obj1_facing_obj2: bool
        - obj2_facing_obj1: bool
        - angle_obj1_to_obj2: angle between obj1's facing and direction to obj2
        - angle_obj2_to_obj1: angle between obj2's facing and direction to obj1
        - facing_angle_diff: angle between the two facing directions
        - distance: distance between objects
        
    Examples:
        >>> # Objects on opposite sides facing each other
        >>> facing, info = are_objects_facing_each_other(
        ...     pos1=[0, 0, 0], angle1=0.0,      # Facing +Z
        ...     pos2=[0, 0, 3], angle2=np.pi     # Facing -Z
        ... )
        >>> print(facing)
        True
    """
    # Get facing directions
    dir1 = angle_to_direction_2d(angle1, convention)
    dir2 = angle_to_direction_2d(angle2, convention)
    
    # Direction from obj1 to obj2
    dir_1_to_2 = compute_direction_between_positions(pos1, pos2)
    dir_2_to_1 = -dir_1_to_2
    
    # Check if obj1 is facing toward obj2
    angle_1_to_2 = compute_angle_between_vectors(dir1, dir_1_to_2)
    
    # Check if obj2 is facing toward obj1
    angle_2_to_1 = compute_angle_between_vectors(dir2, dir_2_to_1)
    
    # Angle between facing directions
    facing_angle_diff = compute_angle_between_vectors(dir1, dir2)
    
    # Distance between objects
    distance = np.linalg.norm(pos2 - pos1)
    
    threshold_rad = np.deg2rad(threshold_degrees)
    obj1_facing_obj2 = angle_1_to_2 < threshold_rad
    obj2_facing_obj1 = angle_2_to_1 < threshold_rad
    
    info = {
        'obj1_facing_obj2': bool(obj1_facing_obj2),
        'obj2_facing_obj1': bool(obj2_facing_obj1),
        'angle_obj1_to_obj2_deg': np.rad2deg(angle_1_to_2),
        'angle_obj2_to_obj1_deg': np.rad2deg(angle_2_to_1),
        'facing_angle_diff_deg': np.rad2deg(facing_angle_diff),
        'distance': distance,
    }
    
    are_facing = obj1_facing_obj2 and obj2_facing_obj1
    
    return are_facing, info


def compute_facing_reward(
    obj1_pos: Union[np.ndarray, list],
    obj1_angle: float,
    obj2_pos: Union[np.ndarray, list],
    obj2_angle: Optional[float] = None,
    mode: str = 'obj1_to_obj2',
    scoring: str = 'cosine_normalized',
    convention: str = 'sin_cos'
) -> float:
    """Compute reward for object facing alignment.
    
    This is the CORRECTED version of the TV viewing angle reward.
    
    Args:
        obj1_pos: Position of object 1 (e.g., bed)
        obj1_angle: Rotation angle of object 1
        obj2_pos: Position of object 2 (e.g., TV)
        obj2_angle: Rotation angle of object 2 (optional, for mutual facing)
        mode: Reward mode
            - 'obj1_to_obj2': Reward how well obj1 faces toward obj2
            - 'mutual': Reward mutual facing (requires obj2_angle)
            - 'obj2_to_obj1': Reward how well obj2 faces toward obj1
        scoring: Scoring method (see compute_alignment_score)
        convention: Direction vector convention
        
    Returns:
        Reward score in [0, 1]
        
    Examples:
        >>> # Bed facing toward TV
        >>> reward = compute_facing_reward(
        ...     obj1_pos=[0, 0, 0], obj1_angle=0.0,  # Bed facing +Z
        ...     obj2_pos=[0, 0, 3], obj2_angle=None  # TV at +Z
        ... )
        >>> print(f"Reward: {reward:.2f}")  # Should be ~1.0
    """
    obj1_pos = np.array(obj1_pos)
    obj2_pos = np.array(obj2_pos)
    
    # Get facing direction of obj1
    obj1_dir = angle_to_direction_2d(obj1_angle, convention)
    
    if mode == 'obj1_to_obj2':
        # How well does obj1 face toward obj2?
        target_dir = compute_direction_between_positions(obj1_pos, obj2_pos)
        reward = compute_alignment_score(obj1_dir, target_dir, scoring)
        
    elif mode == 'obj2_to_obj1':
        if obj2_angle is None:
            raise ValueError("obj2_angle required for mode 'obj2_to_obj1'")
        # How well does obj2 face toward obj1?
        obj2_dir = angle_to_direction_2d(obj2_angle, convention)
        target_dir = compute_direction_between_positions(obj2_pos, obj1_pos)
        reward = compute_alignment_score(obj2_dir, target_dir, scoring)
        
    elif mode == 'mutual':
        if obj2_angle is None:
            raise ValueError("obj2_angle required for mode 'mutual'")
        # Average of both directions
        reward1 = compute_facing_reward(obj1_pos, obj1_angle, obj2_pos, None, 
                                       'obj1_to_obj2', scoring, convention)
        reward2 = compute_facing_reward(obj2_pos, obj2_angle, obj1_pos, None,
                                       'obj1_to_obj2', scoring, convention)
        reward = (reward1 + reward2) / 2.0
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return float(reward)


def get_reward_tv_viewing_angle_FIXED(parsed_scene, **kwargs):
    """
    CORRECTED reward for bed/sofa facing the TV.
    
    Key fixes:
    1. Properly maps cosine similarity from [-1, 1] to [0, 1]
    2. Uses correct angle-to-direction convention
    3. Preserves gradient information for all orientations
    
    Args:
        parsed_scene: Dictionary containing:
            - positions: Tensor [batch, max_objects, 3]
            - orientations: Tensor [batch, max_objects, 2] as [sin, cos] or [cos, sin]
            - object_indices: Tensor [batch, max_objects]
            - is_empty: Tensor [batch, max_objects]
            - device: torch device
        **kwargs: Must contain idx_to_labels mapping
        
    Returns:
        Tensor of rewards [batch] in range [0, 1]
    """
    device = parsed_scene["device"]
    object_indices = parsed_scene["object_indices"]
    positions = parsed_scene["positions"]
    orientations = parsed_scene["orientations"]
    is_empty = parsed_scene["is_empty"]
    idx_to_labels = kwargs["idx_to_labels"]
    
    # Handle both integer and string keys
    if idx_to_labels and isinstance(list(idx_to_labels.keys())[0], str):
        idx_to_labels = {int(k): v for k, v in idx_to_labels.items()}
    
    idx_tv = next((k for k, v in idx_to_labels.items() if "tv_stand" in v), None)
    idx_bed = next((k for k, v in idx_to_labels.items() if "bed" in v or "sofa" in v), None)
    if idx_tv is None or idx_bed is None:
        return torch.zeros(len(object_indices), device=device)

    rewards = torch.zeros(len(object_indices), device=device)
    
    for b in range(len(object_indices)):
        try:
            # Get valid mask
            if isinstance(is_empty, torch.Tensor):
                valid_mask = ~is_empty[b]
            else:
                valid_mask = ~torch.tensor(is_empty[b], dtype=torch.bool, device=device)
            
            if isinstance(valid_mask, torch.Tensor):
                if valid_mask.dtype != torch.bool:
                    valid_mask = valid_mask.bool()
            else:
                continue
            
            if valid_mask.sum().item() == 0:
                continue
                
            valid_indices = object_indices[b][valid_mask]
            valid_pos = positions[b][valid_mask]
            valid_orient = orientations[b][valid_mask]
            
            if not isinstance(valid_indices, torch.Tensor):
                continue
            
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
                valid_pos = valid_pos.unsqueeze(0)
                valid_orient = valid_orient.unsqueeze(0)
            
            if valid_indices.numel() == 0:
                continue

            # Check for TV and bed
            tv_mask = (valid_indices == idx_tv)
            bed_mask = (valid_indices == idx_bed)
            
            has_tv = tv_mask.any().item() if isinstance(tv_mask, torch.Tensor) else bool(tv_mask)
            has_bed = bed_mask.any().item() if isinstance(bed_mask, torch.Tensor) else bool(bed_mask)
            
            if not (has_tv and has_bed):
                continue

            tv_pos = valid_pos[valid_indices == idx_tv][0]
            bed_pos = valid_pos[valid_indices == idx_bed][0]
            bed_dir = valid_orient[valid_indices == idx_bed][0]

            # Compute direction from bed to TV (XZ plane only)
            dir_bed_to_tv = tv_pos - bed_pos
            dir_bed_to_tv_2d = torch.stack([dir_bed_to_tv[0], dir_bed_to_tv[2]])
            dir_bed_to_tv_2d = dir_bed_to_tv_2d / (torch.norm(dir_bed_to_tv_2d) + 1e-6)

            # Compute alignment using cosine similarity
            alignment = F.cosine_similarity(
                bed_dir.unsqueeze(0), 
                dir_bed_to_tv_2d.unsqueeze(0)
            )
            
            # FIX: Properly map [-1, 1] to [0, 1] instead of clamping
            # This preserves gradient information for all orientations
            reward = (alignment + 1.0) / 2.0
            
            rewards[b] = reward.item()
            
        except Exception as e:
            print(f"[ERROR] reward_tv_viewing_angle batch {b}: {e}")
            continue

    return rewards


def test_orientation_utils():
    """Test all utility functions with known configurations."""
    print("=" * 80)
    print("TESTING ORIENTATION UTILITIES")
    print("=" * 80)
    print()
    
    # Test 1: Direction conversion
    print("Test 1: Angle to Direction Conversion")
    print("-" * 80)
    test_angles = [0, np.pi/2, np.pi, -np.pi/2]
    test_names = ["0° (+Z)", "90° (+X)", "180° (-Z)", "-90° (-X)"]
    
    for angle, name in zip(test_angles, test_names):
        direction = angle_to_direction_2d(angle)
        print(f"{name:15s}: {direction}")
    print()
    
    # Test 2: Facing each other
    print("Test 2: Objects Facing Each Other")
    print("-" * 80)
    
    test_cases = [
        ("Opposite walls, facing", [0, 0, 0], 0.0, [0, 0, 3], np.pi, True),
        ("Opposite walls, NOT facing", [0, 0, 0], 0.0, [0, 0, 3], 0.0, False),
        ("Perpendicular walls, facing", [0, 0, 0], np.pi/2, [3, 0, 0], -np.pi/2, True),
        ("Same direction", [0, 0, 0], 0.0, [0, 0, -3], 0.0, False),
    ]
    
    for desc, pos1, ang1, pos2, ang2, expected in test_cases:
        facing, info = are_objects_facing_each_other(
            np.array(pos1), ang1, np.array(pos2), ang2
        )
        status = "✓" if facing == expected else "✗"
        print(f"{status} {desc:35s}: facing={facing}, mutual={info['obj1_facing_obj2'] and info['obj2_facing_obj1']}")
    print()
    
    # Test 3: Facing rewards
    print("Test 3: Facing Rewards (should be high for good alignment)")
    print("-" * 80)
    
    reward_cases = [
        ("Perfect alignment", [0, 0, 0], 0.0, [0, 0, 3], 1.0),
        ("Opposite direction", [0, 0, 0], np.pi, [0, 0, 3], 0.0),
        ("Perpendicular", [0, 0, 0], np.pi/2, [0, 0, 3], 0.5),
        ("45° off", [0, 0, 0], np.pi/4, [0, 0, 3], 0.85),
    ]
    
    for desc, pos1, ang1, pos2, expected_approx in reward_cases:
        reward = compute_facing_reward(pos1, ang1, pos2)
        status = "✓" if abs(reward - expected_approx) < 0.15 else "✗"
        print(f"{status} {desc:25s}: reward={reward:.3f} (expected ~{expected_approx:.2f})")
    print()
    
    print("=" * 80)
    print("✓ All orientation utilities tested!")
    print("=" * 80)


if __name__ == "__main__":
    test_orientation_utils()
