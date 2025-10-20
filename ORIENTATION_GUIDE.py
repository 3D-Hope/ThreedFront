"""
3D-FRONT COORDINATE SYSTEM & ORIENTATION GUIDE
===============================================

This guide explains how to work with angles, orientations, and facing directions
in the 3D-FRONT dataset. Use this whenever you need to:
- Calculate which direction an object is facing
- Check if objects are facing each other
- Analyze or reward object orientations

## COORDINATE SYSTEM

3D-FRONT uses a standard 3D coordinate system:
- X-axis: Left-right (horizontal)
- Y-axis: Up-down (vertical height)
- Z-axis: Forward-backward (depth)

Floor plans are viewed from above (top-down), so we work in the XZ plane.

## ROTATION & FACING DIRECTION

### Initial State
Objects initially face the **+Z direction**: `[0, 0, 1]` in 3D

### Rotation Around Y-Axis
Objects rotate around the Y-axis (vertical) by angle θ (z_angle):
```
Rotation Matrix:
[[ cos(θ),  0, sin(θ) ],
 [ 0,       1, 0      ],
 [ -sin(θ), 0, cos(θ) ]]
```

### After Rotation
Facing direction = Rotation Matrix × Initial Direction
```
[[cos(θ),  0, sin(θ)]]   [[0]]   [[sin(θ)]]
[[0,       1, 0     ]] × [[0]] = [[0     ]]
[[-sin(θ), 0, cos(θ)]]   [[1]]   [[cos(θ)]]
```

**Result: Facing direction in 3D = [sin(θ), 0, cos(θ)]**

### In XZ Plane (2D top-down view)
**Facing direction = [sin(θ), cos(θ)]**

## DATASET STORAGE FORMAT

### What's Stored in boxes.npz
```python
data = np.load("boxes.npz")
angles = data['angles']        # Shape: (N, 1) - rotation angle θ in radians
translations = data['translations']  # Shape: (N, 3) - [x, y, z] positions
class_labels = data['class_labels']  # Shape: (N, num_classes) - one-hot
```

### Computing Orientation from Angle
The dataset often stores computed orientations as `[cos(θ), sin(θ)]`:
```python
angle_rad = angles[i]  # Single angle value
orientation = [np.cos(angle_rad), np.sin(angle_rad)]  # Stored format
```

**CRITICAL: This is NOT the facing direction!**

## CONVERTING TO FACING DIRECTION

### From Angle (radians)
```python
angle_rad = angles[i]
facing_xz = np.array([np.sin(angle_rad), np.cos(angle_rad)])
```

### From Orientation [cos(θ), sin(θ)]
```python
orientation = orientations[i]  # [cos(θ), sin(θ)]
facing_xz = np.array([orientation[1], orientation[0]])  # Swap: [sin(θ), cos(θ)]
```

### In PyTorch
```python
# From orientation tensor
bed_orient = orientations[i]  # tensor([cos(θ), sin(θ)])
facing_xz = torch.tensor([bed_orient[1], bed_orient[0]], device=device)
```

## COMMON OPERATIONS

### 1. Check if Object A Faces Object B
```python
import numpy as np

def is_facing(pos_a, angle_a, pos_b, threshold_deg=45):
    """
    Check if object A is facing toward object B.
    
    Args:
        pos_a: Position of object A, shape (3,)
        angle_a: Rotation angle of object A in radians
        pos_b: Position of object B, shape (3,)
        threshold_deg: Angle threshold in degrees
    
    Returns:
        True if A is facing toward B within threshold
    """
    # Get facing direction of A
    facing_a = np.array([np.sin(angle_a), np.cos(angle_a)])
    
    # Direction from A to B (in XZ plane)
    dir_a_to_b = pos_b - pos_a
    dir_a_to_b_xz = np.array([dir_a_to_b[0], dir_a_to_b[2]])
    dir_a_to_b_xz = dir_a_to_b_xz / (np.linalg.norm(dir_a_to_b_xz) + 1e-8)
    
    # Compute angle between facing and direction to B
    dot_product = np.dot(facing_a, dir_a_to_b_xz)
    angle_diff = np.arccos(np.clip(dot_product, -1, 1))
    
    return np.rad2deg(angle_diff) < threshold_deg
```

### 2. Check if Two Objects Face Each Other
```python
def are_facing_each_other(pos_a, angle_a, pos_b, angle_b, threshold_deg=45):
    """
    Check if objects A and B are facing each other.
    
    Returns:
        (bool, dict): (True if facing, detailed info dict)
    """
    # Facing directions
    facing_a = np.array([np.sin(angle_a), np.cos(angle_a)])
    facing_b = np.array([np.sin(angle_b), np.cos(angle_b)])
    
    # Directions between objects
    dir_a_to_b = pos_b - pos_a
    dir_a_to_b_xz = np.array([dir_a_to_b[0], dir_a_to_b[2]])
    dir_a_to_b_xz = dir_a_to_b_xz / (np.linalg.norm(dir_a_to_b_xz) + 1e-8)
    dir_b_to_a_xz = -dir_a_to_b_xz
    
    # Check if A faces B
    dot_a = np.dot(facing_a, dir_a_to_b_xz)
    angle_a_to_b = np.rad2deg(np.arccos(np.clip(dot_a, -1, 1)))
    
    # Check if B faces A
    dot_b = np.dot(facing_b, dir_b_to_a_xz)
    angle_b_to_a = np.rad2deg(np.arccos(np.clip(dot_b, -1, 1)))
    
    # Both must be within threshold
    a_faces_b = angle_a_to_b < threshold_deg
    b_faces_a = angle_b_to_a < threshold_deg
    
    return (a_faces_b and b_faces_a), {
        'angle_a_to_b': angle_a_to_b,
        'angle_b_to_a': angle_b_to_a,
        'a_faces_b': a_faces_b,
        'b_faces_a': b_faces_a,
        'distance': np.linalg.norm(dir_a_to_b),
    }
```

### 3. Compute Reward for Facing
```python
def compute_facing_reward(pos_a, orientation_a, pos_b):
    """
    Compute reward for object A facing toward object B.
    
    Args:
        pos_a: Position (3,)
        orientation_a: [cos(θ), sin(θ)] format
        pos_b: Position (3,)
    
    Returns:
        reward in [0, 1]: 1.0 = perfectly facing, 0.0 = facing away
    """
    # Convert orientation to facing direction
    facing_a = np.array([orientation_a[1], orientation_a[0]])  # [sin, cos]
    
    # Direction from A to B
    dir_a_to_b = pos_b - pos_a
    dir_a_to_b_xz = np.array([dir_a_to_b[0], dir_a_to_b[2]])
    dir_a_to_b_xz = dir_a_to_b_xz / (np.linalg.norm(dir_a_to_b_xz) + 1e-8)
    
    # Cosine similarity: -1 (facing away) to +1 (facing toward)
    alignment = np.dot(facing_a, dir_a_to_b_xz)
    
    # Map to [0, 1]
    reward = (alignment + 1) / 2
    return reward
```

### 4. Find What Direction an Object is Facing
```python
def get_facing_info(angle_rad):
    """
    Get human-readable info about which direction object faces.
    
    Returns:
        dict with facing direction and cardinal direction
    """
    angle_deg = np.rad2deg(angle_rad)
    
    # Normalize to [0, 360)
    while angle_deg < 0:
        angle_deg += 360
    while angle_deg >= 360:
        angle_deg -= 360
    
    # Determine cardinal direction
    if 337.5 <= angle_deg or angle_deg < 22.5:
        direction = "North (+Z)"
    elif 22.5 <= angle_deg < 67.5:
        direction = "Northeast"
    elif 67.5 <= angle_deg < 112.5:
        direction = "East (+X)"
    elif 112.5 <= angle_deg < 157.5:
        direction = "Southeast"
    elif 157.5 <= angle_deg < 202.5:
        direction = "South (-Z)"
    elif 202.5 <= angle_deg < 247.5:
        direction = "Southwest"
    elif 247.5 <= angle_deg < 292.5:
        direction = "West (-X)"
    else:
        direction = "Northwest"
    
    facing_xz = np.array([np.sin(angle_rad), np.cos(angle_rad)])
    
    return {
        'angle_deg': angle_deg,
        'angle_rad': angle_rad,
        'facing_xz': facing_xz,
        'direction': direction
    }
```

## DEBUGGING TIPS

### Visualize Facing Direction
```python
import matplotlib.pyplot as plt

def plot_scene_top_down(positions, angles, labels):
    """Plot scene from above with facing arrows."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for pos, angle, label in zip(positions, angles, labels):
        x, z = pos[0], pos[2]
        
        # Facing direction
        facing = np.array([np.sin(angle), np.cos(angle)])
        dx, dz = facing * 0.3  # Arrow length
        
        # Plot position
        ax.scatter(x, z, s=100)
        ax.text(x, z + 0.1, label, ha='center')
        
        # Plot facing arrow
        ax.arrow(x, z, dx, dz, head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Top-Down View (facing = red arrows)')
    ax.grid(True)
    ax.axis('equal')
    plt.show()
```

### Print Angle Info
```python
def print_object_orientation(name, pos, angle):
    """Print detailed orientation info for debugging."""
    info = get_facing_info(angle)
    facing = info['facing_xz']
    
    print(f"{name}:")
    print(f"  Position (XYZ): {pos}")
    print(f"  Angle: {info['angle_deg']:.1f}° ({info['angle_rad']:.3f} rad)")
    print(f"  Facing direction (XZ): [{facing[0]:.3f}, {facing[1]:.3f}]")
    print(f"  Cardinal direction: {info['direction']}")
```

## COMMON ANGLE VALUES

| Angle (deg) | Angle (rad) | Facing Direction (XZ) | Cardinal |
|-------------|-------------|----------------------|----------|
| 0°          | 0           | [0, 1]              | +Z North |
| 90°         | π/2         | [1, 0]              | +X East  |
| 180°        | π           | [0, -1]             | -Z South |
| -90° / 270° | -π/2        | [-1, 0]             | -X West  |

## QUICK REFERENCE

**From angle to facing:**
```python
facing_xz = [sin(angle), cos(angle)]
```

**From orientation [cos, sin] to facing:**
```python
facing_xz = [orientation[1], orientation[0]]
```

**Check if A faces B:**
```python
dot(facing_A, normalize(pos_B - pos_A)) > cos(threshold)
```

**Reward for facing:**
```python
reward = (dot(facing_A, normalize(pos_B - pos_A)) + 1) / 2
```

## FILES FOR REFERENCE

- `reward_tv_viewing_angle_FINAL.py` - Complete working example
- `analyze_tv_bed_orientation.py` - Dataset analysis with facing checks
- `debug_scenes.py` - Debugging specific scene orientations

Last updated: 2025-10-20
Tested on: 3D-FRONT bedroom dataset (97.1% accuracy)
"""

if __name__ == "__main__":
    # Quick test
    import numpy as np
    
    print("Quick test of orientation calculations:")
    print("=" * 60)
    
    # Object at origin facing +Z (angle = 0)
    pos_a = np.array([0, 0, 0])
    angle_a = 0.0
    
    # Object at [3, 0, 0] facing -X (angle = 180°)
    pos_b = np.array([3, 0, 0])
    angle_b = np.pi
    
    print(f"Object A at {pos_a}, angle = {np.rad2deg(angle_a):.0f}°")
    facing_a = np.array([np.sin(angle_a), np.cos(angle_a)])
    print(f"  Facing direction: {facing_a}")
    
    print(f"\nObject B at {pos_b}, angle = {np.rad2deg(angle_b):.0f}°")
    facing_b = np.array([np.sin(angle_b), np.cos(angle_b)])
    print(f"  Facing direction: {facing_b}")
    
    print(f"\nDirection from A to B: {(pos_b - pos_a)[[0,2]]}")
    print(f"A facing toward B? {np.dot(facing_a, [1, 0]) > 0.7} (dot = {np.dot(facing_a, [1, 0]):.2f})")
    print(f"B facing toward A? {np.dot(facing_b, [-1, 0]) > 0.7} (dot = {np.dot(facing_b, [-1, 0]):.2f})")
