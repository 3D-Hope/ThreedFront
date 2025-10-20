"""
Explore boxes.npz File Contents
================================

This script loads a boxes.npz file and displays all available data fields,
their shapes, and sample values. Useful for understanding the data structure.
"""

import numpy as np
from pathlib import Path
import argparse


def explore_npz(npz_path):
    """Load and display all contents of a boxes.npz file."""
    print(f"Loading: {npz_path}")
    print("=" * 100)
    
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    print(f"\nAvailable keys in file: {list(data.keys())}")
    print(f"Total keys: {len(data.keys())}")
    print("\n" + "=" * 100)
    
    # Iterate through all keys
    for key in sorted(data.keys()):
        value = data[key]
        
        print(f"\n[{key}]")
        print("-" * 100)
        print(f"  Type: {type(value).__name__}")
        
        if isinstance(value, np.ndarray):
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
            
            # Show statistics for numeric arrays
            if np.issubdtype(value.dtype, np.number) and value.size > 0:
                print(f"  Min: {value.min()}")
                print(f"  Max: {value.max()}")
                print(f"  Mean: {value.mean():.4f}")
                
            # Show first few values
            if value.size > 0:
                print(f"\n  First few values:")
                if value.ndim == 1:
                    print(f"    {value[:min(5, len(value))]}")
                elif value.ndim == 2:
                    print(f"    Shape: {value.shape}")
                    for i in range(min(3, value.shape[0])):
                        print(f"    Row {i}: {value[i]}")
                elif value.ndim == 3:
                    print(f"    Shape: {value.shape}")
                    print(f"    First slice: {value[0]}")
                else:
                    print(f"    {value.flatten()[:10]}...")
            else:
                print(f"  (empty array)")
                
        else:
            print(f"  Value: {value}")
    
    print("\n" + "=" * 100)
    
    # Special focus on floor plan related fields
    print("\n🔍 FLOOR PLAN / ROOM BOUNDARY FIELDS:")
    print("=" * 100)
    
    floor_related = [k for k in data.keys() if 'floor' in k.lower() or 'room' in k.lower() 
                     or 'corner' in k.lower() or 'wall' in k.lower() or 'boundary' in k.lower()]
    
    if floor_related:
        for key in floor_related:
            value = data[key]
            print(f"\n✓ Found: {key}")
            print(f"  Type: {type(value).__name__}")
            if isinstance(value, np.ndarray):
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                print(f"  Data preview:")
                if value.size > 0 and value.size <= 50:
                    print(f"    {value}")
                elif value.size > 50:
                    print(f"    {value.flatten()[:20]}...")
                    print(f"    ... ({value.size} total elements)")
    else:
        print("\n⚠️  No floor plan related fields found with keywords:")
        print("    'floor', 'room', 'corner', 'wall', 'boundary'")
    
    print("\n" + "=" * 100)
    
    # Show object information if available
    if 'class_labels' in data:
        print("\n📦 OBJECT INFORMATION:")
        print("=" * 100)
        
        class_labels = data['class_labels']
        num_objects = class_labels.shape[0]
        
        idx_to_labels = {
            0: "armchair", 1: "bookshelf", 2: "cabinet", 3: "ceiling_lamp",
            4: "chair", 5: "children_cabinet", 6: "coffee_table", 7: "desk",
            8: "double_bed", 9: "dressing_chair", 10: "dressing_table",
            11: "kids_bed", 12: "nightstand", 13: "pendant_lamp", 14: "shelf",
            15: "single_bed", 16: "sofa", 17: "stool", 18: "table",
            19: "tv_stand", 20: "wardrobe",
        }
        
        class_indices = np.argmax(class_labels, axis=1)
        
        print(f"\nTotal objects in scene: {num_objects}")
        print("\nObjects list:")
        for i, idx in enumerate(class_indices):
            obj_name = idx_to_labels.get(idx, f"Unknown({idx})")
            
            pos = data['translations'][i] if 'translations' in data else None
            angle = data['angles'][i] if 'angles' in data else None
            
            info = f"  [{i}] {obj_name}"
            if pos is not None:
                info += f" @ pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
            if angle is not None:
                angle_deg = np.rad2deg(angle[0]) if len(angle.shape) > 0 else np.rad2deg(angle)
                info += f" angle={angle_deg:.1f}°"
            
            print(info)
    
    print("\n" + "=" * 100)
    print("\n✅ Exploration complete!")
    

def main():
    parser = argparse.ArgumentParser(description="Explore contents of boxes.npz file")
    parser.add_argument(
        "npz_path",
        nargs="?",
        help="Path to boxes.npz file (if not provided, uses first scene in dataset)"
    )
    parser.add_argument(
        "--dataset",
        default="/mnt/sv-share/MiData/bedroom",
        help="Dataset directory to find a scene"
    )
    
    args = parser.parse_args()
    
    if args.npz_path:
        npz_path = Path(args.npz_path)
    else:
        # Find first scene in dataset
        dataset_dir = Path(args.dataset)
        scene_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        
        if not scene_dirs:
            print(f"No scenes found in {dataset_dir}")
            return
        
        # Try first few scenes until we find one with boxes.npz
        npz_path = None
        for scene_dir in scene_dirs[:10]:
            test_path = scene_dir / "boxes.npz"
            if test_path.exists():
                npz_path = test_path
                break
        
        if not npz_path:
            print(f"No boxes.npz files found in first 10 scenes of {dataset_dir}")
            return
    
    if not npz_path.exists():
        print(f"File not found: {npz_path}")
        return
    
    explore_npz(npz_path)


if __name__ == "__main__":
    main()
