# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler

# python render_threedfront_scene.py SecondBedroom-44725 ../outputs /mnt/sv-share/3DFRONT/data/3D-FRONT /mnt/sv-share/3DFRONT/data/3D-FUTURE-model /mnt/sv-share/3DFRONT/3D-FUTURE-model/model_info.json /home/ajad/AshokSaugatResearch/ATISS/demo/floor_plan_texture_images
# 

"""Script used for visualizing 3D-FRONT room specified by its scene_id."""
import argparse
import logging
import os
import sys
import pickle
import math

import numpy as np
from PIL import Image
import pyrr
import trimesh

from scene_synthesis.datasets.threed_front import ThreedFront

from simple_3dviz import Scene, Mesh
from simple_3dviz.behaviours.keyboard import SnapshotOnKey
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.renderables.textured_mesh import TexturedMesh
from simple_3dviz.utils import render, save_frame
from simple_3dviz.window import show

from utils import floor_plan_from_scene, export_scene


def create_gif_from_frames(frame_paths, output_path, duration=100):
    """Create a GIF from a list of frame paths."""
    images = []
    for frame_path in frame_paths:
        if os.path.exists(frame_path):
            img = Image.open(frame_path)
            images.append(img)
    
    if images:
        # Save as GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF created: {output_path}")
    else:
        print("No frames found to create GIF")


def generate_camera_positions(center, radius, height, num_frames=36):
    """Generate camera positions in a circle around the center."""
    positions = []
    for i in range(num_frames):
        angle = 2 * math.pi * i / num_frames
        x = center[0] + radius * math.cos(angle)
        z = center[2] + radius * math.sin(angle)
        y = center[1] + height
        positions.append((x, y, z))
    return positions


def scene_init(mesh, up_vector, camera_position, camera_target, background):
    def inner(scene):
        scene.background = background
        scene.up_vector = up_vector
        scene.camera_position = camera_position
        scene.camera_target = camera_target
        scene.light = camera_position
        if mesh is not None:
            scene.add(mesh)
    return inner


def main(argv):
    parser = argparse.ArgumentParser(
        description="Visualize a 3D-FRONT room from json file"
    )
    parser.add_argument(
        "scene_id",
        help="The scene id of the scene to be visualized"
    )
    parser.add_argument(
        "output_directory",
        help="Path to output directory"
    )
    parser.add_argument(
        "path_to_3d_front_dataset_directory",
        help="Path to the 3D-FRONT dataset"
    )
    parser.add_argument(
        "path_to_3d_future_dataset_directory",
        help="Path to the 3D-FUTURE dataset"
    )
    parser.add_argument(
        "path_to_model_info",
        help="Path to the 3D-FUTURE model_info.json file"
    )
    parser.add_argument(
        "path_to_floor_plan_textures",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--annotation_file",
        default="../config/bedroom_threed_front_splits.csv",
        help="Path to the train/test splits file"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,-1,0",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="2.0,0.2,2.0",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--with_orthographic_projection",
        action="store_true",
        help="Use orthographic projection"
    )
    parser.add_argument(
        "--with_floor_layout",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--with_walls",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--with_door_and_windows",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--with_texture",
        action="store_true",
        help="Visualize objects with texture"
    )
    parser.add_argument(
        "--export_gif",
        action="store_true",
        help="Export scene as animated GIF"
    )
    parser.add_argument(
        "--gif_frames",
        type=int,
        default=36,
        help="Number of frames for GIF animation"
    )
    parser.add_argument(
        "--gif_duration",
        type=int,
        default=100,
        help="Duration per frame in milliseconds for GIF"
    )
    parser.add_argument(
        "--gif_radius",
        type=float,
        default=3.0,
        help="Radius for camera rotation in GIF"
    )
    parser.add_argument(
        "--gif_height",
        type=float,
        default=2.0,
        help="Height for camera position in GIF"
    )

    args = parser.parse_args(argv)
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size)
    if args.with_orthographic_projection:
        scene.camera_matrix = pyrr.Matrix44.orthogonal_projection(
            left=-3.1, right=3.1, bottom=-3.1, top=3.1, near=0.1, far=1000
        )
    scene.light = args.camera_position
    behaviours = []

    if os.path.exists("/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_front.pkl"):
        print("Loading dataset from file")
        with open("/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_front.pkl", "rb") as f:
            d = pickle.load(f)
    else:
        d = ThreedFront.from_dataset_directory(
            args.path_to_3d_front_dataset_directory,
            args.path_to_model_info,
            args.path_to_3d_future_dataset_directory,
            path_to_room_masks_dir=None,
            path_to_bounds=None,
            filter_fn=lambda s: s
        )
        with open("/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/output/threed_front.pkl", "wb") as f:
            pickle.dump(d, f)
    print("Loading dataset with {} rooms".format(len(d)))

    for s in d.scenes:
        if s.scene_id == args.scene_id:
            for b in s.bboxes:
                print(b.model_jid, b.label)
            print(s.furniture_in_room, s.scene_id, s.json_path)
            renderables = s.furniture_renderables(
                with_floor_plan_offset=True, with_texture=args.with_texture
            )
            trimesh_meshes = []
            for furniture in s.bboxes:
                # Load the furniture and scale it as it is given in the dataset
                # raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                raw_mesh = Mesh.from_file(furniture.raw_model_path)
                raw_mesh.scale(furniture.scale)

                # Create a trimesh object for the same mesh in order to save
                # everything as a single scene
                tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
                tr_mesh.visual.material.image = Image.open(
                    furniture.texture_image_path
                )
                tr_mesh.vertices *= furniture.scale
                theta = furniture.z_angle
                R = np.zeros((3, 3))
                R[0, 0] = np.cos(theta)
                R[0, 2] = -np.sin(theta)
                R[2, 0] = np.sin(theta)
                R[2, 2] = np.cos(theta)
                R[1, 1] = 1.
                tr_mesh.vertices[...] = \
                    tr_mesh.vertices.dot(R) + furniture.position
                tr_mesh.vertices[...] = tr_mesh.vertices - s.centroid
                trimesh_meshes.append(tr_mesh)

            if args.with_floor_layout:
                # Get a floor plan
                floor_plan, tr_floor, _ = floor_plan_from_scene(
                    s, args.path_to_floor_plan_textures, without_room_mask=True
                )
                renderables += floor_plan
                trimesh_meshes += tr_floor

            if args.with_walls:
                for ei in s.extras:
                    if "WallInner" in ei.model_type:
                        renderables = renderables + [
                            ei.mesh_renderable(
                                offset=-s.centroid,
                                colors=(0.8, 0.8, 0.8, 0.6)
                            )
                        ]

            if args.with_door_and_windows:
                for ei in s.extras:
                    if "Window" in ei.model_type or "Door" in ei.model_type:
                        renderables = renderables + [
                            ei.mesh_renderable(
                                offset=-s.centroid,
                                colors=(0.8, 0.8, 0.8, 0.6)
                            )
                        ]

            if args.export_gif:
                # Generate camera positions for GIF
                camera_positions = generate_camera_positions(
                    center=(0, 0, 0),
                    radius=args.gif_radius,
                    height=args.gif_height,
                    num_frames=args.gif_frames
                )
                
                # Create frames for GIF
                frame_paths = []
                for i, camera_pos in enumerate(camera_positions):
                    frame_path = os.path.join(
                        args.output_directory,
                        f"{s.uid}_{args.scene_id}_gif_{i:03d}.png"
                    )
                    frame_paths.append(frame_path)
                
                # Render each frame with different camera position
                for i, camera_pos in enumerate(camera_positions):
                    print(f"Rendering frame {i + 1}/{len(camera_positions)} with camera position {camera_pos}")
                    
                    # Create a new scene for each frame
                    frame_scene = Scene(size=args.window_size)
                    if args.with_orthographic_projection:
                        frame_scene.camera_matrix = pyrr.Matrix44.orthogonal_projection(
                            left=-3.1, right=3.1, bottom=-3.1, top=3.1, near=0.1, far=1000
                        )
                    frame_scene.light = camera_pos
                    frame_scene.camera_position = camera_pos
                    frame_scene.camera_target = args.camera_target
                    frame_scene.up_vector = args.up_vector
                    frame_scene.background = args.background
                    
                    # Render the frame
                    behaviours = [SaveFrames(frame_paths[i], 1)]
                    render(
                        renderables,
                        size=args.window_size,
                        camera_position=camera_pos,
                        camera_target=args.camera_target,
                        up_vector=args.up_vector,
                        background=args.background,
                        behaviours=behaviours,
                        n_frames=1,
                        scene=frame_scene
                    )
                
                # Create GIF from frames
                gif_path = os.path.join(
                    args.output_directory,
                    f"{s.uid}_{args.scene_id}_animation.gif"
                )
                create_gif_from_frames(frame_paths, gif_path, args.gif_duration)
                
                # Clean up individual frame files
                for frame_path in frame_paths:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                        
            elif args.without_screen:
                path_to_image = "{}/{}_".format(args.output_directory, s.uid)
                behaviours += [SaveFrames(path_to_image+"{:03d}.png", 1)]
                render(
                    renderables,
                    size=args.window_size,
                    camera_position=args.camera_position,
                    camera_target=args.camera_target,
                    up_vector=args.up_vector,
                    background=args.background,
                    behaviours=behaviours,
                    n_frames=2,
                    scene=scene
                )
            else:
                show(
                    renderables,
                    behaviours=behaviours+[SnapshotOnKey()],
                    size=args.window_size,
                    camera_position=args.camera_position,
                    camera_target=args.camera_target,
                    light=args.camera_position,
                    up_vector=args.up_vector,
                    background=args.background,
                )
            # Create a trimesh scene and export it
            path_to_objs = os.path.join(
                args.output_directory,
                "train_{}".format(args.scene_id)
            )
            if not os.path.exists(path_to_objs):
                os.mkdir(path_to_objs)
            export_scene(path_to_objs, trimesh_meshes)


if __name__ == "__main__":
    main(sys.argv[1:])
