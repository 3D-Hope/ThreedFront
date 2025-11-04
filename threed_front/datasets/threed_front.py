# 
# modified from: 
#   https://github.com/nv-tlabs/ATISS.
#   https://github.com/tangjiapeng/DiffuScene
#

from collections import Counter, OrderedDict
from functools import lru_cache
import numpy as np
import json
import os
from copy import deepcopy

from PIL import Image

from .common import BaseDataset
from .threed_front_scene import Room
from .parse_utils import parse_threed_front_scenes
import math

class ThreedFront(BaseDataset):
    """Container for the scenes in the 3D-FRONT dataset.

        Arguments
        ---------
        scenes: list of Room objects for all scenes in 3D-FRONT dataset
    """
    def __init__(self, scenes, bounds=None):
        super().__init__(scenes)
        assert isinstance(self.scenes[0], Room)
        self._object_types = None
        self._room_types = None
        self._count_furniture = None
        self._bbox = None

        self._sizes = self._centroids = self._angles = None
        self._objfeats = self._objfeats_32 = None
        if bounds is not None:
            self._sizes = bounds["sizes"]
            self._centroids = bounds["translations"]
            self._angles = bounds["angles"]
            self._objfeats = bounds.get(
                "objfeats", (np.array([1]), np.array([-1]), np.array([1])) # std, min, max
            )
            self._objfeats_32 = bounds.get(
                "objfeats_32", (np.array([1]), np.array([-1]), np.array([1])) # std, min, max
            )
        
        self._max_length = None

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
                len(self.scenes), self.n_object_types
        )

    @property
    def bbox(self):
        """The bbox for the entire dataset is simply computed based on the
        bounding boxes of all scenes in the dataset.
        """
        if self._bbox is None:
            _bbox_min = np.array([1000, 1000, 1000])
            _bbox_max = np.array([-1000, -1000, -1000])
            for s in self.scenes:
                bbox_min, bbox_max = s.bbox
                _bbox_min = np.minimum(bbox_min, _bbox_min)
                _bbox_max = np.maximum(bbox_max, _bbox_max)
            self._bbox = (_bbox_min, _bbox_max)
        return self._bbox

    def _centroid(self, box, offset):
        return box.centroid(offset)

    def _size(self, box):
        return box.size

    def _compute_bounds(self):
        all_centroids = \
            [self._centroid(f, -s.centroid) for s in self.scenes for f in s.bboxes]
        self._centroids = \
            np.min(all_centroids, axis=0), np.max(all_centroids, axis=0)
        
        all_sizes = [self._size(f) for s in self.scenes for f in s.bboxes]
        self._sizes = \
            np.min(all_sizes, axis=0), np.max(all_sizes, axis=0)
        
        all_angles = [f.z_angle for s in self.scenes for f in s.bboxes]
        self._angles = \
            np.min(all_angles), np.max(all_angles)
        
        all_pc_lat = [
            f.raw_model_norm_pc_lat() for s in self.scenes for f in s.bboxes
        ]
        if any(lat is None for lat in all_pc_lat):
            self._objfeats = None
        else:
            all_objfeats = np.concatenate(all_pc_lat, axis=0)
            self._objfeats = \
                np.std(all_objfeats.flatten()), np.min(all_objfeats), np.max(all_objfeats)
            
        all_pc_lat32 = [
            f.raw_model_norm_pc_lat32() for s in self.scenes for f in s.bboxes
        ]
        if any(lat is None for lat in all_pc_lat32):
            self._objfeats_32 = None
        else:
            all_objfeats_32 = np.concatenate(all_pc_lat32, axis=0)
            self._objfeats_32 = \
                np.std(all_objfeats_32.flatten()), np.min(all_objfeats_32), np.max(all_objfeats_32)
        
    @property
    def bounds(self):
        return {
            "translations": self.centroids,
            "sizes": self.sizes,
            "angles": self.angles,
            "objfeats": self.objfeats,
            "objfeats_32": self.objfeats_32,
        }

    @property
    def sizes(self):
        if self._sizes is None:
            self._compute_bounds()
        return self._sizes

    @property
    def centroids(self):
        if self._centroids is None:
            self._compute_bounds()
        return self._centroids

    @property
    def angles(self):
        if self._angles is None:
            self._compute_bounds()
        return self._angles
    
    @property
    def objfeats(self):
        if self._objfeats is None:
            self._compute_bounds()
        return self._objfeats

    @property
    def objfeats_32(self):
        if self._objfeats_32 is None:
            self._compute_bounds()
        return self._objfeats_32

    @property
    def count_furniture(self):
        if self._count_furniture is None:
            counts = []
            for s in self.scenes:
                counts.append(s.furniture_in_room)
            counts = Counter(sum(counts, []))
            counts = OrderedDict(sorted(counts.items(), key=lambda x: -x[1]))
            self._count_furniture = counts
        return self._count_furniture

    @property
    def class_order(self):
        return dict(zip(
            self.count_furniture.keys(),
            range(len(self.count_furniture))
        ))

    @property
    def class_frequencies(self):
        object_counts = self.count_furniture
        class_freq = {}
        n_objects_in_dataset = sum(
            [object_counts[k] for k, v in object_counts.items()]
        )
        for k, v in object_counts.items():
            class_freq[k] = object_counts[k] / n_objects_in_dataset
        return class_freq

    @property
    def object_types(self):
        if self._object_types is None:
            self._object_types = set()
            for s in self.scenes:
                self._object_types |= set(s.object_types)
            self._object_types = sorted(self._object_types)
        return self._object_types

    @property
    def room_types(self):
        if self._room_types is None:
            self._room_types = set([s.scene_type for s in self.scenes])
        return self._room_types

    @property
    def class_labels(self):
        # [end] is the stop token for autogressive models and empty token for diffusion models
        return self.object_types + ["end"]

    # compute max_lenght for diffusion models
    @property
    def max_length(self):
        if self._max_length is None:
            _room_types = set([str(s.scene_type) for s in self.scenes])
            if 'bed' in _room_types:
                self._max_length = 12
            elif 'living' in _room_types:
                self._max_length = 21
            elif 'dining' in _room_types:
                self._max_length = 21
            elif 'library' in _room_types:
                self._max_length = 11

        return self._max_length

    @classmethod
    def from_dataset_directory(cls, dataset_directory, path_to_model_info,
                               path_to_models, path_to_room_masks_dir=None,
                               path_to_bounds=None, filter_fn=lambda s: s):
        scenes = parse_threed_front_scenes(dataset_directory, path_to_model_info,
            path_to_models, path_to_room_masks_dir)
        bounds = None
        if path_to_bounds:
            bounds = np.load(path_to_bounds, allow_pickle=True)

        return cls([s for s in map(filter_fn, scenes) if s], bounds)


class CachedRoom(object):
    """Dataset class to combine room data after preprocessing."""
    def __init__(self, scene_id, room_layout, floor_plan_vertices, floor_plan_faces,
        floor_plan_centroid, class_labels, translations, sizes, angles, objfeats,
        objfeats_32, image_path, edge_index=None, num_cuboids_list=None):
        self.scene_id = scene_id
        self.room_layout = room_layout
        self.floor_plan_faces = floor_plan_faces
        self.floor_plan_vertices = floor_plan_vertices
        self.floor_plan_centroid = floor_plan_centroid
        self.class_labels = class_labels
        self.translations = translations
        self.sizes = sizes
        self.angles = angles
        self.objfeats = objfeats
        self.objfeats_32 = objfeats_32
        self.image_path = image_path
        self.edge_index = edge_index
        self.num_cuboids_list = num_cuboids_list

    @property
    def floor_plan(self):
        return np.copy(self.floor_plan_vertices), np.copy(self.floor_plan_faces)

    @property
    def room_mask(self):
        return self.room_layout[:, :, None]


class CachedThreedFront(ThreedFront):
    """Container for data paths after preprocessing."""
    def __init__(self, base_dir, config, scene_ids, parse_floor_plan=True,
                 rendered_name="rendered_scene_256.png", include_edges=False):
        self._base_dir = os.path.abspath(base_dir)
        self.config = config

        # Load training set data stats
        self._parse_train_stats(config["train_stats"])

        # Find subdirectory names where the scene id is in the specified list
        self._tags = sorted([tag for tag in os.listdir(self._base_dir)
            if tag.split("_")[1] in scene_ids])
        
        # Paths to orthographic projection images
        if os.path.isfile(os.path.join(self._base_dir, self._tags[0], rendered_name)):
            self._rendered_name = rendered_name
        else:
            self._rendered_name = next(
                (f for f in os.listdir(os.path.join(self._base_dir, self._tags[0])) 
                 if f.startswith("rendered_") and f.endswith(".png")), 
                "None.png"
            )
        
        # Paths to edge indices (False if include_edges=False or file does not extist)
        self._contain_edges = \
            include_edges and os.path.isfile(self._path_to_room(0), "edges.npz")
        
        # Parse dataset
        self._dataset_dict = self._parse_dataset_params(parse_floor_plan)
        self._max_length = \
                max(d["class_labels"].shape[0] for d in self._dataset_dict)
    
    def _path_to_room(self, i):
        # print(os.path.join(self._base_dir, self._tags[i]))
        # import sys; sys.exit()
        if os.path.exists(os.path.join(self._base_dir, self._tags[i], "boxes.npz")):
            path = os.path.join(self._base_dir, self._tags[i], "boxes.npz")
        else:
            path = os.path.join(self._base_dir, self._tags[i], "cuboid_scene.npz")
        return path
    
    def _path_to_render(self, i):
        return os.path.join(self._base_dir, self._tags[i], self._rendered_name)

    def _path_to_edge(self, i):
        return os.path.join(self._base_dir, self._tags[i], "edges.npz") \
            if self.contain_edges else None
    
    def _parse_dataset_params(self, parse_floor_plan=True):
        dataset_dict = []
        for i in range(len(self._tags)):
            data_dict = self._parse_room_params(i, parse_floor_plan)
            dataset_dict.append(data_dict)
        return dataset_dict
    
    
    
    def calc_box_from_polygon(self, points, S):
        l = len(points)
        boxes = {"translation":[],"size":[]}
        for i in range(l):
            # i = 6
            # if i==6:
            #     continue
            x = points[i]
            j = (i+1)%l
            k = (i+2)%l
            vij = np.array(points[j]-points[i])
            vjk = np.array(points[k]-points[j])
            # x->y, x->z
            vxz = np.array([-vij[1],vij[0]])
            # z = x + vxz/math.sqrt(vxz[0]*vxz[0]+vxz[1]*vxz[1])*S
            

            #####calc y
            C = x
            D = x + vij/math.sqrt(vij[0]*vij[0]+vij[1]*vij[1])*S  #y
            line2 = D-C
            intersection_lst = []
            for t in range(1,l-1):
                A = points[(i+t)%l]
                B = points[(i+t+1)%l]
                line1 = B-A
                if Room.is_perpendicular(line1,line2) and np.cross(line2,line1)>0: #chuizhi & left 
                    intersection = Room.segments_intersect(A,B,C,D) #qiu xianduan jiaodian
                    if intersection:
                        intersection_lst.append(np.array([intersection[0],intersection[1]]))

            #find closest intersction
            y = D.copy()
            if len(intersection_lst)>0:
                min_dist = S*S
                for intersection in intersection_lst:
                    d = Room.dist2(intersection,x)
                    if d<min_dist:
                        y = intersection.copy()
                        min_dist = d
            # cross = np.cross(vij,vjk)
            # if cross > 0:  #-| limit
            #     y = points[j]
            # else: # -- or -|
            #     y = x + vij/math.sqrt(vij[0]*vij[0]+vij[1]*vij[1])*S

            #####calc z
            C = x
            Dxz = x + vxz/math.sqrt(vxz[0]*vxz[0]+vxz[1]*vxz[1])*S
            Dyz = y + vxz/math.sqrt(vxz[0]*vxz[0]+vxz[1]*vxz[1])*S

            line2 = Dxz-C
            intersection_lst = []
            dmin2 = S*S
            for t in range(1,l-1):
                A = points[(i+t)%l]
                B = points[(i+t+1)%l]
                line1 = B-A
                if Room.is_perpendicular(line1,line2) and np.cross(line2,line1)>0: #chuizhi & left
                    #xz
                    intersection = Room.segments_intersect(A,B,x,Dxz) #qiu xianduan jiaodian
                    if intersection:
                        dmin2 = min(dmin2,Room.dist2(intersection,x))
                    #yz
                    intersection = Room.segments_intersect(A,B,y,Dyz) #qiu xianduan jiaodian
                    if intersection:
                        dmin2 = min(dmin2,Room.dist2(intersection,y))
                        

            #find closest intersction
            d = np.sqrt(dmin2)
            z = x + vxz/math.sqrt(vxz[0]*vxz[0]+vxz[1]*vxz[1])*d
    
            # cross = np.cross(vij,vjk)
            # if cross > 0:  #-| limit
            #     y = points[j]
            # else: # -- or -|
            #     y = x + vij/math.sqrt(vij[0]*vij[0]+vij[1]*vij[1])*S

            sizez = S
            meanz = 0
            if x[0]==y[0]:
                sizey = abs(x[1]-y[1])
                sizex = abs(x[0]-z[0])
                meany = (x[1]+y[1])/2
                meanx = (x[0]+z[0])/2
            else:
                sizex = abs(x[0]-y[0])
                sizey = abs(x[1]-z[1])
                meany = (x[1]+z[1])/2
                meanx = (x[0]+y[0])/2

            
            translation = np.array([meanx,meany,meanz])
            size = np.array([sizex,sizey,sizez])# half size
            boxes["translation"].append(translation)
            boxes["size"].append(size)

        boxes["translation"] = np.array(boxes["translation"])
        boxes["size"] = np.array(boxes["size"])
        # 
        gt_boxes = [[boxes["translation"][i][0],boxes["translation"][i][1],boxes["translation"][i][2],boxes["size"][i][0],boxes["size"][i][1],boxes["size"][i][2],0] for i in range(len(boxes["translation"]))]
        gt_boxes = np.array(gt_boxes)
        # print(gt_boxes)
        # vis = draw_box_label(gt_boxes, (0, 0, 1))

        return  boxes
        
        
    
    def room_outer_box_from_scene(self, box, centroid):
        max_len = 50
        try:
            L = box["translation"].shape[0]
            # print(L)
            
            # sequence length
            assert(max_len>=L)
            translations = np.zeros((max_len, 3), dtype=np.float32)
            centroid = centroid.copy()
            centroid[2] = centroid[1] #xzy->xyz
            centroid[1] = centroid[2]
            center = np.repeat(centroid[None,:],L,axis=0)
            translations[:L] = box["translation"] - center
            sizes = np.zeros((max_len, 3), dtype=np.float32)
            sizes[:L] = box["size"]
            angles = np.zeros((max_len, 1), dtype=np.float32)
            bbox_outer = np.concatenate([translations,sizes,angles],axis=-1)

            # gt_boxes = [[translations[i][0],translations[i][1],translations[i][2],sizes[i][0],sizes[i][1],sizes[i][2],0] for i in range(L)]
            # gt_boxes = np.array(gt_boxes)
            # from utils.open3d_vis_utils import draw_box_label
            # vis = draw_box_label(gt_boxes, (0, 0, 1))

            # gt_boxes = [[translations[i][0],translations[i][1],translations[i][2],sizes[i][0],sizes[i][1],sizes[i][2],0] for i in range(max_len)]
            # gt_boxes = np.array(gt_boxes)
            # from utils.open3d_vis_utils import draw_box_label
            # vis = draw_box_label(gt_boxes, (0, 0, 1))
            # print("TranslationEncoder",a2-a1,a3-a2,a4-a3)
            # return {"translations":translations,
            #         "sizes":sizes,
            #         "angles":angles}
            return bbox_outer
        
        except:
            translations = np.zeros((max_len, 3), dtype=np.float32)
            sizes = np.zeros((max_len, 3), dtype=np.float32)
            angles = np.zeros((max_len, 1), dtype=np.float32)
            bbox_outer = np.concatenate([translations,sizes,angles],axis=-1)
            return bbox_outer

    def _parse_room_params(self, i, parse_floor_plan=True):
        D = np.load(self._path_to_room(i))
        # Floor plan-derived features (robust to missing keys)
        try:
            points = D["floor_plan_ordered_corners"]
            boxes = self.calc_box_from_polygon(points, S=50)  # same as physcene
            room_outer_box = self.room_outer_box_from_scene(boxes, D["floor_plan_centroid"])
        except Exception:
            points = None
            boxes = None
            room_outer_box = None
        # object features
        output_dict = {
            "class_labels": D["class_labels"],
            "translations": D["translations"],
            "sizes": D["sizes"],
            "angles": D["angles"],
        }
        # Conditionally add floor plan fields if present and computed
        if points is not None:
            output_dict["floor_polygon_points"] = D["floor_plan_ordered_corners"]
        if room_outer_box is not None:
            output_dict["room_outer_box"] = room_outer_box
        if "floor_plan_centroid" in D.keys():
            output_dict["floor_plan_centroid"] = D["floor_plan_centroid"]
        if "floor_plan_vertices" in D.keys():
            output_dict["floor_plan_vertices"] = D["floor_plan_vertices"]
        if "floor_plan_faces" in D.keys():
            output_dict["floor_plan_faces"] = D["floor_plan_faces"]
        if "objfeats" in D.keys():
            output_dict[ "objfeats" ] = D["objfeats"]
        if "objfeats_32" in D.keys():
            output_dict[ "objfeats_32" ] = D["objfeats_32"]
        if "floor_plan_boundary_points_normals" in D.keys():
            output_dict[ "fpbpn" ] = D["floor_plan_boundary_points_normals"]
            
        
        # room layout
        if parse_floor_plan:
            room_rgb_2d = self.config.get('room_rgb_2d', False)
            if room_rgb_2d:
                room = self._get_room_rgb_2d(self._path_to_render(i))
                room = np.transpose(room[:, :, 0:3],  (2, 0, 1))
            else:
                room = self._get_room_layout(D["room_layout"])
                room = np.transpose(room[:, :, None], (2, 0, 1))
            output_dict["room_layout"] = room
        
        if self.contain_edges:
            E = np.load(self._path_to_edge(i))
            try:
                output_dict["edge_index"] = \
                    np.hstack([v for k, v in E.items() if v.size>0 and "_edges" in k])
            except ValueError:
                output_dict["edge_index"] = np.empty((2, 0), dtype=E["on_edges"].dtype)
            output_dict["adj_matrix"] = E["adj_matrix"]
        return output_dict

    def _get_room_layout(self, room_layout):
        # Resize the room_layout if needed
        img = Image.fromarray(room_layout[:, :, 0])
        img = img.resize(
            tuple(map(int, self.config["room_layout_size"].split(","))),
            resample=Image.BILINEAR
        )
        D = np.asarray(img).astype(np.float32) / np.float32(255)
        return D

    def _get_room_rgb_2d(self, img_path):
        # Resize the room_layout if needed
        img = Image.open(img_path)
        img = img.resize(
            tuple(map(int, self.config["room_layout_size"].split(","))),
            resample=Image.BILINEAR
        )
        D = np.asarray(img).astype(np.float32) / np.float32(255)
        return D

    @lru_cache(maxsize=32)
    def __getitem__(self, i):
        D = np.load(self._path_to_room(i))
        # print(f"[Ashok NEW] D: {list(D.keys())}")
        # import sys; sys.exit();

        if self.contain_edges:
            E = np.load(self._path_to_edge(i))
            try:
                edge_index = \
                    np.hstack([v for k, v in E.items() if v.size>0 and "_edges" in k])
            except ValueError:
                edge_index = np.empty((2, 0), dtype=E["on_edges"].dtype)
        else:
            edge_index = None
        
        return CachedRoom(
            scene_id=D["scene_id"],
            room_layout=self._get_room_layout(D["room_layout"]) \
                if "room_layout" in D.keys() else None,
            floor_plan_vertices=D["floor_plan_vertices"],
            floor_plan_faces=D["floor_plan_faces"],
            floor_plan_centroid=D["floor_plan_centroid"],
            class_labels=D["class_labels"],
            translations=D["translations"],
            sizes=D["sizes"],
            angles=D["angles"],
            objfeats=D["objfeats"] if "objfeats" in D.keys() else None,
            objfeats_32=D["objfeats_32"] if "objfeats_32" in D.keys() else None,
            image_path=self._path_to_render(i),
            edge_index=edge_index
        )

    def get_room_params(self, i):
        return deepcopy(self._dataset_dict[i])

    def __len__(self):
        return len(self._tags)

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
                len(self), self.n_object_types
        )

    def _parse_train_stats(self, train_stats):
        with open(os.path.join(self._base_dir, train_stats), "r") as f:
            train_stats = json.load(f)
        self._centroids = train_stats["bounds_translations"]
        self._centroids = (
            np.array(self._centroids[:3], dtype=np.float32), 
            np.array(self._centroids[3:], dtype=np.float32)
        )
        self._sizes = train_stats["bounds_sizes"]
        self._sizes = (
            np.array(self._sizes[:3], dtype=np.float32), 
            np.array(self._sizes[3:], dtype=np.float32)
        )
        self._angles = train_stats["bounds_angles"]
        self._angles = (
            np.array(self._angles[0], dtype=np.float32), 
            np.array(self._angles[1], dtype=np.float32)
        )
        if "bounds_objfeats" in train_stats.keys():
            self._objfeats = train_stats["bounds_objfeats"]
            self._objfeats = (
                np.array([self._objfeats[0]], dtype=np.float32), 
                np.array([self._objfeats[1]], dtype=np.float32), 
                np.array([self._objfeats[2]], dtype=np.float32)
            )
        else:
            self._objfeats = (
                np.array([1], dtype=np.float32), 
                np.array([-1], dtype=np.float32), 
                np.array([1], dtype=np.float32)
            )

        if "bounds_objfeats_32" in train_stats.keys():
            self._objfeats_32 = train_stats["bounds_objfeats_32"]
            self._objfeats_32 = (
                np.array([self._objfeats_32[0]], dtype=np.float32),
                np.array([self._objfeats_32[1]], dtype=np.float32),
                np.array([self._objfeats_32[2]], dtype=np.float32)
            )
        else:
            self._objfeats_32 = (
                np.array([1], dtype=np.float32), 
                np.array([-1], dtype=np.float32), 
                np.array([1], dtype=np.float32)
            )
        
        if "bounds_fpbpn" in train_stats.keys():
            self._fpbpn = train_stats["bounds_fpbpn"]
            self._fpbpn = (
                np.array(self._fpbpn[:4], dtype=np.float32), 
                np.array(self._fpbpn[4:], dtype=np.float32)
            )
        else:
            self._fpbpn = (-np.ones(4), np.ones(4))
        
        self._class_labels = train_stats["class_labels"]
        self._object_types = train_stats["object_types"]
        self._class_frequencies = train_stats["class_frequencies"]
        self._count_furniture = train_stats["count_furniture"]
        try:
            self._class_order = train_stats["class_order"]
        except:
            self._class_order = dict(zip(self._count_furniture.keys(), range(len(self._count_furniture))))

    @property
    def contain_edges(self):
        return self._contain_edges

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def object_types(self):
        return self._object_types

    @property
    def class_frequencies(self):
        return self._class_frequencies

    @property
    def class_order(self):
        return self._class_order

    @property
    def count_furniture(self):
        return self._count_furniture
    
    @property
    def max_length(self):
        return self._max_length
    
    @property
    def bounds(self):
        return {
            "translations": self._centroids,
            "sizes": self._sizes,
            "angles": self._angles,
            "objfeats": self._objfeats,
            "objfeats_32": self._objfeats_32,
            "fpbpn": self._fpbpn,
        }
