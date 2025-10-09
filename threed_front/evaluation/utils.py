import os
from PIL import Image
import numpy as np
from shapely.geometry import Polygon
import torch
from torchvision import models
from tqdm import tqdm

from threed_front.datasets.threed_front import CachedThreedFront, ThreedFront
from threed_front.datasets.threed_front_encoding_base import Scale
from threed_front.rendering import get_floor_plan, get_textured_objects, render_projection


class ImageFolderDataset(torch.utils.data.Dataset):
    """Dataset class for a directory of png images"""
    def __init__(self, directory, train=True):
        image_paths = sorted([os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith("png")])
        
        N = len(image_paths) // 2
        start = 0 if train else N
        self.image_paths = image_paths[start:start+N]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx]


class SyntheticVRealDataset(torch.utils.data.Dataset):
    """Dataset combining preprocessed real images and synthetic images"""
    def __init__(self, raw_dataset: CachedThreedFront, synthetic_image_dataset, 
                 real_render_name="rendered_scene_256.png"):
        self._N = len(raw_dataset)
        self.real = raw_dataset
        self.synthetic = synthetic_image_dataset
        self.real_render_name = real_render_name
    
    def __len__(self):
        return len(self.real) + len(self.synthetic)
    
    def __getitem__(self, idx):
        if idx < self._N:
            image_path = self.real[idx].image_path
            image_path = os.path.join(
                os.path.dirname(image_path), self.real_render_name
            )
            label = 1
        else:
            image_path = self.synthetic[idx - self._N]
            label = 0

        img = Image.open(image_path)
        img = np.asarray(img).astype(np.float32) / np.float32(255)
        img = np.transpose(img[:, :, :3], (2, 0, 1))

        return torch.from_numpy(img), torch.tensor([label], dtype=torch.float)


class AlexNet(torch.nn.Module):
    """Modified AlexNet to distinguish real and synthetic images"""
    def __init__(self):
        super().__init__()

        self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.fc = torch.nn.Linear(9216, 1)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = self.fc(x.view(len(x), -1))
        x = torch.sigmoid(x)

        return x


class AverageMeter:
    """Average computation class"""
    def __init__(self):
        self._value = 0
        self._cnt = 0

    def __iadd__(self, x):
        if torch.is_tensor(x):
            self._value += x.sum().item()
            self._cnt += x.numel()
        else:
            self._value += x
            self._cnt += 1
        return self

    @property
    def value(self):
        return self._value / self._cnt


def compute_loss_acc(model, dataloader, is_train=False, optimizer=None):
    if is_train:
        model.train()
    else:
        model.eval()
    device = next(model.parameters()).device
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad()
        
        y_hat = model(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        acc = (torch.abs(y-y_hat) < 0.5).float().mean()
        loss_meter += loss
        acc_meter += acc
        
        if is_train:
            loss.backward()
            optimizer.step()
            msg_pre = "{: 3d} loss: {:.4f} - acc: {:.4f}"
        else:
            msg_pre = "{: 3d} val_loss: {:.4f} - val_acc: {:.4f}"
        
        msg = msg_pre.format(i, loss_meter.value, acc_meter.value)
        print(msg + "\b"*len(msg), end="", flush=True)
    print()
    
    return loss_meter, acc_meter


def categorical_kl(p, q):
    return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()


def collect_cooccurrence(scenes, num_classes):
    # Count object class co-occurrences given a sequence of generated scenes
    cooccur_count = np.zeros((num_classes, num_classes))
    for scene in scenes:
        label_indices = scene["class_labels"].argmax(axis=-1)
        for ii in range(len(label_indices)):
            label_i = label_indices[ii]
            for jj in range(ii+1, len(label_indices)):
                label_j = label_indices[jj]
                cooccur_count[label_i, label_j] += 1
    cooccur_count += cooccur_count.T
    return cooccur_count


def evaluate_kl_divergence(raw_dataset:CachedThreedFront, scene_indices, layout_list):
    """Compute KL-divergence between ground-truth and synthesized object distributions"""
    # Collect synthesized and ground truth class labels
    n_object_types = raw_dataset.n_object_types
    gt_class_labels = []
    syn_class_labels = []
    for idx, bbox_params in zip(scene_indices, layout_list):
        gt_class_labels.append(
            {"class_labels": 
             raw_dataset.get_room_params(idx)["class_labels"][:, :n_object_types]})
        syn_class_labels.append({"class_labels": bbox_params["class_labels"]})

    # Compute frequencies of the class labels
    gt_total = sum(
        [raw_dataset.get_room_params(i)["class_labels"][:, :n_object_types].sum(0)
         for i in range(len(raw_dataset))]
    )
    gt_freq = gt_total / sum(
        [raw_dataset.get_room_params(i)["class_labels"].shape[0] 
         for i in range(len(raw_dataset))]
    )
    syn_total = sum([d["class_labels"].sum(0) for d in syn_class_labels])
    syn_freq = syn_total / sum([d["class_labels"].shape[0] for d in syn_class_labels])
    
    # Check freqeuncies sum to 1 (i.e. no label outside the 0:n_object_types range)
    print(f"[Ashok] sum of gt_freq: {gt_freq.sum()}, sum of syn_freq: {syn_freq.sum()}")
    assert 0.9999 <= gt_freq.sum() <= 1.0001
    assert 0.9999 <= syn_freq.sum() <= 1.0001

    return categorical_kl(gt_freq, syn_freq), gt_class_labels, syn_class_labels, \
        gt_total, syn_total


def render_projection_from_layout(
        room, bbox_params, objects_dataset, output_path, classes, scene_viz,
        floor_texture=None, floor_color=(0.87, 0.72, 0.53), retrieve_mode="size",
        color_palette=None
    ):

    print("SAUGAT: classes", classes)
    # object renderables
    renderables, _ = get_textured_objects(
        bbox_params, objects_dataset, classes, retrieve_mode=retrieve_mode, 
        color_palette=color_palette, with_trimesh=False
    )
    
    # floor plan if either floor texture or floor color is not None
    if not (floor_texture is None and floor_color is None):
        floor_plan, _, _ = get_floor_plan(room, floor_texture, floor_color)
        renderables.append(floor_plan)
    
    # render projection and save to output_path
    render_projection(scene_viz, renderables, color=None, 
                      mode="shading", frame_path=output_path)


def bbox_xz_corners(translations, sizes, angles, erosion=0.0):
    """return x-z plane projection of bounding boxes as a list of 4x2 numpy arrays"""
    xz_corners = []
    for i in range(sizes.shape[0]):
        size_x = max(sizes[i, 0] - erosion, 0)
        size_z = max(sizes[i, 2] - erosion, 0)
        vector_list = [
            np.array([-size_x, -size_z]), 
            np.array([-size_x,  size_z]),
            np.array([ size_x,  size_z]), 
            np.array([ size_x, -size_z])
        ]
        R = np.array(
            [[np.cos(angles[i, 0]), np.sin(angles[i, 0])],
             [-np.sin(angles[i, 0]), np.cos(angles[i, 0])]]
        )   # x-z components of R_y(theta)

        xz_corners.append(np.vstack(vector_list) @ R.T + translations[i, [0, 2]])
    return xz_corners


def count_out_of_boundary(floorplan_boundary, bboxes, area_tol=1e-5, erosion=0.05):
    """count number of bboxes out of floor boundary 
    (use erosion arg to account for underestimate in floor plan sizes)"""
    floor_polygon = Polygon(floorplan_boundary)
    xz_corners = bbox_xz_corners(
        bboxes["translations"], bboxes["sizes"], bboxes["angles"], 
        erosion=erosion
    )
    
    num_oob = 0
    oob_mask = np.zeros(len(xz_corners), dtype=bool)
    for i, xz_bbox in enumerate(xz_corners):
        object_polygon = Polygon(xz_bbox.tolist())
        oob_area = object_polygon.area - floor_polygon.intersection(object_polygon).area
        if oob_area > area_tol:
            num_oob += 1
            oob_mask[i] = True
    
    return num_oob, oob_mask


def compute_bbox_iou(bboxes):
    "return a list of iou for all pairs of bounding boxes"
    xz_corners = bbox_xz_corners(
        bboxes["translations"], bboxes["sizes"], bboxes["angles"], erosion=0
    )
    xz_polygons = [Polygon(xz_bbox.tolist()) for xz_bbox in xz_corners]
    bbox_volumes = [bbox[0]*bbox[1]*bbox[2]*8 for bbox in bboxes["sizes"].tolist()]

    bbox_iou = []
    for i in range(len(xz_polygons)):
        for j in range(i+1, len(xz_polygons)):
            # compare y-axis overlap
            vertical_overlap = \
                (bboxes["sizes"][i, 1] + bboxes["sizes"][j, 1]) \
                -  abs(bboxes["translations"][i, 1] - bboxes["translations"][j, 1])
            if vertical_overlap > 0:
                intersect_area = xz_polygons[i].intersection(xz_polygons[j]).area
                if intersect_area > 0:
                    intersect_bbox = intersect_area * vertical_overlap
                    bbox_iou.append(
                        intersect_bbox /
                         (bbox_volumes[i] + bbox_volumes[j] - intersect_bbox
                    ))
                else:
                    bbox_iou.append(0.0)
            else:
                bbox_iou.append(0.0)
    return bbox_iou


## this is from iou loss of diffuscenes
# def compute_bbox_iou(bboxes):
#     """Return a list of IoU values for all pairs of bounding boxes using 3D IoU calculation"""
#     import torch  # Adding import here to ensure torch is available
    
#     # Convert numpy arrays to torch tensors
#     translations = torch.tensor(bboxes["translations"])
#     sizes = torch.tensor(bboxes["sizes"])
    
#     # Create axis-aligned bounding box corners in format [x1, y1, z1, x2, y2, z2]
#     mins = translations - sizes  # Shape: [N, 3]
#     maxs = translations + sizes  # Shape: [N, 3]
    
#     # Concatenate to form [x1, y1, z1, x2, y2, z2]
#     axis_aligned_bbox_corners = torch.cat([mins, maxs], dim=-1)  # Shape: [N, 6]
    
#     # Add batch dimension for compatibility with axis_aligned_bbox_overlaps_3d
#     bboxes_tensor = axis_aligned_bbox_corners.unsqueeze(0)  # Shape: [1, N, 6]
    
#     # Calculate IoU directly using 3D IoU function
#     n_boxes = translations.shape[0]
#     bbox_iou_result = []
    
#     # If there are fewer than 2 boxes, return empty list
#     if n_boxes < 2:
#         return bbox_iou_result
    
#     # Calculate all pairwise IoUs at once
#     iou_matrix = axis_aligned_bbox_overlaps_3d(bboxes_tensor, bboxes_tensor, mode="iou")[0]  # Shape: [N, N]
    
#     # Extract upper triangle values (excluding diagonal)
#     for i in range(n_boxes):
#         for j in range(i+1, n_boxes):
#             bbox_iou_result.append(iou_matrix[i, j].item())
    
#     return bbox_iou_result

def axis_aligned_bbox_overlaps_3d(
    bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6
):
    """
    https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/iou_calculators/iou3d_calculator.py
    """
    """Calculate overlap between two set of axis aligned 3D bboxes. If
        ``is_aligned`` is ``False``, then calculate the overlaps between each bbox
        of bboxes1 and bboxes2, otherwise the overlaps between each aligned pair of
        bboxes1 and bboxes2.
        Args:
            bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>
                format or empty.
            bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>
                format or empty.
                B indicates the batch dim, in shape (B1, B2, ..., Bn).
                If ``is_aligned`` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union) or "giou" (generalized
                intersection over union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Defaults to False.
            eps (float, optional): A value added to the denominator for numerical
                stability. Defaults to 1e-6.
        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """

    assert mode in ["iou", "giou"], f"Unsupported mode {mode}"
    # Either the boxes are empty or the length of boxes's last dimension is 6
    assert bboxes1.size(-1) == 6 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 6 or bboxes2.size(0) == 0

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (
        (bboxes1[..., 3] - bboxes1[..., 0])
        * (bboxes1[..., 4] - bboxes1[..., 1])
        * (bboxes1[..., 5] - bboxes1[..., 2])
    )
    area2 = (
        (bboxes2[..., 3] - bboxes2[..., 0])
        * (bboxes2[..., 4] - bboxes2[..., 1])
        * (bboxes2[..., 5] - bboxes2[..., 2])
    )

    if is_aligned:
        lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])  # [B, rows, 3]
        rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])  # [B, rows, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.max(
            bboxes1[..., :, None, :3], bboxes2[..., None, :, :3]
        )  # [B, rows, cols, 3]
        rb = torch.min(
            bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:]
        )  # [B, rows, cols, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 3]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        if mode == "giou":
            enclosed_lt = torch.min(
                bboxes1[..., :, None, :3], bboxes2[..., None, :, :3]
            )
            enclosed_rb = torch.max(
                bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:]
            )

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ["iou"]:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious
