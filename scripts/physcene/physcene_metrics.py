import cv2
import numpy as np
from scipy.ndimage import binary_dilation

"""Script for calculating walkable metric. """

def map_to_image_coordinate(point, scale, image_size):
        x, y = point
        x_image = int(x / scale * image_size/2)+image_size/2
        y_image = int(y / scale * image_size/2)+image_size/2
        return x_image, y_image
    
def calc_bbox_masks(bbox,class_labels,image,image_size,scale,robot_width,floor_plan_mask, box_wall_count): 
    """
    type: ["bbox","front_line", "front_center"]
    """
    box_masks = []
    handle_points = []
    for box,class_label in zip(bbox,class_labels):
        box_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        box_wall_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        center = map_to_image_coordinate(box[:3][0::2], scale, image_size)
        size = (int(box[3] / scale * image_size / 2) * 2,
                int(box[5] / scale * image_size / 2) * 2)
        angle = box[-1]
        w, h = size
        open_size = 0
        rot_center = center
        handle = np.array([0, h/2+open_size+robot_width/2+1])
        bbox = np.array([[-w/2, -h/2],
                        [-w/2, h/2+open_size],
                        [w/2, h/2+open_size],
                        [w/2, -h/2],
                        ])
        
        R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

        box_points = bbox.dot(R) + rot_center
        handle_point = handle.dot(R) + rot_center
        handle_points.append(handle_point)
        box_points = np.intp(box_points)
        # box wall collision
        cv2.fillPoly(box_wall_mask, [box_points], (0, 255, 0))
        # cv2.imwrite("debug1.png", box_wall_mask)
        box_wall_mask = box_wall_mask[:,:,1]==255

        if (box_wall_mask*(255-floor_plan_mask)).sum()>0:
            box_wall_count+=1
        # cv2.imwrite("debug2.png", floor_plan_mask)

        #image connected region
        cv2.drawContours(image, [box_points], 0,
                        (0, 255, 0), robot_width)  # Green (BGR)
        cv2.fillPoly(image, [box_points], (0, 255, 0))
        
        # per box mask
        cv2.drawContours(box_mask, [box_points], 0,
                        (0, 255, 0), robot_width)  # Green (BGR)
        cv2.fillPoly(box_mask, [box_points], (0, 255, 0))
        st_element = np.ones((3, 3), dtype=bool)
        box_mask = binary_dilation((box_mask[:, :, 1].copy()==255).astype(image.dtype), st_element)
        box_masks.append(box_mask)
    return box_masks, handle_points, box_wall_count, image

def cal_walkable_metric(floor_plan, floor_plan_centroid, bboxes, robot_width=0.01, visual_path=None, calc_object_area=False):

    vertices, faces = floor_plan
    vertices = vertices - floor_plan_centroid
    vertices = vertices[:, 0::2]
    scale = np.abs(vertices).max()+0.2
    bboxes = bboxes[bboxes[:, 1] < 1.5]

    image_size = 256
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    robot_width = int(robot_width / scale * image_size/2)


    # draw face
    for face in faces:
        face_vertices = vertices[face]
        face_vertices_image = [
            map_to_image_coordinate(v, scale, image_size) for v in face_vertices]

        pts = np.array(face_vertices_image, np.int32)
        pts = pts.reshape(-1, 1, 2)
        color = (255, 0, 0)  # Blue (BGR)
        cv2.fillPoly(image, [pts], color)

    kernel = np.ones((robot_width, robot_width))
    image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
    # draw bboxes
    # cv2.imwrite("image.png", image)
    for box in bboxes:
        center = map_to_image_coordinate(box[:3][0::2], scale, image_size)
        size = (int(box[3] / scale * image_size / 2) * 2,
                int(box[5] / scale * image_size / 2) * 2)
        angle = box[-1]

        # calculate box vertices
        box_points = cv2.boxPoints(
            ((center[0], center[1]), size, -angle/np.pi*180))
        box_points = np.intp(box_points)

        cv2.drawContours(image, [box_points], 0,
                         (0, 255, 0), robot_width)  # Green (BGR)
        cv2.fillPoly(image, [box_points], (0, 255, 0))

    # cv2.imwrite("image.png", image)

    if calc_object_area:
        green_cnt = 0
        blue_cnt = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if list(image[i][j]) == [0, 255, 0]:
                    green_cnt += 1
                elif list(image[i][j]) == [255, 0, 0]:
                    blue_cnt += 1
        object_area_ratio = green_cnt/(blue_cnt+green_cnt)
        
    walkable_map = image[:, :, 0].copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        walkable_map, connectivity=8)


    walkable_map_max = np.zeros_like(walkable_map)
    for label in range(1, num_labels):
        mask = np.zeros_like(walkable_map)
        mask[labels == label] = 255

        if mask.sum() > walkable_map_max.sum():
            # room connected component with door
            walkable_map_max = mask.copy()

        # print("walkable_rate:", walkable_map_max.sum()/walkable_map.sum())
        if calc_object_area:
            return walkable_map_max.sum()/walkable_map.sum(), object_area_ratio
        else:
            return walkable_map_max.sum()/walkable_map.sum()
    if calc_object_area:    
        return 0.,object_area_ratio
    else:
        return 0.
    
def calc_wall_overlap(synthesized_scenes, floor_plan_lst, floor_plan_centroid_list, cfg, robot_real_width=0.3,calc_object_area=False,classes=None):
    
    box_wall_count = 0
    accessable_count = 0
    box_count = 0
    walkable_metric_list = []
    accessable_rate_list = []
    # from tqdm import tqdm

    # for i in tqdm(range(len(synthesized_scenes))):
    for i in range(len(synthesized_scenes)):
        d = synthesized_scenes[i]
        floor_plan = floor_plan_lst[i]
        floor_plan_centroid = floor_plan_centroid_list[i]
        valid_idx = d["objectness"][:,0]<0 #False
        class_labels = d["class_labels"][valid_idx]
        bbox = np.concatenate([
                    # d["class_labels"],
                    d["translations"][valid_idx],
                    d["sizes"][valid_idx],
                    d["angles"][valid_idx],
                    # d["objfeats_32"]
                ],axis=-1)

            
        vertices, faces = floor_plan
        vertices = vertices - floor_plan_centroid
        vertices = vertices[:, 0::2]
        # vertices = vertices[:, :2]
        scale = np.abs(vertices).max()+0.2

        image_size = 256
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        robot_width = int(robot_real_width / scale * image_size/2)

        
        
        # draw face
        for face in faces:
            face_vertices = vertices[face]
            face_vertices_image = [
                map_to_image_coordinate(v,scale, image_size) for v in face_vertices]

            pts = np.array(face_vertices_image, np.int32)
            pts = pts.reshape(-1, 1, 2)
            color = (255, 0, 0)  # Blue (BGR)
            cv2.fillPoly(image, [pts], color)
        
        floor_plan_mask = (image[:,:,0]==255)*255
        # cv2.imwrite("debug_floor.png", floor_plan_mask)
        # 缩小墙边界，机器人行动范围
        kernel = np.ones((robot_width, robot_width))
        image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
        
        box_masks, handle_points, box_wall_count, image = calc_bbox_masks(bbox,class_labels,image,image_size,scale,robot_width,floor_plan_mask, box_wall_count,classes=classes)
        # cv2.imwrite("debug.png", image)
        # breakpoint()
        walkable_map = image[:, :, 0].copy()
        # cv2.imwrite("debug.png", walkable_map)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            walkable_map, connectivity=8)
        # 遍历每个连通域

        accessable_rate = 0
        for label in range(1, num_labels):
            mask = np.zeros_like(walkable_map)
            mask[labels == label] = 1
            accessable_count = 0
            for box_mask in box_masks:
                if (box_mask*mask).sum()>0:
                    accessable_count += 1
            accessable_rate += accessable_count/len(box_masks)*mask.sum()/(labels!=0).sum()
        accessable_rate_list.append(accessable_rate)
        box_count += len(box_masks)

        #walkable map area rate
        if calc_object_area:
            walkable_rate, object_area_ratio = cal_walkable_metric(floor_plan, floor_plan_centroid, bbox, doors=None, robot_width=0.3, visual_path=None,calc_object_area=True)
        else:
            walkable_rate = cal_walkable_metric(floor_plan, floor_plan_centroid, bbox, doors=None, robot_width=0.3, visual_path=None,)
        walkable_metric_list.append(walkable_rate)

    walkable_average_rate = sum(walkable_metric_list)/len(walkable_metric_list)
    accessable_rate = sum(accessable_rate_list)/len(accessable_rate_list)
    # accessable_handle_rate = sum(accessable_handle_rate_list)/len(accessable_handle_rate_list)
    box_wall_rate = box_wall_count/box_count
    
    print('walkable_average_rate:', walkable_average_rate)
    print('accessable_rate:', accessable_rate)
    # print('accessable_handle_rate:', accessable_handle_rate)
    print('box_wall_rate:', box_wall_rate)
    if calc_object_area:
        print('object_area_ratio:', object_area_ratio)
        return walkable_average_rate, accessable_rate, box_wall_rate, object_area_ratio
    else:
        return walkable_average_rate, accessable_rate, box_wall_rate
    

if __name__ == "__main__":
    vertices = np.array([[-4.6405,  0., -3.4067],
                         [-5.8161,  0.,  0.0457],
                         [-5.8161,  0., -3.4067],
                         [-5.8161,  0.,  0.0457],
                         [-4.6405,  0., -3.4067],
                         [-5.8161,  0.,  2.149],
                         [-5.8161,  0.,  2.149],
                         [-4.6405,  0., -3.4067],
                         [-4.806,  0.,  0.9523],
                         [-4.806,  0.,  0.9523],
                         [-4.6405,  0., -3.4067],
                         [-1.6766,  0.,  0.9523],
                         [-1.6766,  0.,  0.9523],
                         [-4.6405,  0., -3.4067],
                         [-1.6766,  0., -3.4067],
                         [-4.806,  0.,  2.149],
                         [-5.8161,  0.,  2.149],
                         [-4.806,  0.,  0.9523]])
    faces = np.array([[0,  2,  1],
                      [3,  5,  4],
                      [6,  8,  7],
                      [9, 11, 10],
                      [12, 14, 13],
                      [15, 17, 16]])
    floor_plan_centroid = np.array([-3.74635,  0., -0.62885])

    bboxes = np.array([[-0.4438298,  2.48225976, -1.43605851,  0.363684,  0.11344881,
                        0.36719965, -3.12878394],
                       [1.43159928,  0.47924918,  0.20020839,  0.26718597,  0.48059947,
                        0.29016826, -1.57097435],
                       [1.81570315,  1.24139991,  0.52753237,  1.61088674,  1.23749601,
                        0.18798582, -1.58269131],
                       [-0.98245618,  1.52104479,  0.52209809,  0.09574268,  0.45265616,
                        0.09491291, -3.14084959],
                       [1.46061269,  0.47784704, -0.42971056,  0.26653308,  0.48063201,
                        0.28948942, -1.56536341],
                       [-1.28833527,  1.51192221,  0.4921519,  0.09354646,  0.45193829,
                        0.09264546,  3.13901949],
                       [0.63996411,  0.3789686, -0.15421736,  0.80494057,  0.37606669,
                        0.45039837,  1.56570542],
                       [0.0053806,  0.47375414,  0.20516208,  0.26344236,  0.47474433,
                        0.2883519,  1.57637453],
                       [-0.01188186,  0.47401467, -0.47031119,  0.26275272,  0.47505132,
                        0.28851427,  1.57725668]])

    floor_plan = [vertices, faces]
    bboxes = bboxes
    robot_width = 0.3
    from time import time
    tt = 0
    for i in range(100):
        t1 = time()
        walkable_metric = cal_walkable_metric(
            floor_plan, floor_plan_centroid, bboxes, robot_width)
        t2 = time()
        tt += (t1-t2)
    print(tt/100)
