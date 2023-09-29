import tensorflow as tf
import numpy as np


def match_choose_adp(prediction, choose, crop_down_factor, resnet_input_shape):
    """
    params prediction: feature maps [B, H, W, c] -> [B, H*W, c]
    params crop_down_factor bs,
    params choose: indexes for chosen points [B, n_points]
    return: tensor [B, n_points, c]
    """

    shape = tf.shape(prediction)
    bs = shape[0]
    c = shape[-1]
    prediction = tf.reshape(prediction, shape=(bs, -1, c))
    batch_resnet_shape = tf.repeat(tf.expand_dims(resnet_input_shape, axis=0), repeats=bs, axis=0)  # bs, 2
    crop_down_factor = tf.expand_dims(crop_down_factor, -1)
    image_shape = tf.multiply(batch_resnet_shape, crop_down_factor)  # bs, 2

    feats_inds = map_indices_to_feature_map(choose, resnet_input_shape, image_shape)
    feats_inds = tf.reshape(feats_inds, shape=(bs, -1, 1))
    pre_match = tf.gather_nd(prediction, indices=feats_inds, batch_dims=1)
    return pre_match



def match_choose(prediction, choose):
    """
    params prediction: feature maps [B, H, W, c] -> [B, H*W, c]
    params choose: indexes for chosen points [B, n_points]
    return: tensor [B, n_points, c]
    """
    shape = tf.shape(prediction)
    bs = shape[0]
    c = shape[-1]
    prediction = tf.reshape(prediction, shape=(bs, -1, c))
    choose = tf.reshape(choose, shape=(bs, -1, 1))
    choose = tf.cast(choose, dtype=tf.int32)
    pre_match = tf.gather_nd(prediction, indices=choose, batch_dims=1)
    return pre_match


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * tf.math.divide_no_nan(inter_area, union_area)


def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate the iou value between the two bounding boxes
    #iou = inter_area / union_area
    iou = tf.math.divide_no_nan(inter_area, union_area)

    # Calculate the coordinates of the upper left corner
    # and the lower right corner of the smallest closed convex surface

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Calculate the GIoU value according to the GioU formula
    #giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
    epsilon = tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
    epsilon = tf.where(tf.math.is_nan(epsilon), 0.0, epsilon)
    giou = iou - 1.0 * epsilon

    return giou


def bbox_ciou(boxes1, boxes2):
    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left = tf.maximum(boxes1_coor[..., 0], boxes2_coor[..., 0])
    up = tf.maximum(boxes1_coor[..., 1], boxes2_coor[..., 1])
    right = tf.maximum(boxes1_coor[..., 2], boxes2_coor[..., 2])
    down = tf.maximum(boxes1_coor[..., 3], boxes2_coor[..., 3])

    c = (right - left) * (right - left) + (up - down) * (up - down)
    iou = bbox_iou(boxes1, boxes2)

    u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) + (boxes1[..., 1] - boxes2[..., 1]) * (
            boxes1[..., 1] - boxes2[..., 1])
    d = u / c

    ar_gt = boxes2[..., 2] / boxes2[..., 3]
    ar_pred = boxes1[..., 2] / boxes1[..., 3]

    ar_loss = 4 / (np.pi * np.pi) * (tf.atan(ar_gt) - tf.atan(ar_pred)) * (tf.atan(ar_gt) - tf.atan(ar_pred))
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss

    return iou - ciou_term


#@tf.function
def map_indices_to_feature_map(indices, resnet_shape, image_shapes):
    """ indices: [b, n_sample_points]
        resnet_shape: (h,w)
        image_shapes: [b, 2]      b x (h,w)
    """
    scales = tf.cast(resnet_shape[0] / image_shapes[:, 0], tf.float32)[..., tf.newaxis]
    rows_inds = tf.cast(tf.floor(tf.cast(indices // image_shapes[:, 1, tf.newaxis], tf.float32) * scales), tf.int32) * \
                resnet_shape[1]
    cols_inds = tf.cast(tf.floor(tf.cast(indices % image_shapes[:, 1, tf.newaxis], tf.float32) * scales), tf.int32)

    return rows_inds + cols_inds


def dpt_2_cld(dpt, cam_scale, cam_intrinsic, xy_offset=(0, 0), depth_trunc=2.0, downsample_factor=1):
    """
    This function converts 2D depth image into 3D point cloud according to camera intrinsic matrix
    :param dpt: the 2d depth image
    :param cam_scale: scale converting units in meters
    :param cam_intrinsic: camera intrinsic matrix
    :param xy_offset: the crop left upper corner index on the original image

    P(X,Y,Z) = (inv(K) * p2d) * depth
    where:  P(X, Y, Z): the 3D points
            inv(K): the inverse matrix of camera intrinsic matrix
            p2d: the [ u, v, 1].T the pixels in the image
            depth: the pixel-wise depth value
     """

    x1, y1 = xy_offset

    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]

    h_depth, w_depth = dpt.shape

    x_map, y_map = np.mgrid[:h_depth, :w_depth] * downsample_factor

    x_map += y1
    y_map += x1

    msk_dp = np.logical_and(dpt > 1e-6, dpt < depth_trunc)

    pcld_index = msk_dp.flatten().nonzero()[0].astype(np.uint32)  # index for nonzero elements

    if len(pcld_index) < 1:
        return None, None

    dpt_mskd = dpt.flatten()[pcld_index][:, np.newaxis].astype(np.float32)
    xmap_mskd = x_map.flatten()[pcld_index][:, np.newaxis].astype(np.float32)
    ymap_mskd = y_map.flatten()[pcld_index][:, np.newaxis].astype(np.float32)

    pt2 = dpt_mskd / cam_scale  # z
    cam_cx, cam_cy = cam_intrinsic[0][2], cam_intrinsic[1][2]
    cam_fx, cam_fy = cam_intrinsic[0][0], cam_intrinsic[1][1]

    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    pcld = np.concatenate((pt0, pt1, pt2), axis=1)

    return pcld, pcld_index


def dpt_2_cld_with_roi(dpt, roi, cam_scale, cam_intrinsic, xy_offset=(0, 0), depth_trunc=2.0, downsample_factor=1):
    """
    This function converts 2D depth image into 3D point cloud according to camera intrinsic matrix and roi (it returns a point cloud inside the roi, depending of depth)
    :param dpt: the 2d depth image
    :param roi: tuple (y1, x1, y2, x2)
    :param cam_scale: scale converting units in meters
    :param cam_intrinsic: camera intrinsic matrix
    :param xy_offset: the crop left upper corner index on the original image

    P(X,Y,Z) = (inv(K) * p2d) * depth
    where:  P(X, Y, Z): the 3D points
            inv(K): the inverse matrix of camera intrinsic matrix
            p2d: the [ u, v, 1].T the pixels in the image
            depth: the pixel-wise depth value
     """
    y1, x1, y2, x2 = roi

    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]

    h_depth, w_depth = dpt.shape

    x_map, y_map = np.mgrid[:h_depth, :w_depth] * downsample_factor
    # invalidate outside of roi
    in_y = np.logical_and(
        y_map[:, :] >= y1,
        y_map[:, :] < y2,
    )
    in_x = np.logical_and(
        x_map[:, :] >= x1,
        x_map[:, :] < x2,
    )
    in_roi = np.logical_and(in_y, in_x)
    
    msk_dp = np.logical_and(dpt > 1e-6, dpt < depth_trunc)
    msk_dp_and_roi = np.logical_and(in_roi, msk_dp)

    pcld_index = msk_dp_and_roi.flatten().nonzero()[0].astype(np.uint32)  # index for nonzero elements
    if len(pcld_index) < 1:
        return None, None

    dpt_mskd = dpt.flatten()[pcld_index][:, np.newaxis].astype(np.float32)
    xmap_mskd = x_map.flatten()[pcld_index][:, np.newaxis].astype(np.float32)
    ymap_mskd = y_map.flatten()[pcld_index][:, np.newaxis].astype(np.float32)

    pt2 = dpt_mskd / cam_scale  # z
    cam_cx, cam_cy = cam_intrinsic[0][2], cam_intrinsic[1][2]
    cam_fx, cam_fy = cam_intrinsic[0][0], cam_intrinsic[1][1]
    
    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    pcld = np.concatenate((pt0, pt1, pt2), axis=1)

    return pcld, pcld_index



#@tf.function
def dpt_2_cld_tf(dpt, cam_scale, cam_intrinsic, xy_offset=(0, 0), depth_trunc=2.0):
    import tensorflow as tf
    """
    This function converts 2D depth image into 3D point cloud according to camera intrinsic matrix
    :param dpt: the 2d depth image
    :param cam_scale: scale converting units in meters
    :param cam_intrinsic: camera intrinsic matrix
    :param xy_offset: the crop left upper corner index on the original image

    P(X,Y,Z) = (inv(K) * p2d) * depth
    where:  P(X, Y, Z): the 3D points
            inv(K): the inverse matrix of camera intrinsic matrix
            p2d: the [ u, v, 1].T the pixels in the image
            depth: the pixel-wise depth value
     """

    h_depth = tf.shape(dpt)[0]
    w_depth = tf.shape(dpt)[1]

    y_map, x_map = tf.meshgrid(tf.range(w_depth, dtype=tf.float32),
                               tf.range(h_depth, dtype=tf.float32))  # vice versa than mgrid

    x_map = x_map + tf.cast(xy_offset[1], tf.float32)
    y_map = y_map + tf.cast(xy_offset[0], tf.float32)

    msk_dp = tf.math.logical_and(dpt > 1e-6, dpt < depth_trunc)
    msk_dp = tf.reshape(msk_dp, (-1,))

    pcld_index = tf.squeeze(tf.where(msk_dp))

    dpt_mskd = tf.expand_dims(tf.gather(tf.reshape(dpt, (-1,)), pcld_index), -1)
    xmap_mskd = tf.expand_dims(tf.gather(tf.reshape(x_map, (-1,)), pcld_index), -1)
    ymap_mskd = tf.expand_dims(tf.gather(tf.reshape(y_map, (-1,)), pcld_index), -1)

    pt2 = dpt_mskd / tf.cast(cam_scale, dpt_mskd.dtype)  # z
    cam_cx, cam_cy = cam_intrinsic[0][2], cam_intrinsic[1][2]
    cam_fx, cam_fy = cam_intrinsic[0][0], cam_intrinsic[1][1]

    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    pcld = tf.concat((pt0, pt1, pt2), axis=1)

    return pcld, pcld_index


def pcld_processor_tf(depth, rgb, camera_matrix, camera_scale,
                      n_sample_points, xy_ofst=(0, 0), depth_trunc=2.0):
    points, valid_inds = dpt_2_cld_tf(depth, camera_scale, camera_matrix, xy_ofst, depth_trunc=depth_trunc)

    import tensorflow as tf

    n_valid_inds = tf.shape(valid_inds)[0]
    sampled_inds = tf.range(n_valid_inds)

    if n_valid_inds < 10:
        # because tf.function: return same dtypes
        return tf.constant([0.]), tf.constant([0.]), tf.constant([0], valid_inds.dtype)

    if n_valid_inds >= n_sample_points:
        sampled_inds = tf.random.shuffle(sampled_inds)

    else:
        repeats = tf.cast(tf.math.ceil(n_sample_points / n_valid_inds), tf.int32)
        sampled_inds = tf.tile(sampled_inds, [repeats])

    sampled_inds = sampled_inds[:n_sample_points]

    final_inds = tf.gather(valid_inds, sampled_inds)

    points = tf.gather(points, sampled_inds)

    rgbs = tf.reshape(rgb, (-1, 3))
    rgbs = tf.gather(rgbs, final_inds)

    normals = compute_normals(depth, camera_matrix)
    normals = tf.gather(normals, final_inds)

    feats = tf.concat([rgbs, normals], 1)
    return points, feats, final_inds


def old_get_offst(RT_list, pcld_xyz, mask_selected, n_objects, cls_type, centers, kpts, mask_value=255):
    """
        num_obj is == num_classes by default
    """
    RTs = np.zeros((n_objects, 3, 4))
    kp3ds = np.zeros((n_objects, 8, 3))
    ctr3ds = np.zeros((n_objects, 1, 3))
    kpts_targ_ofst = np.zeros((len(pcld_xyz), 8, 3))
    ctr_targ_ofst = np.zeros((len(pcld_xyz), 1, 3))

    for i in range(len(RT_list)):
        RTs[i] = RT_list[i]  # assign RT to each object
        r = RT_list[i][:3, :3]
        t = RT_list[i][:3, 3]

        if cls_type == 'all':
            pass
        else:
            cls_index = np.where(mask_selected == mask_value)[0]  # todo 255 for object

        ctr = centers[i][:, None]
        ctr = np.dot(ctr.T, r.T) + t  # ctr [1 , 3]
        ctr3ds[i] = ctr

        ctr_offset_all = []

        for c in ctr:
            ctr_offset_all.append(np.subtract(c, pcld_xyz))

        ctr_offset_all = np.array(ctr_offset_all).transpose((1, 0, 2))
        ctr_targ_ofst[cls_index, :, :] = ctr_offset_all[cls_index, :, :]

        kpts = kpts[i]
        kpts = np.dot(kpts, r.T) + t  # [8, 3]
        kp3ds[i] = kpts

        kpts_offset_all = []

        for kp in kpts:
            kpts_offset_all.append(np.subtract(kp, pcld_xyz))

        kpts_offset_all = np.array(kpts_offset_all).transpose((1, 0, 2))  # [kp, np, 3] -> [nsp, kp, 3]

        kpts_targ_ofst[cls_index, :, :] = kpts_offset_all[cls_index, :, :]

    return ctr_targ_ofst, kpts_targ_ofst


def get_pcld_rgb(rgb, pcld_index):
    """
    rgb : [h, w, c]
    return: pcld_rgb [h*w, c]
    """
    return rgb.reshape((-1, 3))[pcld_index]

#@tf.function
def compute_normal_map(depth, camera_matrix):
    kernel = np.array([[[[0.5, 0.5]], [[-0.5, 0.5]]], [[[0.5, -0.5]], [[-0.5, -0.5]]]])
    # print('Kernel filter shape ', kernel.shape)

    diff = tf.nn.conv2d(depth, kernel, 1, "VALID")

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    scale_depth = tf.concat([depth / fx, depth / fy], -1)

    # clip=tf.constant(1)
    # diff = tf.clip_by_value(diff, -clip, clip)
    diff = diff / scale_depth[:, :-1, :-1, :]  # allow nan -> filter later

    mask = tf.logical_and(~tf.math.is_nan(diff), tf.abs(diff) < 5)

    diff = tf.where(mask, diff, 0.)

    smooth = tf.constant(4)
    kernel2 = tf.cast(tf.tile([[1 / tf.pow(smooth, 2)]], (smooth, smooth)), tf.float32)
    kernel2 = tf.expand_dims(tf.expand_dims(kernel2, axis=-1), axis=-1)
    kernel2 = kernel2 * tf.eye(2, batch_shape=(1, 1))
    diff2 = tf.nn.conv2d(diff, kernel2, 1, "VALID")

    mask_conv = tf.nn.conv2d(tf.cast(mask, tf.float32)
                             , kernel2, 1, "VALID")

    diff2 = diff2 / mask_conv

    ones = tf.expand_dims(tf.ones(diff2.shape[:3]), -1)
    v_norm = tf.concat([diff2, ones], axis=-1)

    v_norm, _ = tf.linalg.normalize(v_norm, axis=-1)
    v_norm = tf.where(~tf.math.is_nan(v_norm), v_norm, [0])

    v_norm = - tf.image.resize_with_crop_or_pad(v_norm, depth.shape[1], depth.shape[2])  # pad and flip (towards cam)
    return v_norm



def compute_normals(depth, camera_matrix):
    import tensorflow as tf
    # depth = tf.expand_dims(tf.expand_dims(depth, axis=-1), axis=0)
    depth = tf.expand_dims(depth, axis=0)
    # print('Depth shape during Compute_normals', tf.shape(depth))
    normal_map = compute_normal_map(depth, camera_matrix)
    normals = tf.reshape(normal_map[0], (-1, 3))  # reshape als list of normals
    return normals


