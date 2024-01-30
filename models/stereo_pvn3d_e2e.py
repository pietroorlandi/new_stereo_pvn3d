import tensorflow.keras as keras
import tensorflow as tf
from .pprocessnet import _InitialPoseModel
from models.pointnet2_tf import _PointNet2TfXModel, PointNet2Params
from models.disparity_decoder import DisparityDecoder, DisparityDecoderParams
from models.resnet_encoder import ResnetEncoder, ResnetEncoderParams
from .mlp import MlpNets, MlpNetsParams
from typing import Dict, List
from .pointnet_light import PointNetLightParams, _PointNetLightModel, _PointNetMini


class StereoPvn3dE2E(keras.Model):
    def __init__(
        self,
        *,
        num_pts: int,
        num_kpts: int,
        num_cls: int,
        num_cpts: int,
        dim_xyz: int,
        use_disparity: bool,
        relative_disparity: bool,
        res_encoder_params : Dict, 
        disp_decoder_params: Dict,
        point_net2_params: Dict,
        point_net_light_params: Dict,
        mlp_params: Dict,
        use_pointnet2: bool,
        use_pointnet_light: bool, 
        use_pointnet_mini: bool, 
        **kwargs,
    ):
        super(StereoPvn3dE2E, self).__init__()
        self.num_pts = num_pts
        self.num_kpts = num_kpts
        self.num_cls = num_cls
        self.num_cpts = num_cpts
        self.dim_xyz = dim_xyz
        self.use_disparity = use_disparity
        self.relative_disparity = relative_disparity
        self.use_pointnet2 = use_pointnet2
        self.use_pointnet_light = use_pointnet_light
        self.use_pointnet_mini = use_pointnet_mini
                
        self.resenc_params = ResnetEncoderParams(**res_encoder_params)
        self.disp_dec_params = DisparityDecoderParams(**disp_decoder_params)
        self.pointnet2params = PointNet2Params(**point_net2_params)
        self.pointnetlightparams = PointNetLightParams(**point_net_light_params)
        self.mlp_params = MlpNetsParams(**mlp_params)

        self.s1 = tf.Variable(0., trainable = True) # try to give him less weight, 1.1, try also to focus on the mse by setting 0.9
        self.s2 = tf.Variable(0., trainable = True)
        self.s3 = tf.Variable(0., trainable = True)
        self.s4 = tf.Variable(0., trainable = True)
        self.s5 = tf.Variable(0., trainable = True)

        # Resnet Encoder model
        res_enc = ResnetEncoder(self.resenc_params)
        self.resnet_lr = res_enc.build_resnet()
        # Disparity Decoder model
        self.disparity_decoder = DisparityDecoder(self.disp_dec_params)
        # PointNet Mini model
        self.pointnet_mini_model = _PointNetMini()

        # PointNet Light model
        self.pointnet_light_model = _PointNetLightModel(
                self.pointnetlightparams, num_classes=self.num_cls)
        # PointNet++ model
        self.pointnet2_model = _PointNet2TfXModel(
                self.pointnet2params, num_classes=self.num_cls)
        

        
        if self.use_pointnet2:
            num_mlp_input_features = self.pointnet2params.num_out_features + self.disp_dec_params.num_decoder_feats - 1
        elif self.use_pointnet_light:
            num_mlp_input_features = self.pointnetlightparams.num_out_features + self.disp_dec_params.num_decoder_feats - 1
        #     num_mlp_input_features =  
        elif self.use_pointnet_mini:
            num_mlp_input_features = 1088 + self.disp_dec_params.num_decoder_feats - 1
        #     num_mlp_input_features
        else:
            num_mlp_input_features = self.disp_dec_params.num_decoder_feats + 6 - 1 # add 3 point's coordinates xyz, 3 surface normals coordinates, remove disparity channel 
        # MLP model 

        self.mlp_net = MlpNets(self.mlp_params,
                               num_pts= self.num_pts,
                               num_kpts= self.num_kpts,
                               num_cls= self.num_cls,
                               num_cpts= self.num_cpts,
                               channel_xyz= self.dim_xyz)
        

        self.mlp_model = self.mlp_net.build_mlp_model(rgbd_features_shape=(self.num_pts, num_mlp_input_features )) #self.disp_dec_params.num_decoder_feats + 3)) # add the features of Ponintnet
        self.initial_pose_model = _InitialPoseModel()
        # self.sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        # self.sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        


    def sample_index(self, b, h, w, roi, num_sample_points):
        y1, x1, y2, x2 = roi[:, 0], roi[:, 1], roi[:, 2], roi[:, 3]
        x_map, y_map = tf.meshgrid(tf.range(w, dtype=tf.int32), tf.range(h, dtype=tf.int32))
        # print(x_map)
        y1 = y1[:, tf.newaxis, tf.newaxis]  # add, h, w dims
        x1 = x1[:, tf.newaxis, tf.newaxis]
        y2 = y2[:, tf.newaxis, tf.newaxis]
        x2 = x2[:, tf.newaxis, tf.newaxis]
        # invalidate outside of roi
        in_y = tf.logical_and(y_map[tf.newaxis, :, :] >= y1, y_map[tf.newaxis, :, :] < y2,)
        # print('in_y', in_y)
        in_x = tf.logical_and(x_map[tf.newaxis, :, :] >= x1,x_map[tf.newaxis, :, :] < x2,)
        in_roi = tf.logical_and(in_y, in_x)
        # depth = tf.ones((b, h, w, 1))
        # print(depth.shape)

        # get masked indices (valid truncated depth inside of roi)
        # is_valid = tf.logical_and(depth[..., 0] > 1e-6, depth[..., 0] < 2.0)
        # inds = tf.where(tf.logical_and(in_roi, is_valid))
        inds = tf.where(in_roi)
        inds = tf.cast(inds, tf.int32)  # [None, 3]


        inds = tf.random.shuffle(inds)
        # print(f'inds{inds} - inds.shape{inds.shape}')
        # print('inds[:,0]\n',inds[:,0])
        # print('shape pf roi\n',tf.shape(roi)[0])

        # split index list into [b, None, 3] ragged tensor by using the batch index
        inds = tf.ragged.stack_dynamic_partitions(
            inds, inds[:, 0], tf.shape(roi)[0]
        )  # [b, None, 3]
        # TODO if we dont have enough points, we pad the indices with 0s, how to handle that?
        inds = inds[:, :num_sample_points].to_tensor(shape=(b, num_sample_points, 3), default_value=0)  # [b, num_points, 3]
        # _, tensor_points, _ = tf.shape(inds)
        # if tensor_points!=num_sample_points:
        #     print(f'tensor points {tensor_points} different from num_sample_pts {num_sample_points}')
        #     inds = tf.pad(inds, tf.constant([[0,0],[0, num_sample_points-tensor_points], [0,0]]))
        # print(f'inds: {inds} - inds.shape:{inds.shape}')
        return inds

    @tf.function
    def call(self, inputs, training=None):
        full_rgb_l = inputs[0]
        full_rgb_r = inputs[1]  # another layer        
        b, h, w = tf.shape(full_rgb_l)[0], tf.shape(full_rgb_l)[1], tf.shape(full_rgb_r)[2]
        baseline = inputs[2]
        K = inputs[3][0]
        focal_length = inputs[3][:, 0, 0]
        intrinsics = inputs[3]
        roi = inputs[4]
        mesh_kpts = inputs[5]

        sampled_inds_in_original_image = self.sample_index(b, h, w, roi, self.num_pts)

        # crop the image to the aspect ratio for resnet and integer crop factor
        bbox, crop_factor, w_factor_inv, h_factor_inv = self.get_crop_index(
            roi, h, w, self.resenc_params.resnet_input_shape[0], self.resenc_params.resnet_input_shape[1], return_w_h_factors=True
        )  # bbox: [b, 4], crop_factor: [b]

        sampled_inds_in_roi = self.transform_indices_from_full_image_cropped(
            sampled_inds_in_original_image, bbox, crop_factor
        )

        norm_bbox = tf.cast(bbox / [h, w, h, w], tf.float32)  # normalize bounding box
        cropped_rgbs_l = tf.image.crop_and_resize(
            tf.cast(full_rgb_l, tf.float32),
            norm_bbox,
            tf.range(tf.shape(full_rgb_l)[0]),
            self.resenc_params.resnet_input_shape[:2],
        )/255.
        cropped_rgbs_r = tf.image.crop_and_resize(
            tf.cast(full_rgb_r, tf.float32),
            norm_bbox,
            tf.range(tf.shape(full_rgb_r)[0]),
            self.resenc_params.resnet_input_shape[:2],
        )/255.

        # stop gradients for preprocessing
        cropped_rgbs_l = tf.stop_gradient(cropped_rgbs_l)
        cropped_rgbs_r = tf.stop_gradient(cropped_rgbs_r)
        sampled_inds_in_original_image = tf.stop_gradient(sampled_inds_in_original_image)
        sampled_inds_in_roi = tf.stop_gradient(sampled_inds_in_roi)

        f_l_1, f_l_2, f_l_3, f_l_4, f_l_5 = self.resnet_lr(cropped_rgbs_l)
        f_r_1, f_r_2, f_r_3, f_r_4, f_r_5 = self.resnet_lr(cropped_rgbs_r)
        
        stereo_outputs, attended_right, weights = self.disparity_decoder([[f_l_1, f_l_2, f_l_3, f_l_4, f_l_5],
                                                                [f_r_1, f_r_2, f_r_3, f_r_4, f_r_5]])

        if self.use_disparity:
            disp = stereo_outputs[...,:1] # self.disp_head(stereo_outputs)
            if self.relative_disparity:
                disp = (
                    disp * tf.shape(inputs[0])[2]
                )  # BHWC -> *W converts to absolute disparity
            y = baseline[:] * focal_length[:] * w_factor_inv[:]
            y = tf.reshape(y, [-1, 1 ,1, 1])
            disp = disp * 100 # in the first step disp is too low and depth would be too high
            disp = tf.where(tf.math.abs(disp)<0.9, 0.0, disp) # delete disp <= 1
            depth = tf.math.divide_no_nan(y, disp)
        else:
            stereo_outputs = tf.nn.leaky_relu(stereo_outputs)
            depth = stereo_outputs[...,:1] # self.disp_head(stereo_outputs)

        b_new_intrinsics = self.compute_new_b_intrinsics_camera(bbox, crop_factor, intrinsics) # change intrinsics since depth is cropped and scaled
        xyz_pred = self.pcld_processor_tf_by_index(depth+0.00001, b_new_intrinsics, sampled_inds_in_roi) 
        normal_feats = StereoPvn3dE2E.compute_normal_map(depth+0.00001, b_new_intrinsics) # (b, res_h, res_w, 3)
        normal_feats = tf.gather_nd(normal_feats, sampled_inds_in_roi) # (b, n_points, 3)
        
        depth_emb = tf.gather_nd(depth, sampled_inds_in_roi)
        rgb_emb = tf.gather_nd(stereo_outputs[..., 1:], sampled_inds_in_roi) # tf.where(mask, sampled_inds_in_roi, 0))
        feats = tf.concat([xyz_pred, normal_feats, rgb_emb], axis = -1)

        if self.use_pointnet2:
            pcld_emb = self.pointnet2_model((xyz_pred, feats), training=training)
        elif self.use_pointnet_light:
            pcld_emb = self.pointnet_light_model((xyz_pred, feats), training=training)
        elif self.use_pointnet_mini:
            pcld_emb = self.pointnet_mini_model((feats), training=training)
            print(f"pcld_emb.shape: {pcld_emb.shape}")
        else:
            pcld_emb = tf.concat([xyz_pred, normal_feats], axis = -1)


        feats_fused = tf.concat([pcld_emb, rgb_emb], axis = -1)
        kp, sm, cp = self.mlp_model(feats_fused, training=training)

        if training:
            return (depth, kp, sm, cp, xyz_pred, sampled_inds_in_original_image, mesh_kpts, norm_bbox, cropped_rgbs_l, cropped_rgbs_r, weights, attended_right, intrinsics, crop_factor, w_factor_inv, h_factor_inv, disp, depth_emb, normal_feats)
        else:
            batch_R, batch_t, voted_kpts = self.initial_pose_model([xyz_pred, kp, cp, sm, mesh_kpts])
            return (
                batch_R,
                batch_t,
                voted_kpts,
                (depth, kp, sm, cp, xyz_pred, sampled_inds_in_original_image, mesh_kpts, norm_bbox, cropped_rgbs_l, cropped_rgbs_r, weights, attended_right, intrinsics, crop_factor, w_factor_inv, h_factor_inv, disp, depth_emb, normal_feats)
            )


    def compute_errors(gt, pred):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = tf.maximum((gt / pred), (pred / gt))
        a1 = tf.reduce_mean(tf.cast(thresh < 1.25, tf.float32))
        a2 = tf.reduce_mean(tf.cast(thresh < 1.25 ** 2, tf.float32))
        a3 = tf.reduce_mean(tf.cast(thresh < 1.25 ** 3, tf.float32))

        rmse = tf.sqrt(tf.reduce_mean((gt - pred) ** 2))

        rmse_log = tf.sqrt(tf.reduce_mean((tf.log(gt) - tf.log(pred)) ** 2))

        abs_rel = tf.reduce_mean(tf.abs(gt - pred) / gt)

        sq_rel = tf.reduce_mean(((gt - pred) ** 2) / gt)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    @staticmethod
    def replace_values_kp_of_invalid_indices(b_kp_cp, b_index, value=0.):
        """
        Replaces all values of b_kp_cp that correspond to invalid indices [-1, -1, -1] with a value chosen as a parameter
        Args:
            b_kp_cp: (b, n_points, n_kpts_cpts, 3)
            b_index: (b, n_points, 3)
        Returns:
            b_kp_cp_out (b, n_points, kp_cp_size, 3)
        """
        b = b_kp_cp.shape[0]
        n_points = b_kp_cp.shape[1]
        n_kpts_cpts = b_kp_cp.shape[2]
        b_mean_idx = tf.math.reduce_mean(b_index, axis=-1)
        b_valid_mask = b_mean_idx > 0 # (b, n_points)
        b_expanded_valid_mask = tf.reshape(b_valid_mask, [b, n_points, 1, 1]) # (b, n_points, 1, 1)
        b_expanded_valid_mask = tf.tile(b_expanded_valid_mask, (1, 1, n_kpts_cpts, 3)) 
        b_kp_cp_out = tf.where(b_expanded_valid_mask, b_kp_cp, value)
        return b_kp_cp_out
    
    @staticmethod
    def replace_values_seg_of_invalid_indices(b_seg, b_index, value=0.):
        """
        Replaces all values of b_seg that correspond to invalid indices [-1, -1, -1] with a value chosen as a parameter
        Args:
            b_seg: (b, n_points, 1)
            b_index: (b, n_points, 3)
        Returns:
            b_seg_out (b, n_points, 1)
        """
        b_mean_idx = tf.math.reduce_mean(b_index, axis=-1)
        b_valid_mask = b_mean_idx > 0 # (b, n_points)
        b_valid_mask = tf.expand_dims(b_valid_mask, axis=-1) # (b, n_points, 1)
        b_seg_out = tf.where(b_valid_mask, b_seg, value)
        return b_seg_out


    @staticmethod
    def compute_normal_map(depth, camera_matrix):
        kernel = tf.constant([[[[0.5, 0.5]], [[-0.5, 0.5]]], [[[0.5, -0.5]], [[-0.5, -0.5]]]])

        diff = tf.nn.conv2d(depth, kernel, 1, "SAME")

        fx, fy = camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]  # [b,]
        f = tf.stack([fx, fy], axis=-1)[:, tf.newaxis, tf.newaxis, :]  # [b, 1, 1, 2]

        diff = diff * f / depth

        mask = tf.logical_and(~tf.math.is_nan(diff), tf.abs(diff) < 5)

        diff = tf.where(mask, diff, 0.0)

        # smooth = tf.constant(4)
        # kernel2 = tf.cast(tf.tile([[1 / tf.pow(smooth, 2)]], (smooth, smooth)), tf.float32)
        # kernel2 = tf.expand_dims(tf.expand_dims(kernel2, axis=-1), axis=-1)
        # kernel2 = kernel2 * tf.eye(2, batch_shape=(1, 1))
        # diff2 = tf.nn.conv2d(diff, kernel2, 1, "SAME")

        # mask_conv = tf.nn.conv2d(tf.cast(mask, tf.float32), kernel2, 1, "SAME")

        # diff2 = diff2 / mask_conv
        diff2 = diff

        ones = tf.ones(tf.shape(diff2)[:3])[..., tf.newaxis]
        v_norm = tf.concat([diff2, ones], axis=-1)

        v_norm, _ = tf.linalg.normalize(v_norm, axis=-1)
        v_norm = tf.where(~tf.math.is_nan(v_norm), v_norm, [0])

        # v_norm = -tf.image.resize_with_crop_or_pad(
        #    v_norm, tf.shape(depth)[1], tf.shape(depth)[2]
        # )  # pad and flip (towards cam)
        return -v_norm


    @staticmethod
    def pcld_processor_tf(rgb, depth, camera_matrix, roi, num_sample_points, depth_trunc=2.0):
        """This function calculats a pointcloud from a RGB-D image and returns num_sample_points
        points from the pointcloud, randomly selected in the ROI specified.

        Args:
            rgb (b,h,w,3): RGB image with values in [0,255]
            depth (b,h,w,1): Depth image with float values in meters
            camera_matrix (b,3,3): Intrinsic camera matrix in OpenCV format
            roi (b,4): Region of Interest in the image (y1,x1,y2,x2)
            num_sample_points (int): Number of sampled points per image
            depth_trunc (float, optional): Truncate the depth image. Defaults to 2.0.

        Returns:
            xyz (b, num_sample_points, 3): Sampled pointcloud in m.
            feats (b, num_sample_points, 6): Feature pointcloud (RGB + normals)
            inds (b, num_sample_points, 3): Indices of the sampled points in the image
        """

        y1, x1, y2, x2 = roi[:, 0], roi[:, 1], roi[:, 2], roi[:, 3]

        # normals = PVN3D_E2E.compute_normals(depth, camera_matrix) # [b, h*w, 3]
        normal_map = StereoPvn3dE2E.compute_normal_map(depth, camera_matrix)  # [b, h,w,3]

        h_depth = tf.shape(depth)[1]
        w_depth = tf.shape(depth)[2]
        x_map, y_map = tf.meshgrid(
            tf.range(w_depth, dtype=tf.int32), tf.range(h_depth, dtype=tf.int32)
        )
        y1 = y1[:, tf.newaxis, tf.newaxis]  # add, h, w dims
        x1 = x1[:, tf.newaxis, tf.newaxis]
        y2 = y2[:, tf.newaxis, tf.newaxis]
        x2 = x2[:, tf.newaxis, tf.newaxis]

        # invalidate outside of roi
        in_y = tf.logical_and(
            y_map[tf.newaxis, :, :] >= y1,
            y_map[tf.newaxis, :, :] < y2,
        )
        in_x = tf.logical_and(
            x_map[tf.newaxis, :, :] >= x1,
            x_map[tf.newaxis, :, :] < x2,
        )
        in_roi = tf.logical_and(in_y, in_x)

        # get masked indices (valid truncated depth inside of roi)
        is_valid = tf.logical_and(depth[..., 0] > 1e-6, depth[..., 0] < depth_trunc)
        inds = tf.where(tf.logical_and(in_roi, is_valid))
        inds = tf.cast(inds, tf.int32)  # [None, 3]

        inds = tf.random.shuffle(inds)

        # split index list into [b, None, 3] ragged tensor by using the batch index
        inds = tf.ragged.stack_dynamic_partitions(
            inds, inds[:, 0], tf.shape(rgb)[0]
        )  # [b, None, 3]

        # TODO if we dont have enough points, we pad the indices with 0s, how to handle that?
        inds = inds[:, :num_sample_points].to_tensor()  # [b, num_points, 3]

        # calculate xyz
        cam_cx, cam_cy = camera_matrix[:, 0, 2], camera_matrix[:, 1, 2]
        cam_fx, cam_fy = camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]

        # inds[..., 0] == index into batch
        # inds[..., 1:] == index into y_map and x_map,  b times
        sampled_ymap = tf.gather_nd(y_map, inds[:, :, 1:])  # [b, num_points]
        sampled_xmap = tf.gather_nd(x_map, inds[:, :, 1:])  # [b, num_points]
        sampled_ymap = tf.cast(sampled_ymap, tf.float32)
        sampled_xmap = tf.cast(sampled_xmap, tf.float32)

        # z = tf.gather_nd(roi_depth, inds)  # [b, num_points]
        z = tf.gather_nd(depth[..., 0], inds)  # [b, num_points]
        x = (sampled_xmap - cam_cx[:, tf.newaxis]) * z / cam_fx[:, tf.newaxis]
        y = (sampled_ymap - cam_cy[:, tf.newaxis]) * z / cam_fy[:, tf.newaxis]
        xyz = tf.stack((x, y, z), axis=-1)

        rgb_feats = tf.gather_nd(rgb, inds)
        normal_feats = tf.gather_nd(normal_map, inds)
        feats = tf.concat([rgb_feats, normal_feats], -1)

        return xyz, feats, inds

    @staticmethod
    def get_crop_index(roi, in_h, in_w, resnet_h, resnet_w, px_max_disp = 200, return_w_h_factors=False):
        """Given a ROI [y1,x1,y2,x2] in an image with dimensions [in_h, in_w]]
        this function returns the indices and the integer crop factor to crop the image
        according to the original roi, but with the same aspect ratio as [resnet_h, resnet_w]
        and scaled to integer multiple of [resnet_h,resnet_w].
        The resulting crop is centered around the roi center and encompases the whole roi.
        Additionally, the crop indices do not exceed the image dimensions.

        Args:
            roi (b,4): Region of Interest in the image (y1,x1,y2,x2)
            in_h (b,): Batchwise image height
            in_w (b,): Batchwise image width
            resnet_h (int): Height of the resnet input
            resnet_w (int): Width of the resnet input

        Returns:
            bbox (b, 4): Modified bounding boxes
            crop_factor (b,): Integer crop factor
        """

        y1, x1, y2, x2 = roi[:, 0], tf.math.maximum(tf.zeros_like(roi[:, 1]), tf.math.subtract(roi[:,1], px_max_disp)), roi[:, 2], roi[:, 3]
        # y1, x1, y2, x2 = roi[:, 0], roi[:, 1], roi[:, 2], roi[:, 3]

        x_c = tf.cast((x1 + x2) / 2, tf.int32)
        y_c = tf.cast((y1 + y2) / 2, tf.int32)

        bbox_w, bbox_h = (x2 - x1), (y2 - y1)
        w_factor = bbox_w / resnet_w  # factor to scale down to resnet shape
        h_factor = bbox_h / resnet_h
        crop_factor = tf.cast(tf.math.ceil(tf.maximum(w_factor, h_factor)), tf.int32)

        crop_w = resnet_w * crop_factor
        crop_h = resnet_h * crop_factor

        x1_new = x_c - tf.cast(crop_w / 2, tf.int32)
        x2_new = x_c + tf.cast(crop_w / 2, tf.int32)
        y1_new = y_c - tf.cast(crop_h / 2, tf.int32)
        y2_new = y_c + tf.cast(crop_h / 2, tf.int32)

        x2_new = tf.where(x1_new < 0, crop_w, x2_new)
        x1_new = tf.where(x1_new < 0, 0, x1_new)

        x1_new = tf.where(x2_new > in_w, in_w - crop_w, x1_new)
        x2_new = tf.where(x2_new > in_w, in_w, x2_new)

        y2_new = tf.where(y1_new < 0, crop_h, y2_new)
        y1_new = tf.where(y1_new < 0, 0, y1_new)

        y1_new = tf.where(y2_new > in_h, in_h - crop_h, y1_new)
        y2_new = tf.where(y2_new > in_h, in_h, y2_new)

        if return_w_h_factors:
            return tf.stack([y1_new, x1_new, y2_new, x2_new], axis=-1), crop_factor, tf.cast(1/w_factor, tf.float32), tf.cast(1/h_factor, tf.float32)
        else:
            return tf.stack([y1_new, x1_new, y2_new, x2_new], axis=-1), crop_factor
             
    @staticmethod
    def compute_new_b_intrinsics_camera(b_roi_cropped, b_crop_factor, b_intrinsics):
        """
        Return the new b_intrinsics_camera given the b_roi_cropped and b_crop_factor
        TODO: check if is right to use b_roi_cropped instead of b_roi_original

        Args:
            b_roi_cropped: (b, 4): Region of Interest enlarged for stereo crop in the image (y1,x1,y2,x2)
            b_crop_factor: (b,)
            b_intrinsics: (b, 3, 3)

        Returns:
            b_updated_intrinsic_matrix: (b, 3, 3) the new intrinsics matrix considering crop and scaling
        """
        cam_cx, cam_cy = b_intrinsics[:, 0, 2], b_intrinsics[:, 1, 2]
        cam_fx, cam_fy = b_intrinsics[:, 0, 0], b_intrinsics[:, 1, 1]

        b_roi_cropped = tf.cast(b_roi_cropped, dtype=tf.int32)
        b_roi_cropped[:, 0]
        y0, x0, y1, x1 = b_roi_cropped[:, 0], b_roi_cropped[:, 1], b_roi_cropped[:, 2], b_roi_cropped[:, 3]

        # Update cx and cy after cropping
        cam_cx = cam_cx - tf.cast(x0, dtype=tf.float32)
        cam_cy = cam_cy - tf.cast(y0, dtype=tf.float32)

        # Update cx, cy, fx and fy after scaling
        cam_cx = cam_cx / tf.cast(b_crop_factor, dtype=tf.float32)
        cam_cy = cam_cy / tf.cast(b_crop_factor, dtype=tf.float32)
        cam_fx = cam_fx / tf.cast(b_crop_factor, dtype=tf.float32)
        cam_fy = cam_fy / tf.cast(b_crop_factor, dtype=tf.float32)

        row1 = tf.expand_dims(tf.stack([cam_fx, b_intrinsics[:,0,1], cam_cx], axis=1), axis=1)
        row2 = tf.expand_dims(tf.stack([b_intrinsics[:,1,0], cam_fy, cam_cy], axis=1), axis=1)
        row3 = tf.expand_dims(tf.stack([b_intrinsics[:,2,0], b_intrinsics[:,2,1], b_intrinsics[:,2,2]], axis=1), axis=1)
        result_matrix = tf.concat([row1, row2, row3], axis=1)
        return result_matrix

    @staticmethod
    def pcld_processor_tf_by_index(b_depth_pred, b_camera_matrix, b_sampled_index):
        h_depth = tf.shape(b_depth_pred)[1]
        w_depth = tf.shape(b_depth_pred)[2]
        x_map, y_map = tf.meshgrid(
            tf.range(w_depth, dtype=tf.int32), tf.range(h_depth, dtype=tf.int32)
        )

        # calculate xyz
        cam_cx, cam_cy = b_camera_matrix[:, 0, 2], b_camera_matrix[:, 1, 2]
        cam_fx, cam_fy = b_camera_matrix[:, 0, 0], b_camera_matrix[:, 1, 1]

        # inds[..., 0] == index into batch
        # inds[..., 1:] == index into y_map and x_map,  b times
        sampled_ymap = tf.gather_nd(y_map, b_sampled_index[:, :, 1:])  # [b, num_points]
        sampled_xmap = tf.gather_nd(x_map, b_sampled_index[:, :, 1:])  # [b, num_points]
        sampled_ymap = tf.cast(sampled_ymap, tf.float32)
        sampled_xmap = tf.cast(sampled_xmap, tf.float32)

        # z = tf.gather_nd(roi_depth, inds)  # [b, num_points]
        z = tf.gather_nd(b_depth_pred[..., 0], b_sampled_index)  # [b, num_points]
        x = (sampled_xmap - cam_cx[:, tf.newaxis]) * z / cam_fx[:, tf.newaxis]
        y = (sampled_ymap - cam_cy[:, tf.newaxis]) * z / cam_fy[:, tf.newaxis]
        xyz = tf.stack((x, y, z), axis=-1)

        return xyz
    
    @staticmethod
    def transform_indices_from_full_image_cropped(
        sampled_inds_in_original_image, bbox, crop_factor
    ):
        """Transforms indices from full image to croppend and rescaled images.
        Original indices [b, h, w, 3] with the last dimensions as indices into [b, h, w]
        are transformed and have same shape [b,h,w,3], however the indices now index
        into the cropped and rescaled images according to the bbox and crop_factor

        To be used with tf.gather_nd
        Since the first index refers to the batch, no batch_dims is needed

        Examples:
            Index: [500,500] with bounding box [500,500,...] is transform to [0,0]
            index [2, 2] with bounding box [0, 0] and crop_factor 2 is transformed to [1,1]


        Args:
            sampled_inds_in_original_image (b,h,w,3): Indices into the original image
            bbox (b,4): Region of Interest in the image (y1,x1,y2,x2)
            crop_factor (b,): Integer crop factor

        Returns:
            sampled_inds_in_roi (b,h,w,3): Indices into the cropped and rescaled image
        """
        b = tf.shape(sampled_inds_in_original_image)[0]

        # sampled_inds_in_original_image: [b, num_points, 3]
        # with last dimension is index into [b, h, w]
        # crop_factor: [b, ]

        crop_top_left = tf.concat((tf.zeros((b, 1), tf.int32), bbox[:, :2]), -1)  # [b, 3]

        sampled_inds_in_roi = sampled_inds_in_original_image - crop_top_left[:, tf.newaxis, :]

        # apply scaling to indices, BUT ONLY H AND W INDICES (and not batch index)
        crop_factor_bhw = tf.concat(
            (
                tf.ones((b, 1), dtype=tf.int32),
                crop_factor[:, tf.newaxis],
                crop_factor[:, tf.newaxis],
            ),
            -1,
        )  # [b, 3]
        sampled_inds_in_roi = sampled_inds_in_roi / crop_factor_bhw[:, tf.newaxis, :]

        return tf.cast(sampled_inds_in_roi, tf.int32)
    
    # def build_resnet(self, channel_multiplier, base_channels, resnet_input_shape, deep=False, name="resnet"):
    #     f10, f11, f12, f13, f14, f15 = [x * channel_multiplier for x in base_channels]
    #     f20, f21, f22, f23, f24, f25 = [x * 4 * channel_multiplier for x in base_channels]
    #     input = Input(shape=resnet_input_shape, name="rgb")
    #     # write all the code with layers
    #     x = ResInitial(filters=(f10, f20), name=f"{name}_initial")(input)
    #     if deep:
    #         x = ResIdentity(filters=(f10, f20), name=f"{name}_1_1")(x)
    #     x_1 = x
    #     x = ResConv(s=2, filters=(f11, f21), name=f"{name}_1_2")(x)
    #     if deep:
    #         x = ResIdentity(filters=(f11, f21), name=f"{name}_2_1")(x)
    #         x = ResIdentity(filters=(f11, f21), name=f"{name}_2_2")(x)
    #     x_2 = x
    #     x = ResConv(s=2, filters=(f12, f22), name=f"{name}_2_3")(x)
    #     # 3rd stage
    #     if deep:
    #         x = ResIdentity(filters=(f12, f22), name=f"{name}_3_1")(x)
    #         x = ResIdentity(filters=(f12, f22), name=f"{name}_3_2")(x)
    #     x = ResIdentity(filters=(f12, f22), name=f"{name}_3_3")(x)
    #     x_3 = x
    #     x = ResConv(s=2, filters=(f13, f23), name=f"{name}_3_4")(x)
    #     # 4th stage
    #     if deep:
    #         x = ResIdentity(filters=(f13, f23), name=f"{name}_4_1")(x)
    #         x = ResIdentity(filters=(f13, f23), name=f"{name}_4_2")(x)
    #         x = ResIdentity(filters=(f13, f23), name=f"{name}_4_3")(x)
    #     x = ResIdentity(filters=(f13, f23), name=f"{name}_4_4")(x)
    #     x_4 = x
    #     x = ResConv(s=2, filters=(f14, f24), name=f"{name}_4_5")(x)
    #     # 5th stage
    #     if deep:
    #         x = ResIdentity(filters=(f14, f24), name=f"{name}_5_1")(x)
    #         x = ResIdentity(filters=(f14, f24), name=f"{name}_5_2")(x)
    #         x = ResIdentity(filters=(f14, f24), name=f"{name}_5_3")(x)
    #         x = ResIdentity(filters=(f14, f24), name=f"{name}_5_4")(x)
    #     x = ResIdentity(filters=(f14, f24), name=f"{name}_5_5")(x)
    #     x_5 = x

    #     output = [x_1, x_2, x_3, x_4, x_5]
    #     model = keras.Model(inputs=input, outputs=output, name=f"{name}_model")

    #     return model

