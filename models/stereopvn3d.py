import tensorflow.keras as keras
from tensorflow.keras import activations
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    MultiHeadAttention,
    UpSampling2D,
    Conv1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import Input

# from lib.net.utils import match_choose
import tensorflow as tf

from models.densefusion import DenseFusionNet, DenseFusionNetParams
from .pprocessnet import _InitialPoseModel
from metrics.stereo_pvn3d_metrics import StereoPvn3dMetrics

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from .stereo_layers import (
    ResInitial,
    ResUp,
    ResConv,
    ResReduce,
    ResIdentity,
    StereoAttention,
    DisparityAttention,
    ContextAdj,
)

from losses.stereopvn3d_loss import StereoPvn3dLoss

from .mlp import MlpNets, MlpNetsParams

from .utils import match_choose_adp, dpt_2_cld, dpt_2_cld_tf, match_choose, pcld_processor_tf


class StereoPvn3d(keras.Model):
    def __init__(
        self,
        *,
        resnet_input_shape,
        use_disparity,
        relative_disparity,
        channel_multiplier,
        base_channels,
        num_pts,
        num_kpts=8, 
        num_cls=1, 
        num_cpts=1, 
        dim_xyz=3,
        context_adj=False,
        **kwargs,
    ):
        super(StereoPvn3d, self).__init__()
        self.use_disparity = use_disparity
        self.relative_disparity = relative_disparity
        self.context_adjustment = context_adj
        self.channel_multiplier = channel_multiplier
        self.num_pts = num_pts
        self.num_kpts = num_kpts
        self.num_cls = num_cls
        self.num_cpts = num_cpts
        self.dim_xyz = dim_xyz
        self.resnet_input_shape = resnet_input_shape
        self.mlp_params = MlpNetsParams()
        self.custom_metric = StereoPvn3dMetrics()
        self.initial_pose_model = _InitialPoseModel()
        self.segmentation_metric = tf.keras.metrics.CategoricalCrossentropy(from_logits=True)
        self.base_channels = base_channels
        self.resnet_lr = self.build_resnet(self.channel_multiplier, self.base_channels, self.resnet_input_shape)

        _, _, f12, f13, f14, f15 = [
            x * self.channel_multiplier * 2 for x in self.base_channels
        ]

        _, _, f22, f23, f24, f25 = [
            x * 4 * self.channel_multiplier * 2 for x in self.base_channels
        ]
        #self.attention1 = StereoAttention(channels=8, width=4, depth=f25)
        self.attention1 = DisparityAttention(max_disparity=6)
        self.dense_fusion_params = DenseFusionNetParams()
        # self.dense_fusion_net = DenseFusionNet(self.dense_fusion_params)
        # self.dense_fusion_model = self.dense_fusion_net.build_dense_fusion_model(
            # rgb_emb_shape=(num_pts, self.dense_fusion_params.num_embeddings),
            # pcl_emb_shape=(num_pts, 3))#self.dense_fusion_params.num_embeddings))
        
        self.resid1 = ResIdentity(filters=(f14, f24), name="d5_1")
        self.resid2 = ResIdentity(filters=(f14, f24), name="d5_2")
        self.resid3 = ResIdentity(filters=(f14, f24), name="d5_3")
        #self.resid4 = ResIdentity(filters=(f14, f24), name="d5_4")
        self.resup1 = ResUp(s=2, filters=(f13, f23), name="d5_5")
        # self.attention2 = StereoAttention(channels=4, width=4, dilation=3)
        self.attention2 = DisparityAttention(max_disparity=3)
        self.resred1 = ResReduce(f24)
        self.resid4 = ResIdentity(filters=(f14, f24), name="d4_1")
        self.resid5 = ResIdentity(filters=(f14, f24), name="d4_2")
        self.resid6 = ResIdentity(filters=(f14, f24), name="d4_3")
        #self.resid8 = ResIdentity(filters=(f14, f24), name="d4_4")
        self.resup2 = ResUp(s=2, filters=(f13, f23), name="d4_5")
        # self.attention3 = StereoAttention(channels=4, width=8, dilation=2)
        self.attention3 = DisparityAttention(max_disparity=3)
        self.resred2 = ResReduce(f24)
        self.resid7 = ResIdentity(filters=(f14, f24), name="d3_3")
        self.resid8 = ResIdentity(filters=(f14, f24), name="d3_4")
        self.resid9 = ResIdentity(filters=(f14, f24), name="d3_4")
        self.resup3 = ResUp(s=2, filters=(f13, f23), name="d3_4")
        # self.attention4 = StereoAttention(channels=4, width=16, dilation=2)
        self.attention4 = DisparityAttention(max_disparity=3)
        self.resred3 = ResReduce(f24)
        self.resid10 = ResIdentity(filters=(f14, f24), name="d2_3")
        self.resid11 = ResIdentity(filters=(f14, f24), name="d2_3")
        self.resid12 = ResIdentity(filters=(f14, f24), name="d2_4")
        
        self.resup4 = ResUp(s=2, filters=(f13, f23), name="d2_4")
        # self.attention5 = StereoAttention(channels=4, width=32, dilation=2)
        self.attention5 = DisparityAttention(max_disparity=3)
        self.resred4 = ResReduce(f24)
        self.resid13 = ResIdentity(filters=(f14, f24), name="d1_2")
        self.resid14 = ResIdentity(filters=(f14, f24), name="d1_3")
        self.resid15 = ResIdentity(filters=(f14, f24), name="d1_3")

        self.head1 = tf.keras.layers.Conv2D(
            16,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="deph_conv_1",
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        n_cls = 6  # + background + depth
        #self.n_rgbd_feats = 
        

        self.n_rgbd_mlp_feats = self.dense_fusion_params.num_embeddings + self.dense_fusion_params.rgbd_feats2_conv1d_dim + self.dense_fusion_params.rgb_conv1d_dim + self.dense_fusion_params.pcl_conv1d_dim + 3

        self.dense_fusion_layer = Conv1D(filters=self.n_rgbd_mlp_feats, kernel_size=1, activation='relu') # unused

        self.head2 = tf.keras.layers.Conv2D(
            self.n_rgbd_mlp_feats,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="deph_conv_2",
        )
        self.cal1 = ContextAdj(filter=32)
        self.cal2 = ContextAdj(filter=64)


        # MLP model 
        self.mlp_net = MlpNets(self.mlp_params,
                               num_pts= self.num_pts,
                               num_kpts= self.num_kpts,
                               num_cls= self.num_cls,
                               num_cpts= self.num_cpts,
                               channel_xyz= self.dim_xyz)

        self.mlp_model = self.mlp_net.build_mlp_model(rgbd_features_shape=(self.num_pts, self.n_rgbd_mlp_feats))
        [H, W, _] = resnet_input_shape


    #@tf.function
    def call(self, inputs, training=None):
        full_rgb_l = inputs[0]
        full_rgb_r = inputs[1]  # another layer        
        h, w = tf.shape(full_rgb_l)[1], tf.shape(full_rgb_r)[2]

        baseline = inputs[2][0]
        K = inputs[3][0]
        # print('K', K)
        focal_length = K[0,0]
        # print('focal_length', focal_length)
        intrinsics = inputs[3]
        sampled_inds_in_roi = inputs[4]
        depth = inputs[5]
        bbox = inputs[6]
        mesh_kpts = inputs[7]

        norm_bbox = tf.cast(bbox / [h, w, h, w], tf.float32)
        cropped_rgbs_l = tf.image.crop_and_resize(
            tf.cast(full_rgb_l, tf.float32),
            norm_bbox,
            tf.range(tf.shape(full_rgb_l)[0]),
            self.resnet_input_shape[:2],
        )

        cropped_rgbs_r = tf.image.crop_and_resize(
            tf.cast(full_rgb_r, tf.float32),
            norm_bbox,
            tf.range(tf.shape(full_rgb_r)[0]),
            self.resnet_input_shape[:2],
        )

        cropped_depth = tf.image.crop_and_resize(
            tf.cast(depth, tf.float32),
            norm_bbox,
            tf.range(tf.shape(depth)[0]),
            self.resnet_input_shape[:2],
        )

        # stop gradients for preprocessing
        # cropped_rgbs_l = tf.stop_gradient(cropped_rgbs_l)
        # cropped_rgbs_r = tf.stop_gradient(cropped_rgbs_r)
        # cropped_depth = tf.stop_gradient(cropped_depth)
        # norm_bbox = tf.stop_gradient(norm_bbox)

        #sampled_inds_in_roi = tf.stop_gradient(sampled_inds_in_roi)



        # StereoNet ouputs is of dimensions HxWx1032
        f_l_1, f_l_2, f_l_3, f_l_4, f_l_5 = self.resnet_lr(cropped_rgbs_l)
        f_r_1, f_r_2, f_r_3, f_r_4, f_r_5 = self.resnet_lr(cropped_rgbs_r)

        deep = True
        # x, w, w_1 = self.attention1(f_l_5, f_r_5)
        x, w = self.attention1([f_l_5, f_r_5])
        print(f'x after the 1st layer of attention: {x.shape}')
        if deep:
            x = self.resid1(x)
            x = self.resid2(x)
            x = self.resid3(x)
        #x = self.resid4(x)
        x = (x)

        x = self.resup1(x)

        x_skip = x

        x, w = self.attention2([f_l_4,f_r_4],w)
        print(f'x after the 2st layer of attention: {x.shape}')
        x = tf.concat([x, x_skip], axis=-1) #cut the concat and put it in the disparity layer? non avrebbe senso. Forse bisogna cambiare
        

        x = self.resred1(x)
        print(f'x after the resred1: {x.shape}')
        if deep:
            x = self.resid4(x)
            x = self.resid5(x)
            x = self.resid6(x)
        #x = self.resid8(x)
        x = self.resup2(x)

        x_skip = x

        x, w = self.attention3(
            [f_l_3,
            f_r_3],
            previous_weights=w
        )
        print(f'x after the 3st layer of attention: {x.shape}')
        x = tf.concat([x, x_skip], axis=-1)
        print(f'x after the concat: {x.shape}')
   
        x = self.resred2(x)

        # if deep:
        #     x = ResIdentity(filters=(f23, f23), name="d3_1")(x)
        #     x = ResIdentity(filters=(f23, f23), name="d3_2")(x)
        x = self.resid7(x)
        x = self.resid8(x)
        x = self.resid9(x)
        x = self.resup3(x)

        x_skip = x

        x, w= self.attention4(
            [f_l_2, f_r_2], previous_weights=w,
        )
        print(f'x after the 4st layer of attention: {x.shape}')
        x = tf.concat([x, x_skip], axis=-1)
        print(f'x after the concat: {x.shape}')

        x = self.resred3(x)

        # if deep:
        #     x = ResIdentity(filters=(f23, f23), name="d2_1")(x)
        #     x = ResIdentity(filters=(f23, f23), name="d2_2")(x)
        x = self.resid10(x)
        x = self.resid11(x)
        x = self.resid12(x)
        x = self.resup4(x)

        x_skip = x

        x, w = self.attention5(
            [f_l_1,
            f_r_1],
            previous_weights=w,
        )
        print(f'x after the 5st layer of attention: {x.shape}')
        x = tf.concat([x, x_skip], axis=-1)
        print(f'x after the concat: {x.shape}')

        x = self.resred4(x)

        # if deep:
        #     x = ResIdentity(filters=(f23, f23), name="d1_1")(x)
        x = self.resid13(x)
        x = self.resid14(x)
        x = self.resid15(x)
        # before_head = x

        

        
        x = self.head1(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        stereo_outputs = self.head2(x)
        #print(f"stereo_outputs.shape: {stereo_outputs.shape} - stereo_outputs: {stereo_outputs}")

        # if debug:
        #    if self.context_adjustment:
        #        adj_outputs = self.cal1(tf.cast(x_l, dtype=tf.float32), outputs)
        #        adj_outputs = self.cal2(adj_outputs, outputs)
        #        return adj_outputs, w_1, w_2, w_3, w_4, w_5, before_head, f_l_5, f_r_5, f_l_4, f_r_4
        #    else:
        #        return outputs, w_1, w_2, w_3, w_4, w_5, before_head, f_l_5, f_r_5, f_l_4, f_r_4

        if self.context_adjustment:
            raise NotImplementedError
            adj_outputs = self.cal1(tf.cast(x_l, dtype=tf.float32), outputs)
            adj_outputs = self.cal2(adj_outputs, outputs)
            return adj_outputs
        else:
            if self.use_disparity:
                disp = tf.nn.relu(stereo_outputs[..., :1])
                # print('disp after stereo atention', disp)
                if self.relative_disparity:
                    disp = (
                        disp * tf.shape(inputs[0])[2]
                    )  # BHWC -> *W converts to absolute disparity
                y = baseline * focal_length
                #print(y)
                depth = tf.math.divide_no_nan(y, disp)
                #print('depth after math divide no nan', depth)
                depth = tf.clip_by_value(depth, 0.0, 50.0)
                #print('depth after clip by value', depth)
                #stereo_seg = tf.concat([depth, stereo_outputs[..., 1:7]], axis=-1) #H x W x (1 + n_cls)
                stereo_outputs = tf.concat([depth, stereo_outputs[..., 1:]], axis=-1, name="final_concat") #H x W x (n_features)
            else:
                depth = stereo_outputs[..., :1]
                
        #print(f"stereo_outputs.shape: {stereo_outputs.shape} - stereo_outputs: {stereo_outputs}")

        

        crop_factor = False

        # rgb_emb = match_choose_adp(self.n_rgbd_feats, sampled_index, crop_factor, stereo_outputs.shape)
        #print('Before match_choose ...')
        # rgb_emb = match_choose(self.n_rgbd_feats, sampled_index)

        #print(f"sampled_inds_in_roi.shape: {sampled_inds_in_roi.shape}")

        rgb_emb = tf.gather_nd(stereo_outputs, sampled_inds_in_roi)
        #rgb_emb = match_choose(stereo_outputs, sampled_inds_in_roi)

        #print(f"rgb_emb.shape: {rgb_emb.shape} - rgb_emb: {rgb_emb}")

        #print('rgb_emb after match_choose', rgb_emb)
        camera_scale = 1
        feats_fused = rgb_emb
        kp, sm, cp = self.mlp_model(feats_fused, training=training)
        if training:
            return (depth, kp, sm, cp, norm_bbox)
        else:
            xyz,  = pcld_processor_tf_batched_by_index(depth, intrinsics, self.num_pts, sampled_inds_in_roi)
            batch_R, batch_t, voted_kpts = self.initial_pose_model([xyz, kp, cp, sm, mesh_kpts])
            return (
                batch_R,
                batch_t,
                voted_kpts,
                (depth, kp, sm, cp, norm_bbox)
            )

    # @tf.function
    # def train_step(self, data):
    #     # Unpack the data. Its structure depends on your model and
    #     # on what you pass to `fit()`.
    #     x = data[0]
    #     y = data[1]
    #     batch_size = tf.cast(tf.shape(x[0])[0], tf.float32)
    #     with tf.GradientTape() as tape:
    #         y_pred = self(x, training=True)  # Forward pass
    #         # Compute the loss value
    #         # (the loss function is configured in `compile()`)
    #         loss= self.loss(y_true=y, y_pred=y_pred) # / batch_size

    #     # Compute gradients
    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     # Update weights
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    #     return {'loss': loss}#, 'ssim_loss': ssim_loss}

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


    #@tf.function
    def test_step(self, data):
        print(f"data: {data}")

        x = data[0]
        print(f"x: {x}")
        y = data[1]
        print(f"y: {y}")
        batch_size = tf.cast(tf.shape(x[0])[0], tf.float32)
        y_pred = self(x, training=False)

        gt_sm = y[1]
        pred_sm = y_pred[2]

        loss = self.loss(y, y_pred)
        loss = loss #/ batch_size
        self.custom_metric.update_state(y, y_pred)
        dictionary_metrics = self.custom_metric.result()
        
        mlse_metric = dictionary_metrics.get("depth_mlse")
        mse_metric = dictionary_metrics.get("depth_mse")
        mae_metric = dictionary_metrics.get("depth_mae")
        ssim_metric = dictionary_metrics.get("depth_ssim")
        kp_l1_metric = dictionary_metrics.get("kp_l1")
        cp_l1_metric = dictionary_metrics.get("cp_l1")
        segmentation_crossentropy = self.segmentation_metric(gt_sm, pred_sm)


        return {"loss": loss,
                'depth_mlse':mlse_metric,
                "depth_mse":mse_metric,
                "depth_mae":mae_metric,
                "depth_ssim":ssim_metric,
                    "kp_l1":kp_l1_metric,
                    "cp_l1":cp_l1_metric,
                    "segmentation_crossentropy":segmentation_crossentropy}


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
        normal_map = StereoPvn3d.compute_normal_map(depth, camera_matrix)  # [b, h,w,3]

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
    def get_crop_index(roi, in_h, in_w, resnet_h, resnet_w):
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

        y1, x1, y2, x2 = roi[:, 0], roi[:, 1], roi[:, 2], roi[:, 3]

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

        return tf.stack([y1_new, x1_new, y2_new, x2_new], axis=-1), crop_factor


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
    
    def build_resnet(self, channel_multiplier, base_channels, resnet_input_shape, deep=False, name="resnet"):
        f10, f11, f12, f13, f14, f15 = [x * channel_multiplier for x in base_channels]
        f20, f21, f22, f23, f24, f25 = [x * 4 * channel_multiplier for x in base_channels]
        input = Input(shape=resnet_input_shape, name="rgb")
        # write all the code with layers
        x = ResInitial(filters=(f10, f20), name=f"{name}_initial")(input)
        if deep:
            x = ResIdentity(filters=(f10, f20), name=f"{name}_1_1")(x)
        x_1 = x
        x = ResConv(s=2, filters=(f11, f21), name=f"{name}_1_2")(x)
        if deep:
            x = ResIdentity(filters=(f11, f21), name=f"{name}_2_1")(x)
            x = ResIdentity(filters=(f11, f21), name=f"{name}_2_2")(x)
        x_2 = x
        x = ResConv(s=2, filters=(f12, f22), name=f"{name}_2_3")(x)
        # 3rd stage
        if deep:
            x = ResIdentity(filters=(f12, f22), name=f"{name}_3_1")(x)
            x = ResIdentity(filters=(f12, f22), name=f"{name}_3_2")(x)
        x = ResIdentity(filters=(f12, f22), name=f"{name}_3_3")(x)
        x_3 = x
        x = ResConv(s=2, filters=(f13, f23), name=f"{name}_3_4")(x)
        # 4th stage
        if deep:
            x = ResIdentity(filters=(f13, f23), name=f"{name}_4_1")(x)
            x = ResIdentity(filters=(f13, f23), name=f"{name}_4_2")(x)
            x = ResIdentity(filters=(f13, f23), name=f"{name}_4_3")(x)
        x = ResIdentity(filters=(f13, f23), name=f"{name}_4_4")(x)
        x_4 = x
        x = ResConv(s=2, filters=(f14, f24), name=f"{name}_4_5")(x)
        # 5th stage
        if deep:
            x = ResIdentity(filters=(f14, f24), name=f"{name}_5_1")(x)
            x = ResIdentity(filters=(f14, f24), name=f"{name}_5_2")(x)
            x = ResIdentity(filters=(f14, f24), name=f"{name}_5_3")(x)
            x = ResIdentity(filters=(f14, f24), name=f"{name}_5_4")(x)
        x = ResIdentity(filters=(f14, f24), name=f"{name}_5_5")(x)
        x_5 = x

        output = [x_1, x_2, x_3, x_4, x_5]
        model = keras.Model(inputs=input, outputs=output, name="resnet_model")

        return model

