import tensorflow as tf
import focal_loss
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2



def l1_loss_kp_cp(kp_cp_pre, kp_cp_targ):
    diffs = tf.subtract(kp_cp_pre, kp_cp_targ)
    abs_diff = tf.math.abs(diffs)
    l1_loss_kp_cp = tf.reduce_mean(abs_diff)
    return l1_loss_kp_cp

def l1_loss(pred_ofst, targ_ofst, mask_labels):
    """
    :param: pred_ofsts: [bs, n_pts, n_kpts, c] or [bs, n_pts, n_cpts, c]
            targ_ofst: [bs, n_pts, n_kpts, c] for kp,  [bs, n_pts, n_cpts, c] for cp
            mask_labels: [bs, n_pts]
    """
    bs, n_pts, n_kpts, c = pred_ofst.shape
    num_nonzero = tf.cast(tf.math.count_nonzero(mask_labels), tf.float32)

    w = tf.cast(mask_labels, dtype=tf.float32)
    w = tf.reshape(w, shape=[bs, n_pts, 1, 1])
    w = tf.repeat(w, repeats=n_kpts, axis=2)

    diff = tf.subtract(pred_ofst, targ_ofst)
    abs_diff = tf.multiply(tf.math.abs(diff), w)
    in_loss = abs_diff
    l1_loss = tf.reduce_sum(in_loss) / (num_nonzero + 1e-3)

    return l1_loss



class StereoPvn3dLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        resnet_input_shape,
        mse_loss_discount,
        l1_loss_discount,
        ssim_loss_discount,
        mlse_loss_discount,
        seg_loss_discount,
        ssim_max_val,
        ssim_filter_size,
        ssim_k1,
        ssim_k2,
        kp_loss_discount,
        cp_loss_discount,
        sm_loss_discount,
        kp_cp_loss_discount,
        kp_cp_ofst_loss_discount,
        distribute_training=False,

        **kwargs
    ):
        super().__init__()
        self.resnet_input_shape = resnet_input_shape
        self.mse_loss_discount = mse_loss_discount
        self.l1_loss_discount = l1_loss_discount
        self.ssim_loss_discount = ssim_loss_discount
        self.mlse_loss_discount = mlse_loss_discount
        self.seg_loss_discount = seg_loss_discount
        self.ssim_max_val = ssim_max_val
        self.ssim_filter_size = ssim_filter_size
        self.ssim_k1 = ssim_k1
        self.ssim_k2 = ssim_k2
        # for pvn3d
        self.kp_loss_discount = kp_loss_discount
        self.cp_loss_discount = cp_loss_discount
        self.sm_loss_discount = sm_loss_discount
        self.kp_cp_loss_discount = kp_cp_loss_discount
        self.kp_cp_ofst_loss_discount = kp_cp_ofst_loss_discount
        self.distribute_training = distribute_training
        self.seg_from_logits = True
        self.reduction = tf.keras.losses.Reduction.NONE if self.distribute_training else tf.keras.losses.Reduction.AUTO
        self.BinaryFocalLoss = focal_loss.BinaryFocalLoss(gamma=2, from_logits=True)

        red = tf.keras.losses.Reduction.NONE
        self.i =0
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mlse = tf.keras.losses.MeanSquaredLogarithmicError()
        #self.l1 = tf.keras.losses.Huber(reduction=red)
        # self.seg_loss = focal_loss.SparseCategoricalFocalLoss(
        #     2, from_logits=True, reduction=red
        # )
        self.seg_loss = focal_loss.SparseCategoricalFocalLoss(
            2, from_logits=True
        )

        self.sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        self.sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)

        self.sobel_x = tf.expand_dims(tf.expand_dims(self.sobel_x, axis=2), axis=3)
        self.sobel_y = tf.expand_dims(tf.expand_dims(self.sobel_y, axis=2), axis=3)

    def pcld_processor_tf_by_index(self, b_depth_pred, b_camera_matrix, b_sampled_index):
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


    def l1_loss_kp_cp(self, kp_cp_pre, kp_cp_targ):
        diffs = tf.subtract(kp_cp_pre, kp_cp_targ)
        abs_diff = tf.math.abs(diffs)
        l1_loss_kp_cp = tf.reduce_mean(abs_diff)
        return l1_loss_kp_cp

    def l1_loss(self, pred_ofst, targ_ofst, mask_labels):
        """
        :param: pred_ofsts: [bs, n_pts, n_kpts, c] or [bs, n_pts, n_cpts, c]
                targ_ofst: [bs, n_pts, n_kpts, c] for kp,  [bs, n_pts, n_cpts, c] for cp
                mask_labels: [bs, n_pts]
        """
        # bs, n_pts, n_kpts, c = pred_ofst.shape
        # num_nonzero = tf.cast(tf.math.count_nonzero(mask_labels), tf.float32)
        # w = tf.cast(mask_labels, dtype=tf.float32)
        # w = tf.reshape(w, shape=[bs, n_pts, 1, 1])
        # w = tf.repeat(w, repeats=n_kpts, axis=2)

        # diff = tf.subtract(pred_ofst, targ_ofst)
        # abs_diff = tf.multiply(tf.math.abs(diff), w)
        # in_loss = abs_diff
        # l1_loss =tf.reduce_sum(in_loss) / (num_nonzero + 1e-3)
        print(f'In Loss l1 call kp = pred[1] {tf.math.reduce_mean(pred_ofst).numpy()}')
        print(f'In Loss l1 call kp = targ[1] {tf.math.reduce_mean(targ_ofst).numpy()}')
        diff = tf.math.abs(pred_ofst - targ_ofst)  # (b, n_pts, n_kpts, 3)
        print(f'In Loss l1 call diff {tf.math.reduce_mean(diff).numpy()}')

        mask = tf.cast(mask_labels[:, :, tf.newaxis, :], tf.float32)  # (b, n_pts, 1, 3)
        print(f'In Loss l1 call mask {tf.math.reduce_mean(mask).numpy()}')

        masked_diff = diff * mask
        print(f'In Loss l1 call masked_diff {tf.math.reduce_mean(masked_diff).numpy()}')

        num_on_object = tf.math.reduce_sum(tf.cast(mask_labels, tf.float32))
        print(f'num_on_object {num_on_object}')
        loss = tf.math.reduce_mean(masked_diff)/(tf.math.reduce_mean(mask))# tf.reduce_sum(masked_diff) / (1e-5 + num_on_object)
        print(f'loss {loss}')

        return loss



    def loss_pvn_ktps(self, kp_cp_ofst_pre, kp_cp_ofst_target, kp_cp_pre, kp_cp_target, seg_pre, label, mask_label):
        """
        :param kp_cp_ofst_pre:
        :param kp_cp_ofst_target:
        :param kp_cp_pre:
        :param kp_cp_target:
        :param seg_pre: One-shot encoding for pixel-wise semantics
        :param label:
        :param mask_label:
        :param target_cls:
        :return:
        """
        loss_kp_cp_ofst = self.l1_loss(pred_ofst=kp_cp_ofst_pre, targ_ofst=kp_cp_ofst_target, mask_labels=mask_label)
        seg_pre = tf.unstack(seg_pre, axis=2)[1]
        label = tf.argmax(label, axis=2)
        loss_seg = self.BinaryFocalLoss(label, seg_pre)  # return batch-wise loss
        loss_kp_cp = self.l1_loss_kp_cp(kp_cp_pre, kp_cp_target)

        loss = self.params.kp_cp_ofst_loss_discount * loss_kp_cp_ofst + \
                self.params.sm_loss_discount * loss_seg + \
                self.params.kp_cp_loss_discount * loss_kp_cp

        return loss, loss_kp_cp_ofst, loss_seg, loss_kp_cp


    def ssim_loss(self, y_true, y_pred):        
        # C1 = (0.01 * max_val) ** 2
        # C2 = (0.03 * max_val) ** 2
        ssim = tf.image.ssim(y_true, y_pred, self.ssim_max_val, filter_size=self.ssim_filter_size, filter_sigma=1.5, k1=self.ssim_k1, k2=self.ssim_k2, return_index_map=False)
        #print(f"SSIM before mean: {ssim}")
        ssim = tf.reduce_mean(ssim, axis=-1)
        #print(f"SSIM after mean: {ssim}")

        # Compute SSIM loss (SSIM is a similiarity, so the 1 - SSIM is the loss)
        return 1 - ssim
    
    @staticmethod
    def get_offst(
        RT,  # [b, 4,4]
        pcld_xyz,  # [b, n_pts, 3]
        mask_selected,  # [b, n_pts, 1] 0|1
        kpts_cpts,  # [b, 9,3]
    ):
        """Given a pointcloud, keypoints in a local coordinate frame and a transformation matrix,
        this function calculates the offset for each point in the pointcloud to each
        of the keypoints, if the are transformed by the transformation matrix.
        Additonally a binary segmentation mask is used to set the offsets to 0 for points,
        that are not part of the object.
        The last keypoint is treated as the center point of the object.


        Args:
            RT (b,4,4): Homogeneous transformation matrix
            pcld_xyz (b,n_pts,3): Pointcloud in camera frame
            mask_selected (b,n_pts,1): Mask of selected points (0|1)
            kpts_cpts (b,n_kpts,3): Keypoints in local coordinate frame

        Returns:
            kp_offsets: (b,n_pts,n_kpts,3) Offsets to the keypoints
            cp_offsets: (b,n_pts,1,3) Offsets to the center point
        """

        # WHICH ONE IS CORRECT!?????
        # transform kpts_cpts to camera frame using rt
        kpts_cpts_cam = (
            tf.matmul(kpts_cpts, tf.transpose(RT[:, :3, :3], (0, 2, 1))) + RT[:, tf.newaxis, :3, 3]
        )

        # calculate offsets to the pointcloud
        kpts_cpts_cam = tf.expand_dims(kpts_cpts_cam, axis=1)  # [b, 1, 9, 3] # add num_pts dim
        pcld_xyz = tf.expand_dims(pcld_xyz, axis=2)  # [b, n_pts, 1, 3] # add_kpts dim
        offsets = tf.subtract(kpts_cpts_cam, pcld_xyz)  # [b, n_pts, 9, 3]
        # mask offsets to the object points

        # print("offsets shape:", offsets.shape)
        # print("mask_selected shape:", mask_selected.shape)

        offsets = offsets * tf.cast(mask_selected[:, :, tf.newaxis], tf.float32)
        # offsets = tf.where(mask_selected[:, :, tf.newaxis] == 1, offsets, 0.0)
        kp_offsets = offsets[:, :, :-1, :]  # [b, n_pts, 8, 3]
        cp_offsets = offsets[:, :, -1:, :]  # [b, n_pts, 1, 3]
        return kp_offsets, cp_offsets

    def call(self, y_true_list, y_pred_list):
        # Get data
        [gt_depth, gt_RT, gt_mask, gt_disp] = y_true_list[0], y_true_list[1], y_true_list[2], y_true_list[3]
        gt_mask = tf.expand_dims(gt_mask, axis=-1) # better to move this to simpose.py
        
        # depth, kp, sm, cp, xyz_pred, sampled_inds_in_original_image, mesh_kpts, norm_bbox, cropped_rgbs_l, cropped_rgbs_r, w, attention, normalized_magnitude
        [pred_depth, pred_kp, pred_sm, pred_cp] = y_pred_list[0], y_pred_list[1], y_pred_list[2], y_pred_list[3]
        print(f'In loss call pred_depth = pred[2] {tf.math.reduce_mean(pred_depth).numpy()}')
        print(f'In loss call pred_sm = pred[2] {tf.math.reduce_mean(pred_sm).numpy()}')
        print(f'In loss call pred_cp = pred[3] {tf.math.reduce_mean(pred_cp).numpy()}')
        print(f'In Loss call kp = pred[1] {tf.math.reduce_mean(pred_kp).numpy()}')
        xyz_pred, sampled_inds, mesh_kpts, norm_bbox = y_pred_list[4], y_pred_list[5], y_pred_list[6],  y_pred_list[7]
        print(f'In Loss call xyz_pred {tf.math.reduce_mean(xyz_pred).numpy()}')
        print(f'In Loss call sampled_inds {tf.math.reduce_mean(sampled_inds).numpy()}')
        w = y_pred_list[10]
        intrinsics = y_pred_list[12]
        crop_factor = y_pred_list[13]
        before_head = y_pred_list[14]
        print(f'In Loss call before_head {tf.math.reduce_mean(before_head).numpy()}')
        # pred_derivative = y_pred_list[12]
        xyz_gt = self.pcld_processor_tf_by_index(gt_depth+0.0001, intrinsics, sampled_inds)



        # print(f"gt_depth in the loss.call shape: {gt_depth.shape} - dtype: {gt_depth.dtype} - gt_depth min: {tf.math.reduce_min(gt_depth)} - gt_depth max: {tf.math.reduce_max(gt_depth)}")
        # print(f"pred_depth shape: {pred_depth.shape} - dtype: {pred_depth.dtype} - pred_depth min: {tf.math.reduce_min(pred_depth)} - pred_depth max: {tf.math.reduce_max(pred_depth)} ")  

        # Processing data to compare gt and pred
        mask_selected = tf.gather_nd(gt_mask, sampled_inds)
        gt_kp, gt_cp = self.get_offst(
            gt_RT,
            xyz_gt,
            mask_selected,
            mesh_kpts
        )

        gt_depth = gt_depth[..., :1]
        gt_disp = gt_disp[..., :1]
        pred_depth = pred_depth[..., :1]
        gt_depth = tf.image.crop_and_resize(
                        tf.cast(gt_depth, tf.float32),
                        norm_bbox,
                        tf.range(tf.shape(gt_depth)[0]),
                        self.resnet_input_shape[:2]
                    )
        gt_disp = tf.image.crop_and_resize(
                        tf.cast(gt_disp, tf.float32),
                        norm_bbox,
                        tf.range(tf.shape(gt_disp)[0]),
                        self.resnet_input_shape[:2]
                    )


        gt_disp = gt_disp / tf.cast(crop_factor, tf.float32)
        gt_disp = gt_disp[..., :1]
        # pred_depth_masked = pred_depth * tf.cast(gt_mask, tf.float32)
        pred_depth_masked = tf.where(gt_depth>0.01, pred_depth, 0.0)
        # print(f"shape gt_depth: {gt_depth.shape} - dtype: {gt_depth.dtype}")
        # print(f"shape pred_depth: {pred_depth.shape} - dtype: {pred_depth.dtype}")
        # print(f"shape pred_depth_masked: {pred_depth_masked.shape} - dtype: {pred_depth_masked.dtype}")
        # print(f"shape gt_kp: {gt_kp.shape} - dtype: {gt_kp.dtype}")
        # print(f"shape pred_kp: {pred_kp.shape} - dtype: {pred_kp.dtype}")
        # print(f"shape gt_cp: {gt_cp.shape} - dtype: {gt_cp.dtype}")
        # print(f"shape pred_cp: {pred_cp.shape} - dtype: {pred_cp.dtype}")
        
        # grad_x_loss = self.mlse(sobel_x_pred, sobel_x_gt)
        # gradx_discount = 0.5

        # Apply Sobel filter in the y-direction using convolution
        sobel_x_der = tf.nn.conv2d(gt_disp, self.sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        sobel_y_der = tf.nn.conv2d(gt_disp, self.sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        sobel_xx_der = tf.nn.conv2d(sobel_x_der, self.sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        sobel_yy_der = tf.nn.conv2d(sobel_y_der, self.sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        sobel_xy_der = tf.nn.conv2d(sobel_x_der, self.sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        sobel_yx_der = tf.nn.conv2d(sobel_y_der, self.sobel_x, strides=[1, 1, 1, 1], padding='SAME')

        # Compute the gradient magnitude (norm)
        gradient_magnitude = tf.sqrt(tf.square(sobel_x_der) + tf.square(sobel_y_der))
        grad_mask = tf.cast(tf.where(gradient_magnitude < 0.1, 1, 0), tf.float32) 


        # print(f'grad_mask {grad_mask.dtype} - {tf.shape(grad_mask)} ')
        # print(f'sobel_xx_der {sobel_xx_der.dtype} - {tf.shape(sobel_xx_der)} - sobel_yy_der {sobel_yy_der.dtype} - {tf.shape(sobel_yy_der)} - sobel_yx_der {sobel_yx_der.dtype} - {tf.shape(sobel_yx_der)}')
        # print(f'tf.multiply(grad_mask, sobel_xx_der) {tf.multiply(grad_mask, sobel_xx_der).dtype} - {tf.shape(tf.multiply(grad_mask, sobel_xx_der))} - tf.multiply(grad_mask, sobel_xy_der) {tf.multiply(grad_mask, sobel_xy_der).dtype} - {tf.shape(tf.multiply(grad_mask, sobel_xy_der))} - sobel_yx_der {sobel_yx_der.dtype} - {tf.shape(sobel_yx_der)}')
        # print(f'tf.multiply(grad_mask, sobel_yx_der) {tf.multiply(grad_mask, sobel_yx_der).dtype} - {tf.shape(tf.multiply(grad_mask, sobel_yx_der))} - tf.multiply(grad_mask, sobel_xy_der) {tf.multiply(grad_mask, sobel_xy_der).dtype} - {tf.shape(tf.multiply(grad_mask, sobel_xy_der))} -  tf.multiply(grad_mask, sobel_yy_der) { tf.multiply(grad_mask, sobel_yy_der).dtype} - {tf.shape( tf.multiply(grad_mask, sobel_yy_der))}')
        # print(f'tf.concat([tf.multiply(grad_mask, sobel_xx_der), tf.multiply(grad_mask, sobel_xy_der)], axis = 2) {tf.shape(tf.concat([tf.multiply(grad_mask, sobel_xx_der), tf.multiply(grad_mask, sobel_xy_der)], axis = 2))} ')


        hessian_m = tf.concat([tf.concat([tf.multiply(grad_mask, sobel_xx_der), tf.multiply(grad_mask, sobel_xy_der)], axis = 2), 
                               tf.concat([tf.multiply(grad_mask, sobel_yx_der), tf.multiply(grad_mask, sobel_yy_der)], axis = 2)], axis = 1)


        # Apply Sobel filter in the y-direction using convolution
        sobel_x_der_pred = tf.nn.conv2d(pred_depth_masked, self.sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        sobel_y_der_pred = tf.nn.conv2d(pred_depth_masked, self.sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        sobel_xx_der_pred = tf.nn.conv2d(sobel_x_der_pred, self.sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        sobel_yy_der_pred = tf.nn.conv2d(sobel_y_der_pred, self.sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        sobel_xy_der_pred = tf.nn.conv2d(sobel_x_der_pred, self.sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        sobel_yx_der_pred = tf.nn.conv2d(sobel_y_der_pred, self.sobel_x, strides=[1, 1, 1, 1], padding='SAME')



        hessian_m_pred =tf.concat([tf.concat([sobel_xx_der_pred, sobel_xy_der_pred], axis = 2), tf.concat([sobel_yx_der_pred, sobel_yy_der_pred], axis = 2)], axis = 1)






        

        # Normalize the gradient magnitude (optional)
        gt_derivative = tf.cast(255.0 * gradient_magnitude / tf.reduce_max(gradient_magnitude), tf.float32)


        deriv_loss = self.mse(grad_mask * sobel_x_der, grad_mask * sobel_x_der_pred) + self.mse(grad_mask * sobel_y_der, grad_mask * sobel_y_der_pred)
        hessian_loss = self.mse(hessian_m, hessian_m_pred)

        #l1_loss = self.l1(gt_depth, pred_depth)
        mse_loss = self.mse(gt_disp, pred_depth_masked)
        # mlse_loss = self.mlse(gt_depth, pred_depth)

        # loss = square(log(y_true + 1.) - log(y_pred + 1.))

        mlse_loss = self.mlse(gt_disp, pred_depth_masked) # tf.math.reduce_sum(tf.math.abs(gt_depth - pred_depth_masked)) / tf.cast(tf.math.count_nonzero(gt_depth), tf.float32) #tf.coun_non_zero

        ssim_loss_value = self.ssim_loss(gt_disp, pred_depth_masked)

        xyz_loss = self.mse(xyz_pred, xyz_gt)
        

        # pvn3d loss functions
        bs, _, _, _ = pred_kp.shape
        print(f'In Loss call 2 kp = pred[1] {tf.math.reduce_mean(pred_kp).numpy()}')
        loss_kp = self.l1_loss(pred_ofst=pred_kp,
                               targ_ofst=gt_kp,
                               mask_labels=mask_selected)
        # if binary_loss is True:

        #     if not self.params.seg_from_logits:
        #         seg_pre = tf.nn.softmax(seg_pre)

        #     seg_pre = tf.unstack(seg_pre, axis=2)[1]
        #     label = tf.argmax(label, axis=2)
        #     loss_seg = self.BinaryFocalLoss(label, seg_pre)  # return batch-wise value
        # else:
        # print(f"seg: mask_label.shape: {mask_label.shape} - pred_sm: {pred_sm.shape}")
        # print(f"seg: mask_label[:,:,0].shape: {mask_label[:,:,0].shape} - pred_sm[:,:,0]: {pred_sm[:,:,0].shape}")
        # print(f"seg: mask_label[:,:,0].dtype: {mask_label[:,:,0].dtype} - pred_sm[:,:,0]: {pred_sm[:,:,0].dtype}")
        # mask_label = tf.cast(mask_label, dtype=tf.float32)
        # pred_sm = tf.cast(pred_sm, dtype=tf.float32)


        loss_seg = self.BinaryFocalLoss(mask_selected, pred_sm)  # labels [bs, n_pts, n_cls] this is from logits
        # change in something similiar: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=self.reduction)


        loss_cp = self.l1_loss(pred_ofst=pred_cp,
                               targ_ofst=gt_cp,
                              mask_labels=mask_selected)

        # if self.params.distribute_training:
        #     loss_seg = tf.reduce_sum(loss_seg) * (1 / bs)


        '''Print all losses for debugging:'''
        print("-------------------\nLoss PVN3D")

        print('loss_kp', loss_kp)
        print('loss_seg', loss_seg)
        print('loss_cp', loss_cp)
        print('loss_ssim', ssim_loss_value)
        print('loss_mse', mse_loss)
        print('loss_mlse', mlse_loss)
        print('loss_deriv', deriv_loss)
        print('loss_hessian', hessian_loss)
        print('loss_xyz', xyz_loss)

        #print('loss_l1_dpt', l1_loss)        


        loss_cp_scaled = self.cp_loss_discount * loss_cp
        loss_kp_scaled = self.kp_loss_discount * loss_kp
        loss_seg_scaled = self.sm_loss_discount * loss_seg
        mse_loss_scaled = self.mse_loss_discount * mse_loss
        mlse_loss_scaled = self.mlse_loss_discount * mlse_loss
        ssim_loss_scaled = self.ssim_loss_discount * ssim_loss_value
        deriv_loss_scaled = 0.0005 * deriv_loss
        hessian_loss_scaled = 0.0005 * hessian_loss
        xyz_loss_scaled = 1 * xyz_loss
        

        loss = (
            mse_loss_scaled
            + mlse_loss_scaled
            + ssim_loss_scaled
            + deriv_loss_scaled
            # + loss_cp_scaled # from pvn3d
            # + loss_kp_scaled # from pvn3d
            # + loss_seg_scaled # from pvn3d
            + hessian_loss_scaled
            # + xyz_loss_scaled
        )

        return (
            loss,
            mse_loss_scaled,
            mlse_loss_scaled,
            ssim_loss_scaled,
            deriv_loss_scaled,
            loss_cp_scaled,
            loss_kp_scaled,
            loss_seg_scaled,
            hessian_loss_scaled,
            xyz_loss_scaled,
        )



        return loss #, logs # ssim_loss_value # , loss_cp, loss_kp, loss_seg
