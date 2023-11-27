import tensorflow as tf
import focal_loss
from skimage.metrics import structural_similarity as ssim
from models.stereopvn3d import StereoPvn3d
from models.pprocessnet import _InitialPoseModel
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
        ssim_loss_discount,
        mlse_loss_discount,
        ssim_max_val,
        ssim_filter_size,
        ssim_k1,
        ssim_k2,
        kp_loss_discount,
        cp_loss_discount,
        sm_loss_discount,
        deriv_loss_discount,
        hessian_loss_discount,
        distribute_training=False,

        **kwargs
    ):
        super().__init__()
        self.resnet_input_shape = resnet_input_shape
        self.mse_loss_discount = mse_loss_discount
        self.ssim_loss_discount = ssim_loss_discount
        self.mlse_loss_discount = mlse_loss_discount
        self.ssim_max_val = ssim_max_val
        self.ssim_filter_size = ssim_filter_size
        self.ssim_k1 = ssim_k1
        self.ssim_k2 = ssim_k2
        # for pvn3d
        self.kp_loss_discount = kp_loss_discount
        self.cp_loss_discount = cp_loss_discount
        self.sm_loss_discount = sm_loss_discount
        self.deriv_loss_discount = deriv_loss_discount
        self.hessian_loss_discount = hessian_loss_discount
        self.distribute_training = distribute_training
        self.seg_from_logits = True
        self.reduction = tf.keras.losses.Reduction.NONE if self.distribute_training else tf.keras.losses.Reduction.AUTO
        self.BinaryFocalLoss = focal_loss.BinaryFocalLoss(gamma=2, from_logits=True)
        red = tf.keras.losses.Reduction.NONE
        self.i =0
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mlse = tf.keras.losses.MeanSquaredLogarithmicError()
        self.mae = tf.keras.losses.MeanAbsoluteError()
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
        # print(f'In Loss l1 call kp = pred[1] {tf.math.reduce_mean(pred_ofst).numpy()}')
        # print(f'In Loss l1 call kp = targ[1] {tf.math.reduce_mean(targ_ofst).numpy()}')
        diff = tf.math.abs(pred_ofst - targ_ofst)  # (b, n_pts, n_kpts, 3)
        # print(f'In Loss l1 call diff {tf.math.reduce_mean(diff).numpy()}')

        mask = tf.cast(mask_labels[:, :, tf.newaxis, :], tf.float32)  # (b, n_pts, 1, 3)
        #print(f'In Loss l1 call mask {tf.math.reduce_mean(mask).numpy()}')

        masked_diff = diff * mask
        #print(f'In Loss l1 call masked_diff {tf.math.reduce_mean(masked_diff).numpy()}')

        num_on_object = tf.math.reduce_sum(tf.cast(mask_labels, tf.float32))
        #print(f'num_on_object {num_on_object}')
        # loss = tf.math.reduce_mean(masked_diff)/(tf.math.reduce_mean(mask)) 
        loss = tf.reduce_sum(masked_diff) / (1e-5 + num_on_object) # lukas loss
        print(f'loss {loss}')

        return loss


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

    def call(self, y_true_list, y_pred_list, s1, s2, s3, s4, s5):
        # Get data
        [gt_depth, gt_RT, gt_mask, gt_disp] = y_true_list[0], y_true_list[1], y_true_list[2], y_true_list[3]
        gt_mask = tf.expand_dims(gt_mask, axis=-1) # better to move this to simpose.py
        cropped_rgb_l, cropped_rgb_r = y_pred_list[8], y_pred_list[9]
        norm_bbox = y_pred_list[7]
        # depth, kp, sm, cp, xyz_pred, sampled_inds_in_original_image, mesh_kpts, norm_bbox, cropped_rgbs_l, cropped_rgbs_r, w, attention, normalized_magnitude
        [pred_depth, pred_kp, pred_sm, pred_cp] = y_pred_list[0], y_pred_list[1], y_pred_list[2], y_pred_list[3]
        print(f"statistics before crop gt_depth: mean: {tf.math.reduce_mean(gt_depth)} - min: {tf.math.reduce_min(gt_depth)} - max: {tf.math.reduce_max(gt_depth)} - std: {tf.math.reduce_std(gt_depth)}")
        print(f"statistics before crop gt_disp: mean: {tf.math.reduce_mean(gt_disp)} - min: {tf.math.reduce_min(gt_disp)} - max: {tf.math.reduce_max(gt_disp)} - std: {tf.math.reduce_std(gt_disp)}")
        print(f'In loss call pred_depth = pred[2] {tf.math.reduce_mean(pred_depth).numpy()}')
        # print(f'In loss call pred_sm = pred[2] {tf.math.reduce_mean(pred_sm).numpy()}')
        # print(f'In loss call pred_cp = pred[3] {tf.math.reduce_mean(pred_cp).numpy()}')
        # print(f'In Loss call kp = pred[1] {tf.math.reduce_mean(pred_kp).numpy()}')
        xyz_pred, sampled_inds, mesh_kpts, norm_bbox = y_pred_list[4], y_pred_list[5], y_pred_list[6],  y_pred_list[7]
        #print(f"statistic sampled_inds: mean: {tf.math.reduce_mean(tf.cast(sampled_inds, dtype=tf.float32))} - std: {tf.math.reduce_std(tf.cast(sampled_inds, dtype=tf.float32))}")
        # print(f'In Loss call xyz_pred {tf.math.reduce_mean(xyz_pred).numpy()}')
        # print(f'In Loss call sampled_inds {tf.shape(sampled_inds).numpy()}')
        # print(f'In Loss call cropped_rgb_l {tf.shape(cropped_rgb_l).numpy()}')
        # print(f'In Loss call cropped_rgb_r {tf.shape(cropped_rgb_r).numpy()}')
        print(f'In Loss call norm_bbox {norm_bbox}')


        

        w = y_pred_list[10]
        intrinsics = y_pred_list[12]
        crop_factor = y_pred_list[13]
        before_head = y_pred_list[14]
        w_factor_inv, h_factor_inv = y_pred_list[15], y_pred_list[16]
        disp_pred = y_pred_list[17]
        depth_emb_pred = y_pred_list[18]
        depth_emb_gt = tf.gather_nd(gt_depth, sampled_inds)

        # print(f'In Loss call f_l_5 0 {tf.math.reduce_mean(before_head[0]).numpy()}')
        # print(f'In Loss call f_l_5 1 {tf.math.reduce_mean(before_head[1]).numpy()}')
        # print(f'In Loss call f_l_5 2 {tf.math.reduce_mean(before_head[2]).numpy()}')
        # print(f'In Loss call f_l_5 3 {tf.math.reduce_mean(before_head[3]).numpy()}')

        # pred_derivative = y_pred_list[12]
        xyz_gt = StereoPvn3d.pcld_processor_tf_by_index(gt_depth+0.0001, intrinsics, sampled_inds)



        # print(f"gt_depth in the loss.call shape: {gt_depth.shape} - dtype: {gt_depth.dtype} - gt_depth min: {tf.math.reduce_min(gt_depth)} - gt_depth max: {tf.math.reduce_max(gt_depth)}")
        # print(f"pred_depth shape: {pred_depth.shape} - dtype: {pred_depth.dtype} - pred_depth min: {tf.math.reduce_min(pred_depth)} - pred_depth max: {tf.math.reduce_max(pred_depth)} ")  

        # Processing data to compare gt and pred
        #mask_valid_inds = sampled_inds>=0
        mask_selected = tf.gather_nd(gt_mask, sampled_inds)
        #mask_selected = tf.cast(mask_selected, dtype=tf.float32)
        #mask_selected = StereoPvn3d.replace_values_seg_of_invalid_indices(mask_selected, sampled_inds, value=0.)
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

        print(f'gt_disp {tf.shape(gt_disp)} - w factor inv {tf.shape(w_factor_inv)}')

        w_factor_inv = tf.reshape(w_factor_inv, [-1, 1 ,1, 1])
        gt_disp = gt_disp * w_factor_inv # tf.cast(crop_factor, tf.float32)
        print(f'gt_disp {tf.shape(gt_disp)} - w factor inv {tf.shape(w_factor_inv)}')
        gt_disp = gt_disp[..., :1]
        

        print(f"statistics after crop gt_depth: mean: {tf.math.reduce_mean(gt_depth)} - min: {tf.math.reduce_min(gt_depth)} - max: {tf.math.reduce_max(gt_depth)} - std: {tf.math.reduce_std(gt_depth)}")
        print(f"statistics after crop gt_disp: mean: {tf.math.reduce_mean(gt_disp)} - min: {tf.math.reduce_min(gt_disp)} - max: {tf.math.reduce_max(gt_disp)} - std: {tf.math.reduce_std(gt_disp)}")
        # print(f"statistics gt_kp per batch: mean: {tf.math.reduce_mean(gt_kp, axis=[1,2,3])} - std: {tf.math.reduce_std(gt_kp, axis=[1,2,3])} - max: {tf.math.reduce_max(gt_kp, axis=[1,2,3])} - min: {tf.math.reduce_min(gt_kp, axis=[1,2,3])} - num_nonzeros: {tf.math.count_nonzero(gt_kp, axis=[1,2,3])}")
        # print(f"statistics pred_kp per batch: mean: {tf.math.reduce_mean(pred_kp, axis=[1,2,3])} - std: {tf.math.reduce_std(pred_kp, axis=[1,2,3])} - max: {tf.math.reduce_max(pred_kp, axis=[1,2,3])} - min: {tf.math.reduce_min(pred_kp, axis=[1,2,3])} - num_nonzeros: {tf.math.count_nonzero(pred_kp, axis=[1,2,3])}")
        self.initial_pose_model = _InitialPoseModel()

        b_R_pred, b_t_pred, b_voted_kpts_pred = self.initial_pose_model([xyz_gt, pred_kp, pred_cp, pred_sm, mesh_kpts])

        print(f"b_voted_kpts_pred[0,0,:]: {b_voted_kpts_pred[0,0,:]}")
        print(f"b_mesh_kpts[0,0,:]: {mesh_kpts[0,0,:]}")
        
        # pred_depth_masked = pred_depth * tf.cast(gt_mask, tf.float32)
        pred_depth_masked = tf.where(gt_depth>0.0001, pred_depth, 0.0)
        # print(f"shape gt_depth: {gt_depth.shape} - dtype: {gt_depth.dtype}")
        # print(f"shape pred_depth: {pred_depth.shape} - dtype: {pred_depth.dtype}")
        # print(f"shape pred_depth_masked: {pred_depth_masked.shape} - dtype: {pred_depth_masked.dtype}")
        # print(f"shape gt_kp: {gt_kp.shape} - dtype: {gt_kp.dtype}")
        # print(f"shape pred_kp: {pred_kp.shape} - dtype: {pred_kp.dtype}")
        # print(f"shape gt_cp: {gt_cp.shape} - dtype: {gt_cp.dtype}")
        # print(f"shape pred_cp: {pred_cp.shape} - dkp_cp_loss_discounttype: {pred_cp.dtype}")
        
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
        mse_loss = self.mae(gt_depth, pred_depth_masked)
        # mlse_loss = self.mlse(gt_depth, pred_depth)

        mse_disp_loss = self.mae(gt_disp, disp_pred)

        # loss = square(log(y_true + 1.) - log(y_pred + 1.))

        mlse_loss = self.mlse(gt_depth, pred_depth_masked) # tf.math.reduce_sum(tf.math.abs(gt_depth - pred_depth_masked)) / tf.cast(tf.math.count_nonzero(gt_depth), tf.float32) #tf.coun_non_zero
        mlse_disp_loss = self.mlse(gt_disp, disp_pred)
        ssim_loss_value = self.ssim_loss(gt_depth, pred_depth_masked)
        ssim_disp_loss_value = self.ssim_loss(gt_disp, disp_pred)

        # xyz_loss = self.mse(xyz_pred, xyz_gt)
        

        # pvn3d loss functions
        bs, _, _, _ = pred_kp.shape
        #print(f'In Loss call 2 kp = pred[1] {tf.math.reduce_mean(pred_kp).numpy()}')
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
        
        depth_emb_loss = self.mae(depth_emb_gt, depth_emb_pred)


        # if self.params.distribute_training:
        #     loss_seg = tf.reduce_sum(loss_seg) * (1 / bs)


        '''Print all losses for debugging:'''
        print("-------------------\nLoss PVN3D before apply discounts")

        print('loss_kp', loss_kp)
        print('loss_seg', loss_seg)
        print('loss_cp', loss_cp)
        print('loss ssim disp', ssim_disp_loss_value)
        print('loss_mse_disp', mse_disp_loss)
        print('loss_mlse_disp', mlse_disp_loss)
        print('loss_deriv', deriv_loss)
        print('loss_hessian', hessian_loss)
        print('loss_dpt_large_image', mse_loss)
        print('loss_dpt_emb', depth_emb_loss)
        # print('loss_xyz', xyz_loss)

        #print('loss_l1_dpt', l1_loss)        


        # loss_cp_scaled = self.cp_loss_discount * loss_cp #10 
        loss_cp_scaled = 1./(2.*tf.math.exp(s4*2)) * loss_cp 
        # loss_kp_scaled = self.kp_loss_discount * loss_kp
        loss_kp_scaled = 1./(2.*tf.math.exp(s3*2)) * loss_kp 
        # loss_seg_scaled = self.sm_loss_discount * loss_seg
        loss_seg_scaled = 1./(tf.math.exp(s5*2)) * loss_seg 
        # mse_loss_scaled = self.mse_loss_discount * mse_disp_loss
        mse_loss_scaled = 1./(2.*tf.math.exp(s1*2)) * mse_disp_loss 
        mlse_loss_scaled = self.mlse_loss_discount * mlse_disp_loss # 0.000005
        #ssim_loss_scaled = self.ssim_loss_discount * ssim_disp_loss_value
        ssim_loss_scaled = 1./(2.*tf.math.exp(s2*2) )* ssim_disp_loss_value 
        deriv_loss_scaled = self.deriv_loss_discount * deriv_loss
        hessian_loss_scaled = self.hessian_loss_discount * hessian_loss
        # xyz_loss_scaled = 1 * xyz_loss

        
        # main loss         

        # loss = (
        #     mse_loss_scaled + tf.math.maximum(s1, 0.0)
        #     #+ mlse_loss_scaled
        #     #+ ssim_loss_scaled + tf.math.maximum(s2, 0.0)
        #     # + deriv_loss_scaled
        #     + loss_cp_scaled + tf.math.maximum(s4, 0.0) # from pvn3d
        #     + loss_kp_scaled + tf.math.maximum(s3, 0.0) # from pvn3d
        #     + loss_seg_scaled + tf.math.maximum(s5, 0.0) # from pvn3d
        #     # + hessian_loss_scaled
        #     # + xyz_loss_scaled
        # )

        # loss for final kp prediction
        loss = (
            loss_kp
            + loss_cp
            + loss_seg
            + depth_emb_loss
        )

        return (
            loss,
            mse_loss,
            mse_disp_loss,
            ssim_disp_loss_value,
            deriv_loss_scaled,
            loss_cp,
            loss_kp,
            loss_seg,
            hessian_loss,
            depth_emb_loss,
            #xyz_loss_scaled,
        )



        return loss #, logs # ssim_loss_value # , loss_cp, loss_kp, loss_seg
