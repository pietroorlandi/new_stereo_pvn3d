import tensorflow as tf
import focal_loss
from skimage.metrics import structural_similarity as ssim



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

        self.mse = tf.keras.losses.MeanSquaredError()
        self.mlse = tf.keras.losses.MeanSquaredLogarithmicError()
        #self.l1 = tf.keras.losses.Huber(reduction=red)
        # self.seg_loss = focal_loss.SparseCategoricalFocalLoss(
        #     2, from_logits=True, reduction=red
        # )
        self.seg_loss = focal_loss.SparseCategoricalFocalLoss(
            2, from_logits=True
        )
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
        print(f"SSIM before mean: {ssim}")
        ssim = tf.reduce_mean(ssim, axis=-1)
        print(f"SSIM after mean: {ssim}")

        # Compute SSIM loss (SSIM is a similiarity, so the 1 - SSIM is the loss)
        return 1 - ssim

    def call(self, y_true_list, y_pred_list):
        # gt_depth = y_true[..., :1]
        # gt_seg = y_true[..., 1:]

        # pred_depth = y_pred[..., :1]
        # pred_seg = y_pred[..., 1:]

        # labels: depth, label_list, kp_targ_ofst, ctr_targ_ofst, mask_label
        # print('y_true inside the loss funciton', y_true_list)
        # print('y_pred inside the loss function', y_pred_list)


        # data: rgb, rgb_R, self.baseline, self.intrinsic_matrix, sampled_index
        # model_output: depth, kp, sm, cp
        [gt_depth, gt_kp, gt_cp, mask_label, pnt_cld] = y_true_list
        # change in [gt_depth, gt_kp, gt_cp, mask_label, pnt_cld] = y_true_list
        [pred_depth, pred_kp, pred_sm, pred_cp, norm_bbox] = y_pred_list

        print("Loss shapes")
        print(f"depth: gt_depth.shape: {gt_depth.shape} - pred_depth.shape: {pred_depth.shape}")
        print(f"kp: gt_kp.shape: {gt_kp.shape} - pred_kp.shape: {pred_kp.shape}")
        print(f"cp: gt_cp.shape: {gt_cp.shape} - pred_cp: {pred_cp.shape}")
        print(f"seg: mask_label.shape: {mask_label.shape} - pred_sm: {pred_sm.shape}")


        gt_depth = gt_depth[..., :1]
        pred_depth = pred_depth[..., :1]

        gt_depth = tf.image.crop_and_resize(
            tf.cast(gt_depth, tf.float32),
            norm_bbox,
            tf.range(tf.shape(gt_depth)[0]),
            self.resnet_input_shape[:2]
        )
        # print('gt_depth', gt_depth)
        # print('pred_depth', pred_depth)
        

        #l1_loss = self.l1(gt_depth, pred_depth)
        mse_loss = self.mse(gt_depth, pred_depth)
        mlse_loss = self.mlse(gt_depth, pred_depth)
        # seg_loss_value = self.seg_loss_discount * self.seg_loss(gt_stereo_seg, pred_stereo_seg)

        # ssim_loss = tf.math.reduce_mean(
        #     1.0 - tf.image.ssim_multiscale(gt_depth, pred_depth, max_val=self.ssim_max_val)
        # )
        ssim_loss_value = self.ssim_loss(gt_depth, pred_depth)
        

        # pvn3d loss functions
        bs, _, _, _ = pred_kp.shape

        loss_kp = self.l1_loss(pred_ofst=pred_kp,
                               targ_ofst=gt_kp,
                               mask_labels=mask_label)
        # if binary_loss is True:

        #     if not self.params.seg_from_logits:
        #         seg_pre = tf.nn.softmax(seg_pre)

        #     seg_pre = tf.unstack(seg_pre, axis=2)[1]
        #     label = tf.argmax(label, axis=2)
        #     loss_seg = self.BinaryFocalLoss(label, seg_pre)  # return batch-wise value
        # else:
        print(f"seg: mask_label.shape: {mask_label.shape} - pred_sm: {pred_sm.shape}")
        print(f"seg: mask_label[:,:,0].shape: {mask_label[:,:,0].shape} - pred_sm[:,:,0]: {pred_sm[:,:,0].shape}")
        print(f"seg: mask_label[:,:,0].dtype: {mask_label[:,:,0].dtype} - pred_sm[:,:,0]: {pred_sm[:,:,0].dtype}")
        mask_label = tf.cast(mask_label, dtype=tf.float32)
        pred_sm = tf.cast(pred_sm, dtype=tf.float32)


        loss_seg = self.BinaryFocalLoss(mask_label, pred_sm)  # labels [bs, n_pts, n_cls] this is from logits
        # change in something similiar: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=self.reduction)


        loss_cp = self.l1_loss(pred_ofst=pred_cp,
                               targ_ofst=gt_cp,
                              mask_labels=mask_label)

        # if self.params.distribute_training:
        #     loss_seg = tf.reduce_sum(loss_seg) * (1 / bs)


        '''Print all losses for debugging:'''
        print("-------------------\nLoss PVN3D")

        print('loss_kp', loss_kp)
        print('loss_seg', loss_seg)
        print('loss_cp', loss_cp)
        #print('loss_ssim', ssim_loss)
        print('loss_mse', mse_loss)
        print('loss_mlse', mlse_loss)
        #print('loss_l1_dpt', l1_loss)        


        loss_cp_scaled = self.cp_loss_discount * loss_cp
        loss_kp_scaled = self.kp_loss_discount * loss_kp
        loss_seg_scaled = self.sm_loss_discount * loss_seg
        mse_loss_scaled = self.mse_loss_discount * mse_loss
        mlse_loss_scaled = self.mlse_loss_discount * mlse_loss
        ssim_loss_scaled = self.ssim_loss_discount * ssim_loss_value
        

        loss = (
             mse_loss_scaled
            + mlse_loss_scaled
            + ssim_loss_scaled
            + loss_cp_scaled # from pvn3d
            + loss_kp_scaled # from pvn3d
            + loss_seg_scaled # from pvn3d
        )

        return (
            loss,
            mse_loss_scaled,
            mlse_loss_scaled,
            ssim_loss_scaled,
            loss_cp_scaled,
            loss_kp_scaled,
            loss_seg_scaled 
        )



        return loss#, logs # ssim_loss_value # , loss_cp, loss_kp, loss_seg
