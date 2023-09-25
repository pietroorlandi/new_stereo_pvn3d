import numpy as np
import tensorflow as tf
import focal_loss


class PvnLoss:
    def __init__(
        self,
        *,
        binary_loss,
        kp_loss_discount,
        cp_loss_discount,
        seg_loss_discount,
        use_segmentation: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.binary_loss = binary_loss
        self.use_segmentation = use_segmentation
        self.kp_loss_discount = kp_loss_discount
        self.cp_loss_discount = cp_loss_discount
        self.seg_loss_discount = seg_loss_discount
        self.BinaryFocalLoss = focal_loss.BinaryFocalLoss(gamma=2, from_logits=True)
        self.CategoricalCrossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    @tf.function
    def call(self, y_true, y_pred):
        rt, mask = y_true[0], y_true[1]

        kp_pred, seg_pred, cp_pred = y_pred[0], y_pred[1], y_pred[2]
        xyz, sampled_inds, kpts_cpts = y_pred[3], y_pred[4], y_pred[5]

        mask_selected = tf.gather_nd(mask, sampled_inds)  # [b, n_pts, 1]

        kp_gt, cp_gt = self.get_offst(
            rt,
            xyz,
            mask_selected,
            kpts_cpts,
        )

        if self.use_segmentation:
            loss_kp = self.l1_loss(offset_pred=kp_pred, offset_gt=kp_gt, mask_labels=mask_selected)
            loss_cp = self.l1_loss(offset_pred=cp_pred, offset_gt=cp_gt, mask_labels=mask_selected)
        else:
            loss_kp = tf.reduce_mean(tf.math.abs(kp_pred - kp_gt))
            loss_cp = tf.reduce_mean(tf.math.abs(cp_pred - cp_gt))

        if self.binary_loss:
            loss_seg = self.BinaryFocalLoss(mask_selected, seg_pred)
        else:
            raise NotImplementedError
            loss_seg = self.CategoricalCrossentropy(
                label, seg_pred
            )  # labels [bs, n_pts, n_cls] this is from logits

        _loss_cp = self.cp_loss_discount * loss_cp
        _loss_kp = self.kp_loss_discount * loss_kp
        _loss_seg = self.seg_loss_discount * loss_seg

        loss = _loss_cp + _loss_kp + _loss_seg

        return loss, loss_cp, loss_kp, loss_seg

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

        print("offsets shape:", offsets.shape)
        print("mask_selected shape:", mask_selected.shape)

        offsets = offsets * tf.cast(mask_selected[:, :, tf.newaxis], tf.float32)
        # offsets = tf.where(mask_selected[:, :, tf.newaxis] == 1, offsets, 0.0)
        kp_offsets = offsets[:, :, :-1, :]  # [b, n_pts, 8, 3]
        cp_offsets = offsets[:, :, -1:, :]  # [b, n_pts, 1, 3]
        return kp_offsets, cp_offsets

    @staticmethod
    def l1_loss(offset_pred, offset_gt, mask_labels):
        """Calculate the average l1 loss between the predicted and the ground truth offsets.
        The loss is calculated only for points, that are part of the object.

        Args:
            offset_pred (b,n_pts,n_kpts,3): Predicted offsets
            offset_gt (b,n_pts,n_kpts,3): Ground truth offsets
            mask_labels (b,n_pts,1): Mask of selected points (0|1)

        Returns:
            loss: l1 loss
        """
        diff = tf.math.abs(offset_pred - offset_gt)  # (b, n_pts, n_kpts, 3)
        mask = tf.cast(mask_labels[:, :, tf.newaxis, :], tf.float32)  # (b, n_pts, 1, 3)
        masked_diff = diff * mask
        num_on_object = tf.math.reduce_sum(tf.cast(mask_labels, tf.float32))
        loss = tf.reduce_sum(masked_diff) / (1e-5 + num_on_object)

        return loss
