import tensorflow as tf
import focal_loss


class StereoLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        mse_loss_discount,
        l1_loss_discount,
        ssim_loss_discount,
        mlse_loss_discount,
        seg_loss_discount,
        ssim_max_val,
        **kwargs
    ):
        super().__init__()
        self.mse_loss_discount = mse_loss_discount
        self.l1_loss_discount = l1_loss_discount
        self.ssim_loss_discount = ssim_loss_discount
        self.mlse_loss_discount = mlse_loss_discount
        self.seg_loss_discount = seg_loss_discount
        self.ssim_max_val = ssim_max_val

        red = tf.keras.losses.Reduction.NONE

        self.mse = tf.keras.losses.MeanSquaredError(reduction=red)
        self.mlse = tf.keras.losses.MeanSquaredLogarithmicError(reduction=red)
        self.l1 = tf.keras.losses.Huber(reduction=red)
        self.seg_loss = focal_loss.SparseCategoricalFocalLoss(
            2, from_logits=True, reduction=red
        )

    def call(self, y_true, y_pred):
        gt_depth = y_true[..., :1]
        gt_seg = y_true[..., 1:]

        pred_depth = y_pred[..., :1]
        pred_seg = y_pred[..., 1:]

        l1_loss = self.l1(gt_depth, pred_depth)
        mse_loss = self.mse(gt_depth, pred_depth)
        mlse_loss = self.mlse(gt_depth, pred_depth)
        seg_loss_value = self.seg_loss_discount * self.seg_loss(gt_seg, pred_seg)

        ssim_loss = tf.math.reduce_mean(
            1.0 - tf.image.ssim(gt_depth, pred_depth, max_val=self.ssim_max_val)
        )

        mse_loss_scaled = self.mse_loss_discount * mse_loss
        mlse_loss_scaled = self.mlse_loss_discount * mlse_loss
        l1_loss_scaled = self.l1_loss_discount * l1_loss
        ssim_loss_scaled = self.ssim_loss_discount * ssim_loss

        loss = (
            l1_loss_scaled
            + mse_loss_scaled
            + mlse_loss_scaled
            + ssim_loss_scaled
            + seg_loss_value
        )
        return loss
