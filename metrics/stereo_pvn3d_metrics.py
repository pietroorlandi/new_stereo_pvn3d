import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from cvde.job.job_tracker import JobTracker

class StereoPvn3dMetrics(tf.keras.metrics.Metric):
    def __init__(self, name="stereo_pvn3d_metrics", **kwargs):
        super().__init__(name=name, **kwargs)
        self.count = self.add_weight(name="count", initializer="zeros")
        self.l1_kp = self.add_weight(name="l1_kp", initializer="zeros")
        self.l1_cp = self.add_weight(name="l1_cp", initializer="zeros")
        self.ssim = self.add_weight(name="ssim", initializer="zeros")
        self.mlse = self.add_weight(name="mlse", initializer="zeros")
        self.mse = self.add_weight(name="mse", initializer="zeros")
        self.mae = self.add_weight(name="mae", initializer="zeros")


    def calculate_ssim(self, y_true, y_pred):
        # SSIM constants (to avoid that denominator is zero)
        max_val = 5
        K1 = 0.05
        K2 = 0.15
        ssim = tf.image.ssim(y_true, y_pred, max_val, filter_size=11, filter_sigma=1.5, k1=K1, k2=K2, return_index_map=False)
        ssim = tf.reduce_mean(ssim, axis=-1)
        return ssim

    def calculate_l1(self, pred_ofst, targ_ofst, mask_labels):

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
    

    def update_state(self, y_true_list, y_pred_list, sample_weight=None):
        [gt_depth, gt_sm, gt_kp, gt_cp, mask_label, pnt_cld, ctr, kpts] = y_true_list
        [pred_depth, pred_kp, pred_sm, pred_cp] = y_pred_list

        gt_depth = gt_depth[..., :1]
        pred_depth = pred_depth[..., :1]

        # y_true = tf.cast(y_true, tf.float32)
        # y_pred = tf.cast(y_pred, tf.float32)
        print(f"gt_depth: {gt_depth} - shape: {gt_depth.shape}")
        print(f"pred_depth: {gt_depth} - shape: {pred_depth.shape}")
        print(f"gt_kp: {gt_kp} - shape: {gt_kp.shape}")   
        print(f"pred_kp: {pred_kp} - shape: {pred_kp.shape}")
        mlse = tf.reduce_mean(tf.math.square(tf.math.log1p(pred_depth) - tf.math.log1p(gt_depth)))
        mse = tf.reduce_mean(tf.square(pred_depth-gt_depth))
        mae = tf.reduce_mean(tf.abs(pred_depth-gt_depth))
        ssim_score = self.calculate_ssim(gt_depth, pred_depth)
        l1_kp = self.calculate_l1(pred_kp, gt_kp, mask_label)
        l1_cp = self.calculate_l1(pred_cp, gt_cp, mask_label)


        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            mlse *= sample_weight
            mse *= sample_weight
            mae *= sample_weight
            ssim_score *= sample_weight
            l1_kp *= sample_weight
            l1_cp *= sample_weight
        self.count.assign_add(1.) 
        self.mlse.assign_add(mlse)
        self.mse.assign_add(mse)
        self.mae.assign_add(mae)
        self.ssim.assign_add(ssim_score)     
        self.l1_kp.assign_add(l1_kp)
        self.l1_cp.assign_add(l1_cp)


    def result(self):
        mlse = tf.math.divide_no_nan(self.mlse, self.count)
        mse = tf.math.divide_no_nan(self.mse, self.count)
        mae = tf.math.divide_no_nan(self.mae, self.count)
        ssim_score = tf.math.divide_no_nan(self.ssim, self.count)
        l1_kp = tf.math.divide_no_nan(self.l1_kp, self.count)
        l1_cp = tf.math.divide_no_nan(self.l1_cp, self.count)


        # Return a dictionary that will be used in test_step method of the model
        return {"depth_mlse": mlse,
                'depth_mse':mse,
                'depth_mae':mae,
                "depth_ssim": ssim_score,
                "kp_l1": l1_kp,
                "cp_l1": l1_cp,
                }

    def reset_state(self):
        self.mlse.assign(0.0)
        self.ssim.assign(0.0)
        self.l1_kp.assign(0.0)
        self.l1_cp.assign(0.0)
        self.mse.assign(0.0)
        self.mae.assign(0.0)
        self.count.assign(0.0)
