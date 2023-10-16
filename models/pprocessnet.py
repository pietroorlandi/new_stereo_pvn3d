import tensorflow as tf
from .geometry import batch_rt_svd_transform, batch_pts_clustering_with_std, batch_get_pt_candidates_tf

class _InitialPoseModel(tf.keras.Model):
    def __init__(self, n_point_candidate=10):
        super(_InitialPoseModel, self).__init__(name = "Voting_and_Clustering")
        self.n_point_candidates = n_point_candidate

    def call(self, inputs, training=False, mask=None):
        pcld_input, kpts_pre_input, cpt_pre_input, seg_pre_input, mesh_kpts_input = inputs


        # num_nan_pcl_input = tf.reduce_sum(tf.cast(tf.math.is_nan(pcld_input), tf.int32))
        # num_nan_kpts_pre_input= tf.reduce_sum(tf.cast(tf.math.is_nan(kpts_pre_input), tf.int32))
        # num_nan_seg_pre_input = tf.reduce_sum(tf.cast(tf.math.is_nan(seg_pre_input), tf.int32))
        # num_nan_cpt_pre_input = tf.reduce_sum(tf.cast(tf.math.is_nan(cpt_pre_input), tf.int32))
        # max_pcl_input = tf.math.reduce_max(pcld_input)
        # min_pcl_input = tf.math.reduce_min(pcld_input)
        # max_kpts_pre_input = tf.math.reduce_max(kpts_pre_input)
        # min_kpts_pre_input = tf.math.reduce_min(kpts_pre_input)
        # max_seg_pre_input = tf.math.reduce_max(seg_pre_input)
        # min_seg_pre_input = tf.math.reduce_min(seg_pre_input)
        # max_cpt_pre_input = tf.math.reduce_max(cpt_pre_input)
        # min_cpt_pre_input = tf.math.reduce_min(cpt_pre_input)
        # num_nonzero_pcl_input = tf.math.count_nonzero(pcld_input)
        # num_nonzero_kpts_pre_input= tf.math.count_nonzero(kpts_pre_input)
        # num_nonzero_seg_pre_input = tf.math.count_nonzero(seg_pre_input)
        # num_nonzero_cpt_pre_input = tf.math.count_nonzero(cpt_pre_input)

        # print(f'pcld_input: num_nan: {num_nan_pcl_input} - min: {min_pcl_input} - max: {max_pcl_input} - num_non_zero: {num_nonzero_pcl_input}')
        # print(f'kpts_pre_input: num_nan: {num_nan_kpts_pre_input} - min: {min_kpts_pre_input} - max: {max_kpts_pre_input} - num_non_zero: {num_nonzero_kpts_pre_input}')
        # print(f'cpt_pre_input: num_nan: {num_nan_cpt_pre_input} - min: {min_cpt_pre_input} - max: {max_cpt_pre_input} - num_non_zero: {num_nonzero_cpt_pre_input}')
        # print(f'seg_pre_input: num_nan: {num_nan_seg_pre_input} - min: {min_seg_pre_input} - max: {max_seg_pre_input} - num_non_zero: {num_nonzero_seg_pre_input}')


        obj_kpts = batch_get_pt_candidates_tf(pcld_input, kpts_pre_input, seg_pre_input,
                                              cpt_pre_input, self.n_point_candidates)
        
        # print(f"obj_kpts: {obj_kpts}")

        kpts_voted = batch_pts_clustering_with_std(obj_kpts)
        # print(f"kpts_voted: {kpts_voted}")

        n_pts = tf.shape(kpts_voted)[1]
        weights_vector = tf.ones(shape=(n_pts,))
        batch_R, batch_t = batch_rt_svd_transform(mesh_kpts_input, kpts_voted, weights_vector)
        batch_t = tf.reshape(batch_t, shape=(-1, 3))  # reshape from [bs, 3, 1] to [bs, 3]
        return batch_R, batch_t, kpts_voted
