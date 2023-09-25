import tensorflow as tf
from .geometry import batch_rt_svd_transform, batch_pts_clustering_with_std, batch_get_pt_candidates_tf, batch_pts_clustering_with_std_pietro

class _InitialPoseModel(tf.keras.Model):
    def __init__(self, n_point_candidate=10):
        super(_InitialPoseModel, self).__init__(name = "Voting_and_Clustering")
        self.n_point_candidates = n_point_candidate

    def call(self, inputs, training=False, mask=None):
        pcld_input, kpts_pre_input, cpt_pre_input, seg_pre_input = inputs

        obj_kpts = batch_get_pt_candidates_tf(pcld_input, kpts_pre_input, seg_pre_input,
                                              cpt_pre_input, self.n_point_candidates)
        #print(f"obj_kpts shape: {obj_kpts.shape}")
        #print(f"obj_kpts: {obj_kpts}")
        kpts_voted = batch_pts_clustering_with_std(obj_kpts)
        return kpts_voted
        n_pts = tf.shape(kpts_voted)[1]
        weights_vector = tf.ones(shape=(n_pts,))
        batch_R, batch_t = batch_rt_svd_transform(mesh_kpts_input, kpts_voted, weights_vector)
        batch_t = tf.reshape(batch_t, shape=(-1, 3))  # reshape from [bs, 3, 1] to [bs, 3]
        return batch_R, batch_t, kpts_voted