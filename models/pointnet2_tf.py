import numpy as np
import tensorflow as tf
from typing import List
from dataclasses import dataclass


@dataclass
class PointNet2Params:
    bn: bool
    is_train: bool
    keep_prob: float
    return_features: bool
    use_tf_interpolation: bool
    use_tfx: bool
    
class _PointNet2TfXModel(tf.keras.Model):
    """exclusive tensorflow model"""

    def __init__(self, params: PointNet2Params, num_classes):
        super().__init__(name="PointNet2")
        self.params = params
        self.activation = tf.nn.relu
        self.keep_prob = self.params.keep_prob
        self.num_classes = num_classes
        self.bn = self.params.bn

        self.kernel_initializer = "glorot_normal"
        self.kernel_regularizer = None

        self.init_network()

    def init_network(self):
        self.sa_1 = Pointnet_SA(
            npoint=512,
            radius=0.1,
            nsample=32,
            mlp=[512, 512, 256], #mlp=[32, 32, 64],
            activation=self.activation,
            bn=self.bn,
        )

        self.sa_2 = Pointnet_SA(
            npoint=128,
            radius=0.2,
            nsample=32,
            mlp = [256, 256, 128],# mlp=[64, 64, 128],
            activation=self.activation,
            bn=self.bn,
        )

        self.sa_3 = Pointnet_SA(
            npoint=32,
            radius=0.4,
            nsample=32,
            mlp=[128, 128, 256],
            activation=self.activation,
            bn=self.bn,
        )

        self.sa_4 = Pointnet_SA(
            npoint=8,
            radius=0.8,
            nsample=32,
            mlp=[256, 256, 512],
            activation=self.activation,
            bn=self.bn,
        )

        self.fp_1 = Pointnet_FP(mlp=[256, 256], activation=self.activation, bn=self.bn)

        self.fp_2 = Pointnet_FP(mlp=[256, 256], activation=self.activation, bn=self.bn)

        self.fp_3 = Pointnet_FP(mlp=[256, 128], activation=self.activation, bn=self.bn)

        self.fp_4 = Pointnet_FP(mlp=[128, 128, 128], activation=self.activation, bn=self.bn)

        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.num_classes, kernel_size=1, activation=None
        )

    def call(self, inputs, training=None):
        l0_xyz = inputs[0]
        l0_points = inputs[1]

        l1_xyz, l1_points = self.sa_1(l0_xyz, l0_points, training=training)
        l2_xyz, l2_points = self.sa_2(l1_xyz, l1_points, training=training)
        l3_xyz, l3_points = self.sa_3(l2_xyz, l2_points, training=training)
        l4_xyz, l4_points = self.sa_4(l3_xyz, l3_points, training=training)

        l3_points = self.fp_1(l3_xyz, l4_xyz, l3_points, l4_points, training=training)
        l2_points = self.fp_2(l2_xyz, l3_xyz, l2_points, l3_points, training=training)
        l1_points = self.fp_3(l1_xyz, l2_xyz, l1_points, l2_points, training=training)
        l0_points = self.fp_4(l0_xyz, l1_xyz, l0_points, l1_points, training=training)

        if self.params.return_features:
            return l0_points
        else:
            seg_features = self.conv1d(l0_points)

        return seg_features


class Pointnet_SA(tf.keras.layers.Layer):
    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        mlp: List[int],
        activation=tf.nn.relu,
        bn=False,
    ):
        super(Pointnet_SA, self).__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = mlp
        self.activation = activation
        self.bn = bn

        self.mlp_list: List[CustomConv2d] = []

    def build(self, input_shape):
        for i, n_filters in enumerate(self.mlp):
            self.mlp_list.append(CustomConv2d(n_filters, activation=self.activation, bn=self.bn))

        super(Pointnet_SA, self).build(input_shape)

    def call(self, xyz, points, training=True):
        # xyz: (b, n_pts, 3)
        # points: (b, n_pts, 6) # rgb + normals

        new_xyz, new_points, idx, grouped_xyz = self.sample_and_group(
            self.npoint, self.radius, self.nsample, xyz, points
        )

        for i, mlp_layer in enumerate(self.mlp_list):
            new_points = mlp_layer(new_points, training=training)

        new_points = tf.math.reduce_max(new_points, axis=2, keepdims=False)

        return new_xyz, new_points

    def sample_and_group(
        self, npoint: int, radius: float, nsample: int, xyz, points
    ):  # xyz: (b, n_pts, 3), points: (b, n_pts, 6)
        # fps_inds = farthest_point_sample(npoint, xyz)
        fps_inds = self.uniform_sampling(npoint, xyz)
        new_xyz = tf.gather(xyz, fps_inds, batch_dims=1)  # (batch_size, npoint, 3)
        idx = self.query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = tf.gather(xyz, idx, batch_dims=1)  # (batch_size, npoint, nsample, 3)
        grouped_xyz -= new_xyz[:, :, tf.newaxis, :]  # translation normalization
        grouped_points = tf.gather(points, idx, batch_dims=1)

        new_points = tf.concat(
            [grouped_xyz, grouped_points], axis=-1
        )  # (batch_size, npoint, nample, 3+channel)

        return new_xyz, new_points, idx, grouped_xyz

    @staticmethod
    def uniform_sampling(npoint, xyz):
        b, n = tf.shape(xyz)[0], tf.shape(xyz)[1]
        inds_tf = tf.range(n)
        inds_tf = tf.tile(inds_tf[:, tf.newaxis], (1, b))  # [num_points, 5]
        inds_tf = tf.random.shuffle(inds_tf)[:npoint]
        inds_tf = tf.transpose(inds_tf)
        return inds_tf

    @staticmethod
    def query_ball_point(radius, nsample, xyz1, xyz2):
        # find distance for each point of xyz1 to each point of xyz2
        # xyz1: (batch_size, n, 3)
        # xyz2: (batch_size, m, 3)
        # return: (batch_size, m, nsample)

        # returns indices to the *nsample* nearest neighbors if dist[i]<radius
        # if there are less than nsample neighbors, the remaining entries are filled with the first found index

        dists = tf.linalg.norm(
            xyz2[:, :, tf.newaxis, :] - xyz1[:, tf.newaxis, :, :], axis=-1
        )  # [b, m, n]

        # arg sort
        idx = tf.argsort(dists, axis=-1)  # [b, m, n]
        idx = idx[:, :, :nsample]  # [b, m, nsample]

        # dists to the nsample nearest neighbors
        corresponding_dists = tf.gather(dists, idx, batch_dims=2)
        valid_mask = corresponding_dists < radius
        # [b, m, nsample] distance at index i to the i-th nearest neighbor

        # we sort to exactly replicate the CUDA version.
        # although, sorting shouldn't be necessary and it kinda doesn't make sense
        idx = tf.where(valid_mask, idx, tf.int32.max)  # [b, m, nsample]
        sorted_idx = tf.sort(idx, axis=-1)  # [b, m, nsample]
        sorted_idx = tf.where(valid_mask, sorted_idx, sorted_idx[:, :, :1])  # [b, m, nsample]

        return sorted_idx


class Pointnet_FP(tf.keras.layers.Layer):
    def __init__(self, mlp, activation=tf.nn.relu, bn=False):
        super(Pointnet_FP, self).__init__()

        self.mlp = mlp
        self.activation = activation
        self.bn = bn

        self.mlp_list = []

    def build(self, input_shape):
        for i, n_filters in enumerate(self.mlp):
            self.mlp_list.append(SharedMlP(n_filters, activation=self.activation, bn=self.bn))
        super(Pointnet_FP, self).build(input_shape)

    def call(self, xyz_target, xyz_source, feats_target, feats_source, training=True):
        weight, neighbour_idx = self.three_nn(xyz_source, xyz_target)
        interpolated_feats = self.three_interpolate(feats_source, neighbour_idx, weight)

        if feats_target is not None:
            new_feats_target = tf.concat(
                axis=2, values=[interpolated_feats, feats_target]
            )  # B,ndataset1,nchannel1+nchannel2
        else:
            new_feats_target = interpolated_feats

        new_feats_target = tf.expand_dims(
            new_feats_target, 2
        )  # new feats bs, n_points_2, 1, feats

        for i, mlp_layer in enumerate(self.mlp_list):
            new_feats_target = mlp_layer(new_feats_target, training=training)

        new_feats_target = new_feats_target[:, :, 0, :]

        return new_feats_target

    @tf.function
    def three_nn(self, xyz_source, xyz_target):
        """
        Up-interpolating features of xyz_source to xyz_target
        Building interpolation correspondences in xyz_source based on the distance, where n_pts_source < n_pts_target
        :param xyz_source: bs, n_pts_source, 3
        :param xyz_target: bs, n_pts_target, 3
        :return: dis, idx
        """

        k = 3
        n_pts_target = tf.shape(xyz_target)[1]
        xyz_source = tf.repeat(
            tf.expand_dims(xyz_source, axis=1), repeats=n_pts_target, axis=1
        )  # bs, n_pts_target, n_pts_source, 3
        xyz_target = tf.expand_dims(xyz_target, axis=2)  # bs, n_pts_target, 1, 3
        dis = tf.linalg.norm(
            tf.subtract(xyz_source, xyz_target), axis=-1
        )  # bs, n_pts_target, n_pts_source
        neighbour_dis, neighbour_idx = tf.math.top_k(-1 * dis, k=k)  # bs, n_pts_target, 3

        neighbour_dis = tf.maximum(-1 * neighbour_dis, 1e-10)
        norm = tf.reduce_sum((1.0 / neighbour_dis), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / neighbour_dis) / norm

        return weight, neighbour_idx

    @tf.function
    def three_interpolate(self, feats_source, idx, weights):
        """
        Interpolating features of xyz_source to xyz_target
        :param feats_source: the features from previous layer bs, n_pts_source, c
        :param idx: corresponding index in xyz_source
        :param weights: inverse distance weights (the farther the less important)
        :return: interpolated features bs, n_pts_target
        """
        weights_sum = tf.reduce_sum(weights, axis=-1, keepdims=True) + 1e-6  # bs, n_pts_target, 1
        weights_expand = tf.expand_dims(weights, axis=-1)  # bs, n_pts_target, 3, 1
        feats_selected = tf.gather(
            feats_source, indices=idx, batch_dims=1
        )  # bs, n_pts_target, 3, c
        inter_feats = (
            tf.reduce_sum(weights_expand * feats_selected, axis=2) / weights_sum
        )  # bs, n_pts_target, c
        return inter_feats


class SharedMlP(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        strides=[1, 1],
        activation=tf.nn.relu,
        padding="VALID",
        initializer="glorot_normal",
        bn=False,
    ):
        super(SharedMlP, self).__init__()

        self.filters = filters
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.initializer = initializer
        self.bn = bn

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(1, 1, input_shape[-1], self.filters),
            initializer=self.initializer,
            trainable=True,
            name="pnet_conv",
        )

        if self.bn:
            self.bn_layer = tf.keras.layers.BatchNormalization()

        super(SharedMlP, self).build(input_shape)

    def call(self, inputs, training=True):
        points = tf.nn.conv2d(inputs, filters=self.w, strides=self.strides, padding=self.padding)

        if self.bn:
            points = self.bn_layer(points, training=training)

        if self.activation:
            points = self.activation(points)

        return points


class CustomConv2d(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        strides=[1, 1],
        activation=tf.nn.relu,
        padding="VALID",
        initializer="glorot_normal",
        bn=False,
    ):
        super(CustomConv2d, self).__init__()

        self.filters = filters
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.initializer = initializer
        self.bn = bn

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(1, 1, input_shape[-1], self.filters),
            initializer=self.initializer,
            trainable=True,
            name="pnet_conv",
        )

        if self.bn:
            self.bn_layer = tf.keras.layers.BatchNormalization()

        super(CustomConv2d, self).build(input_shape)

    def call(self, inputs, training=True):
        points = tf.nn.conv2d(inputs, filters=self.w, strides=self.strides, padding=self.padding)

        if self.bn:
            points = self.bn_layer(points, training=training)

        if self.activation:
            points = self.activation(points)

        return points