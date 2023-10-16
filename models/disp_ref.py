import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D


class DispRefNetParams:
    def __init__(self):
        self.kp_conv1d_1_dim = 1024
        self.kp_conv1d_2_dim = 512
        self.kp_conv1d_3_dim = 256
        self.cp_conv1d_1_dim = 1024
        self.cp_conv1d_2_dim = 512
        self.cp_conv1d_3_dim = 128
        self.seg_conv1d_1_dim = 1024
        self.seg_conv1d_2_dim = 512
        self.seg_conv1d_3_dim = 128


class DispRefNets:
    def __init__(self, params: DispRefNetParams, num_pts=12228, num_kpts=8, num_cls=6, num_cpts=1, channel_xyz=3):
        self.params = params
        self.num_pts = num_pts
        self.num_kpts = num_kpts
        self.num_cls = num_cls
        self.num_cpts = num_cpts
        self.channel_xyz = channel_xyz

    def layers(self, rgbd_features):
        conv1d_1 = Conv1D(
            filters=self.params.kp_conv1d_1_dim, kernel_size=1, activation='relu', name="kp_conv1d_1")(rgbd_features)
        conv1d_2 = Conv1D(
            filters=self.params.kp_conv1d_2_dim, kernel_size=1, activation='relu', name="kp_conv1d_2")(conv1d_1)
        conv1d_3 = Conv1D(
            filters=self.params.kp_conv1d_3_dim, kernel_size=1, activation='relu', name="kp_conv1d_3")(conv1d_2)
        conv1d_4 = Conv1D(
            filters=self.num_kpts * self.channel_xyz, kernel_size=1, activation=None, name="kp_conv1d_4")(conv1d_3)
        print(f'conv1d_4 {conv1d_4}')
        print(f'num_points {self.num_pts}')
        kp_pre = tf.reshape(conv1d_4, shape=[-1, self.num_pts, self.num_kpts, self.channel_xyz])
        return kp_pre

    def build_ref_model(self, rgbd_features_shape):
        input_rgbd_features = Input(shape=rgbd_features_shape, name='rgbd_features_input')
        output = self.layers(input_rgbd_features)

        model = Model(inputs=input_rgbd_features, outputs=[kp_pre_output, sm_pre_output, cp_pre_output],
                      name='mlps_model')

        return model
