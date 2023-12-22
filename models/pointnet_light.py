import numpy as np
import tensorflow as tf
from typing import List
from dataclasses import dataclass


@dataclass
class PointNetLightParams:
    num_out_features: int
    # bn: bool
    # is_train: bool
    # keep_prob: float
    # return_features: bool
    # use_tf_interpolation: bool
    # use_tfx: bool


class TransformNet(tf.keras.layers.Layer):
  def __init__(self, K=3, S=1, **kwargs):
    super(TransformNet, self).__init__(**kwargs)
    self.K = K
    self.S = S
    self.mlp1 = tf.keras.layers.Conv2D(64, (1, self.S), padding='VALID', strides=(1, 1), activation=tf.nn.relu)
    self.mlp2 = tf.keras.layers.Conv2D(128, (1, 1), padding='VALID', strides=(1, 1), activation=tf.nn.relu)
    self.mlp3 = tf.keras.layers.Conv2D(1024, (1, 1), padding='VALID', strides=(1, 1), activation=tf.nn.relu)
    self.maxpool2d = tf.keras.layers.GlobalMaxPooling2D()
    self.mlp4 = tf.keras.layers.Conv2D(512, (1, 1), padding='VALID', strides=(1, 1), activation=tf.nn.relu)
    self.mlp5 = tf.keras.layers.Conv2D(256, (1, 1), padding='VALID', strides=(1, 1), activation=tf.nn.relu)
    self.mlp6 = tf.keras.layers.Conv2D(self.K*self.K, kernel_size=(1, 1))
  def call(self, inputs, bn_decay=None):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return: Transformation matrix of size 3xK """
    batch_size = inputs.shape[0]
    num_point = inputs.shape[1]
    input_image = tf.expand_dims(inputs, -1)
    net = self.mlp1(input_image)
    net = self.mlp2(net)
    net = self.mlp3(net)
    net = self.maxpool2d(net)
    net = tf.keras.layers.Reshape((1, 1, 1024))(net)  # Add extra dimensions for compatibility
    net = self.mlp4(net)
    net = self.mlp5(net)
    transform = self.mlp6(net)
    transform = tf.reshape(transform, [-1, self.K, self.K])
    
    return transform


class _PointNetLightModel(tf.keras.Model):
    """exclusive tensorflow model"""


    def __init__(self, params: PointNetLightParams, num_classes):
        super().__init__(name="PointNetLight")
        self.params = params
        self.num_out_features = self.params.num_out_features
        self.transform1 = TransformNet(K=3, S=3)
        self.pointnetmini = _PointNetMini()
        self.mlp = tf.keras.layers.Conv1D(self.num_out_features, kernel_size = 1)
        # self.mlp1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        # self.mlp2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        # self.mlp3 = tf.keras.layers.Dense(self.num_out_features, activation=None)
    def call(self, inputs):
        # inputs_expanded = tf.expand_dims(inputs, -1)
        xyz, feats = inputs 
        # print(xyz.shape)
        # print(tf.shape(feats))
        t1 = self.transform1(xyz) # b, 3, 3 
        point_cloud_transformed = tf.matmul(xyz, t1) # b, num_point, 3
        # print(point_cloud_transformed.shape)
        net_input = tf.concat([point_cloud_transformed, feats], axis =-1) # b, num_points, 3 + num_feats
        # print(net.shape)
        net = self.pointnetmini(net_input) # b, num_points, 1088
        print(net)
        # net = tf.expand_dims(net, axis =-1)
        net = self.mlp(net) # b, num_points, num_out_features
        print(net)

        

        # input_image = tf.expand_dims(point_cloud_transformed, -1)


        # add pointnet with maxpool ------------------

        # net = self.mlp6(point_cloud_transformed)
        # net = tf.keras.layers.Dropout(0.7)(net)
        # net = self.mlp7(net)
        # net = tf.keras.layers.Dropout(0.7)(net)
        # net = self.mlp8(net)
        return net



        

class _PointNetMini(tf.keras.Model):
  """
  TF Model that returns point features (local features + global features)
  """
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.mlp1 = tf.keras.layers.Conv1D(64, 1, padding='VALID', strides=1, activation=tf.nn.relu)
    self.mlp2 = tf.keras.layers.Conv1D(128, 1, padding='VALID', strides=1, activation=tf.nn.relu)
    self.mlp3 = tf.keras.layers.Conv1D(1024, 1, padding='VALID', strides=1, activation=tf.nn.relu)
    self.maxpool1d = tf.keras.layers.GlobalMaxPooling1D()

  def call(self, inputs):
    batch_size = inputs.shape[0]
    n_points = inputs.shape[1]
    n_features = inputs.shape[2]

    local_features1 = self.mlp1(inputs) # (b, n_points, 64)
    local_features2 = self.mlp2(local_features1) # (b, n_points, 128)
    local_features3 = self.mlp3(local_features2) # (b, n_points, 1024)
    
    global_features =  self.maxpool1d(local_features3) # (b, 1024)
    global_features_expanded = tf.tile(tf.expand_dims(global_features, axis=1), [1, n_points, 1]) # (b, n_points, 1024)

    feats = tf.concat([local_features1, global_features_expanded], axis=-1) # (b, n_points, 1088)
    return feats