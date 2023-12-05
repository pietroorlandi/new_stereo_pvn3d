import tensorflow as tf
import tensorflow.keras as keras
from .stereo_layers import (
    ResUp,
    ResReduce,
    ResIdentity,
    DisparityAttention,
)
from dataclasses import dataclass
from typing import List


@dataclass
class DisparityDecoderParams:
    channel_multiplier: int
    base_channels: List
    num_decoder_feats: int

class DisparityDecoder(keras.Model):
    def __init__(self,
                params: DisparityDecoderParams):
        super().__init__()
        self.params = params
        self.channel_multiplier = self.params.channel_multiplier
        self.base_channels = self.params.base_channels
        self.num_decoder_feats = self.params.num_decoder_feats

        _, _, f12, f13, f14, f15 = [
            x * self.channel_multiplier * 2 for x in self.base_channels
        ]

        _, _, f22, f23, f24, f25 = [
            x * 4 * self.channel_multiplier * 2 for x in self.base_channels
        ]
        self.attention1 = DisparityAttention(max_disparity=6)
        self.resid1 = ResIdentity(filters=(f14, f24), name="resid_1_1")
        self.resid2 = ResIdentity(filters=(f14, f24), name="resid_1_2")
        self.resid3 = ResIdentity(filters=(f14, f24), name="resid_1_3")
        self.resup1 = ResUp(s=2, filters=(f13, f23), name="resup_1")

        self.attention2 = DisparityAttention(max_disparity=3)
        self.resred1 = ResReduce(f24, name="resred_1")
        self.resid4 = ResIdentity(filters=(f14, f24), name="resid_2_1")
        self.resid5 = ResIdentity(filters=(f14, f24), name="resid_2_2")
        self.resid6 = ResIdentity(filters=(f14, f24), name="resid_2_3")
        self.resup2 = ResUp(s=2, filters=(f13, f23), name="resup_2")

        self.attention3 = DisparityAttention(max_disparity=3)
        self.resred2 = ResReduce(f24, name="resred_2")
        self.resid7 = ResIdentity(filters=(f14, f24), name="resid_3_1")
        self.resid8 = ResIdentity(filters=(f14, f24), name="resid_3_2")
        self.resid9 = ResIdentity(filters=(f14, f24), name="resid_3_3")
        self.resup3 = ResUp(s=2, filters=(f13, f23), name="resup_3")

        self.attention4 = DisparityAttention(max_disparity=3)
        self.resred3 = ResReduce(f24, name="resred_3")
        self.resid10 = ResIdentity(filters=(f14, f24), name="resid_4_1")
        self.resid11 = ResIdentity(filters=(f14, f24), name="resid_4_2")
        self.resid12 = ResIdentity(filters=(f14, f24), name="resid_4_3")
        self.resup4 = ResUp(s=2, filters=(f13, f23), name="resup_4")

        self.attention5 = DisparityAttention(max_disparity=3)
        self.resred4 = ResReduce(f24, name="resred_4")
        self.resid13 = ResIdentity(filters=(f14, f24), name="resid_5_1")
        self.resid14 = ResIdentity(filters=(f14, f24), name="resid_5_2")
        self.resid15 = ResIdentity(filters=(f14, f24), name="resid_5_3")

        self.head1 = tf.keras.layers.Conv2D(
            128, # or 16?
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="deph_conv_1",
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.head2 = tf.keras.layers.Conv2D(
            self.num_decoder_feats - 2, # From the total of num_decoder_feats should be removed depth and added point cloud
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="deph_conv_2",
            #activation = 'relu',
        )

        self.disp_head = tf.keras.layers.Conv2D(
            1, 
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="disp_conv",
        )


    @tf.function
    def call(self, inputs, training=None):
        f_l_1, f_l_2, f_l_3, f_l_4, f_l_5 = inputs[0]
        f_r_1, f_r_2, f_r_3, f_r_4, f_r_5 = inputs[1]

        before_head = f_l_5
        deep = True
        attention = []
        x, w = self.attention1([f_l_5, f_r_5])
        attention.append(x)

        print(f'x after the 1st layer of attention: {x.shape}')
        if deep:
            x = self.resid1(x)
            x = self.resid2(x)
            x = self.resid3(x)
        x = (x)

        x = self.resup1(x)

        x_skip = x

        x, w = self.attention2([f_l_4,f_r_4],w)
        attention.append(x)

        print(f'x after the 2st layer of attention: {x.shape}')
        x = tf.concat([x, x_skip, w[-1]], axis=-1) #cut the concat and put it in the disparity layer? non avrebbe senso. Forse bisogna cambiare
        
        x = self.resred1(x)
        print(f'x after the resred1: {x.shape}')
        if deep:
            x = self.resid4(x)
            x = self.resid5(x)
            x = self.resid6(x)
        x = self.resup2(x)

        x_skip = x

        x, w = self.attention3(
            [f_l_3,
            f_r_3],
            previous_weights=w
        )
        attention.append(x)

        print(f'x after the 3st layer of attention: {x.shape}')
        x = tf.concat([x, x_skip, w[-1]], axis=-1)
        print(f'x after the concat: {x.shape}')
   
        x = self.resred2(x)

        x = self.resid7(x)
        x = self.resid8(x)
        x = self.resid9(x)

        x = self.resup3(x)

        x_skip = x

        x, w = self.attention4(
            [f_l_2, f_r_2], previous_weights=w,
        )
        attention.append(x)

        print(f'x after the 4st layer of attention: {x.shape}')
        x = tf.concat([x, x_skip, w[-1]], axis=-1)
        print(f'x after the concat: {x.shape}')

        x = self.resred3(x)

        x = self.resid10(x)
        x = self.resid11(x)
        x = self.resid12(x)
        x = self.resup4(x)

        x_skip = x

        x, w = self.attention5(
            [f_l_1,
            f_r_1],
            previous_weights=w,
        )
        attention.append(x)
        print(f'x after the 5st layer of attention: {x.shape}')
        x = tf.concat([x, x_skip], axis=-1)
        print(f'x after the concat: {x.shape}')

        x = self.resred4(x)

        x = self.resid13(x)
        x = self.resid14(x)
        x = self.resid15(x)
        
        x = self.head1(x)
        x = self.bn(x)
        x = self.relu(x)
        stereo_features = self.head2(x)

        return stereo_features, attention, w
