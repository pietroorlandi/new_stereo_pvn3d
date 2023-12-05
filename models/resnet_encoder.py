import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from .stereo_layers import (
    ResInitial,
    ResUp,
    ResConv,
    ResReduce,
    ResIdentity,
    StereoAttention,
    DisparityAttention,
    ContextAdj,
)
from dataclasses import dataclass
#from typing import Dict, List

@dataclass
class ResnetEncoderParams:
    channel_multiplier: int
    base_channels: list
    resnet_input_shape: list
    deep: bool


class ResnetEncoder:
    def __init__(self, params: ResnetEncoderParams):
        self.params = params

    def build_resnet(self, name="resnet_encoder"):
        
        f10, f11, f12, f13, f14, f15 = [x * self.params.channel_multiplier for x in self.params.base_channels]
        f20, f21, f22, f23, f24, f25 = [x * 4 * self.params.channel_multiplier for x in self.params.base_channels]
        input = Input(shape=self.params.resnet_input_shape, name="rgb")
        # write all the code with layers
        x = ResInitial(filters=(f10, f20), name=f"{name}_initial")(input)
        if self.params.deep:
            x = ResIdentity(filters=(f10, f20), name=f"{name}_1_1")(x)
        x_1 = x
        x = ResConv(s=2, filters=(f11, f21), name=f"{name}_1_2")(x)
        if self.params.deep:
            x = ResIdentity(filters=(f11, f21), name=f"{name}_2_1")(x)
            x = ResIdentity(filters=(f11, f21), name=f"{name}_2_2")(x)
        x_2 = x
        x = ResConv(s=2, filters=(f12, f22), name=f"{name}_2_3")(x)
        # 3rd stage
        if self.params.deep:
            x = ResIdentity(filters=(f12, f22), name=f"{name}_3_1")(x)
            x = ResIdentity(filters=(f12, f22), name=f"{name}_3_2")(x)
        x = ResIdentity(filters=(f12, f22), name=f"{name}_3_3")(x)
        x_3 = x
        x = ResConv(s=2, filters=(f13, f23), name=f"{name}_3_4")(x)
        # 4th stage
        if self.params.deep:
            x = ResIdentity(filters=(f13, f23), name=f"{name}_4_1")(x)
            x = ResIdentity(filters=(f13, f23), name=f"{name}_4_2")(x)
            x = ResIdentity(filters=(f13, f23), name=f"{name}_4_3")(x)
        x = ResIdentity(filters=(f13, f23), name=f"{name}_4_4")(x)
        x_4 = x
        x = ResConv(s=2, filters=(f14, f24), name=f"{name}_4_5")(x)
        # 5th stage
        if self.params.deep:
            x = ResIdentity(filters=(f14, f24), name=f"{name}_5_1")(x)
            x = ResIdentity(filters=(f14, f24), name=f"{name}_5_2")(x)
            x = ResIdentity(filters=(f14, f24), name=f"{name}_5_3")(x)
            x = ResIdentity(filters=(f14, f24), name=f"{name}_5_4")(x)
        x = ResIdentity(filters=(f14, f24), name=f"{name}_5_5")(x)
        x_5 = x

        output = [x_1, x_2, x_3, x_4, x_5]
        model = tf.keras.Model(inputs=input, outputs=output, name=f"{name}_model")

        return model
