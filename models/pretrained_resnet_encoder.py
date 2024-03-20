import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import layers
from tensorflow.keras.models import Model
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
from typing import List, Optional

#from typing import Dict, List

@dataclass
class ResnetEncoderParams:
    channel_multiplier: int
    base_channels: list
    resnet_input_shape: list
    deep: bool
    layers_to_save: list
    freeze: bool



class ResnetEncoder:
    def __init__(self, params: ResnetEncoderParams, freeze=False):
        # This model receive as inpuy the image h x w x 3 and gave as output 4 features extractions from a pretrained network
        self.params = params
        self.freeze = self.params.freeze
        self.layers_to_save = self.params.layers_to_save #[2, 13, 44, 94] # 349  

        # [2, 21/32/, 53/57/60/64/71, 98/109/116/127/131/138/142/149 ]
        self.resnet = tf.keras.applications.resnet_v2.ResNet101V2(include_top=False,
                                                    weights='imagenet',  # Load weights pre-trained on ImageNet.
                                                    input_shape=(self.params.resnet_input_shape[0], self.params.resnet_input_shape[1], 3),)  # Do not include the ImageNet classifier at the top.


    def build_resnet(self, name="resnet_encoder"):
        
        # input = Input(shape=self.params.resnet_input_shape, name="rgb")

        if self.freeze:
            # Optionally, you can freeze layers of the base model
            for layer in self.resnet.layers:
                layer.trainable = False
        selected_layers = [self.resnet.layers[layer] for layer in self.layers_to_save]
        print('selected layers', selected_layers)
        print([layer.output for layer in selected_layers])
        model = Model(inputs=self.resnet.input, outputs=[layer.output for layer in selected_layers], name=name)

        return model



    
