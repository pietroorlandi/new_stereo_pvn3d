import tensorflow.keras as keras
from tensorflow.keras import activations
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    MultiHeadAttention,
    UpSampling2D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import Input

# from lib.net.utils import match_choose
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
    ContextAdj,
)


def build_res_initial(x, filters, name="res_in"):
    # resnet block where dimension does not change.
    # The skip connection is just simple identity connection
    # we will have 3 blocks and then input will be added

    inputs = Input(shape=x.shape, name="x")
    f1, f2 = filters
    f1 = max(f1, 8)
    f2 = max(f2, 8)

    # second block # bottleneck (but size kept same with padding)
    x = Conv2D(
        f2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_regularizer=l2(0.001),
        name="{}_focus_conv_2".format(name),
    )(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block activation after adding the input
    x = Conv2D(
        f2,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        kernel_regularizer=l2(0.001),
        name="{}_conv_3".format(name),
    )(x)
    x = BatchNormalization()(x)

    # add the input
    output = Activation(activations.relu)(x)
    model = keras.Model(inputs=input, outputs=output, name="res_initial_model")

    return model


def build_resnet(channel_multiplier, deep=False, name="resnet"):
    base_channels = [1, 2, 4, 8, 16, 32]
    f10, f11, f12, f13, f14, f15 = [x * 2 * channel_multiplier for x in base_channels]
    f20, f21, f22, f23, f24, f25 = [x * 4 * channel_multiplier for x in base_channels]
    input = Input(shape=(480, 640, 3), name="rgb")
    # write all the code with layers
    x = ResInitial(filters=(f10, f20), name=f"{name}_initial")(input)
    if deep:
        x = ResIdentity(filters=(f10, f20), name=f"{name}_1_1")(x)
    x_1 = x
    x = ResConv(s=2, filters=(f11, f21), name=f"{name}_1_2")(x)
    if deep:
        x = ResIdentity(filters=(f11, f21), name=f"{name}_2_1")(x)
        x = ResIdentity(filters=(f11, f21), name=f"{name}_2_2")(x)
    x_2 = x
    x = ResConv(s=2, filters=(f12, f22), name=f"{name}_2_3")(x)
    # 3rd stage
    if deep:
        x = ResIdentity(filters=(f12, f22), name=f"{name}_3_1")(x)
        x = ResIdentity(filters=(f12, f22), name=f"{name}_3_2")(x)
    x = ResIdentity(filters=(f12, f22), name=f"{name}_3_3")(x)
    x_3 = x
    x = ResConv(s=2, filters=(f13, f23), name=f"{name}_3_4")(x)
    # 4th stage
    if deep:
        x = ResIdentity(filters=(f13, f23), name=f"{name}_4_1")(x)
        x = ResIdentity(filters=(f13, f23), name=f"{name}_4_2")(x)
        x = ResIdentity(filters=(f13, f23), name=f"{name}_4_3")(x)
    x = ResIdentity(filters=(f13, f23), name=f"{name}_4_4")(x)
    x_4 = x
    x = ResConv(s=2, filters=(f14, f24), name=f"{name}_4_5")(x)
    # 5th stage
    if deep:
        x = ResIdentity(filters=(f14, f24), name=f"{name}_5_1")(x)
        x = ResIdentity(filters=(f14, f24), name=f"{name}_5_2")(x)
        x = ResIdentity(filters=(f14, f24), name=f"{name}_5_3")(x)
        x = ResIdentity(filters=(f14, f24), name=f"{name}_5_4")(x)
    x = ResIdentity(filters=(f14, f24), name=f"{name}_5_5")(x)
    x_5 = x

    output = [x_1, x_2, x_3, x_4, x_5]
    model = keras.Model(inputs=input, outputs=output, name="resnet_model")

    return model


def build_decoder_model(channel_multiplier, deep=False):
    input_l = Input(shape=(5,), name="rgb_features_l")
    input_r = Input(shape=(5,), name="rgb_features_r")
    # input = Input(shape=[input_l.shape, input_r.shape], name='rgb')
    f_r_1, f_r_2, f_r_3, f_r_4, f_r_5 = input_r
    f_l_1, f_l_2, f_l_3, f_l_4, f_l_5 = input_l

    base_channels = [1, 2, 4, 8, 16, 32]
    _, _, f12, f13, f14, f15 = [x * channel_multiplier for x in base_channels]
    _, _, f22, f23, f24, f25 = [x * 2 * channel_multiplier for x in base_channels]

    x, w = StereoAttention(channels=8, width=4, depth=f25)(f_l_5, f_r_5)
    if deep:
        x = ResIdentity(filters=(f15, f25), name="d5_1")(x)
        x = ResIdentity(filters=(f15, f25), name="d5_2")(x)
        x = ResIdentity(filters=(f15, f25), name="d5_3")(x)
    x = ResIdentity(filters=(f15, f25), name="d5_4")(x)
    x = ResUp(s=2, filters=(f15, f24), name="d5_5")(x)

    x_skip = x
    x, w = StereoAttention(
        channels=8, width=4, query=x, previous_weights=w, dilation=3
    )(f_l_4, f_r_4)
    x = tf.concat([x, x_skip], axis=-1)
    x = ResReduce(x, f24)
    if deep:
        x = ResIdentity(filters=(f14, f24), name="d4_1")(x)
        x = ResIdentity(filters=(f14, f24), name="d4_2")(x)
        x = ResIdentity(filters=(f14, f24), name="d4_3")(x)
    x = ResIdentity(filters=(f14, f24), name="d4_4")(x)
    x = ResUp(s=2, filters=(f14, f23), name="d4_5")(x)

    x_skip = x
    x, w = StereoAttention(width=8, query=x, previous_weights=w, dilation=2)(
        f_l_3, f_r_3
    )
    x = tf.concat([x, x_skip], axis=-1)
    x = ResReduce(f23)(x)

    if deep:
        x = ResIdentity(filters=(f23, f23), name="d3_1")(x)
        x = ResIdentity(filters=(f23, f23), name="d3_2")(x)
    x = ResIdentity(filters=(f23, f23), name="d3_3")(x)
    x = ResIdentity(s=2, filters=(f23, f23), name="d3_4")(x)

    x_skip = x
    x, w = StereoAttention(
        channels=8, width=16, query=x, previous_weights=w, dilation=2
    )(f_l_2, f_r_2)
    x = tf.concat([x, x_skip], axis=-1)
    x = ResReduce(f23)(x)

    if deep:
        x = ResIdentity(filters=(f23, f23), name="d2_1")(x)
        x = ResIdentity(filters=(f23, f23), name="d2_2")(x)
    x = ResIdentity(filters=(f23, f23), name="d2_3")(x)
    x = ResUp(s=2, filters=(f23, f23), name="d2_4")(x)

    x_skip = x
    x, w = StereoAttention(
        channels=8, width=32, query=x, previous_weights=w, dilation=2
    )(f_l_1, f_r_1)
    x = tf.concat([x, x_skip], axis=-1)
    x = ResReduce(f23)(x)

    if deep:
        x = ResIdentity(filters=(f23, f23), name="d1_1")(x)
    output = ResIdentity(filters=(f23, f23), name="d1_2")(x)

    # output = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input)
    model = keras.Model(inputs=[input_l, input_r], outputs=output, name="decoder_model")

    return model


class StereoNet(keras.Model):
    def __init__(
        self,
        *,
        use_disparity,
        relative_disparity,
        channel_multiplier,
        base_channels,
        context_adj=False,
        **kwargs,
    ):
        super(StereoNet, self).__init__()
        # self.params = SVe2eParams()
        # self.sve2e = SVe2e(self.params, (480, 640))
        # self.identityR = self.sve2e.build_decoder_model(x_shape=(480, 640, 3))
        # self.identityR.summary()
        # self.identityL = self.sve2e.build_decoder_model(x_shape=(480, 640, 3))
        # self.identityL.summary()
        # self.res_init = ResInitial(filters=(4,8))
        # self.identityR = build_decoder_model()
        # self.identityR.summary()
        # self.identityL = build_decoder_model()
        # self.identityL.summary()

        # self.identityR = Prova(name='prova')
        # # self.identityR.summary()
        # self.identityL = Prova(name='prova2')
        # self.identityL.summary()
        self.use_disparity = use_disparity
        self.relative_disparity = relative_disparity
        self.context_adjustment = context_adj
        self.channel_multiplier = channel_multiplier
        self.resnet_lr = build_resnet(self.channel_multiplier)
        # self.resnet_r = build_resnet(self.channel_multiplier)

        # self.decoder = build_decoder_model(channel_multiplier=self.channel_multiplier)
        # self.trivial_conv = tf.keras.layers.Conv2D(1, 1, padding='same')
        # self.dummy_backbone = tf.keras.layers.Conv2D(32, 1)
        # self.identityR = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name ='idr')
        # self.identityL = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same',
        #                                         activation='relu', name='idl')
        # self.dummy_backbone = tf.keras.layers.Conv2D(32, 1, name = 'dummy_backbone')
        """Decoder Initialization"""
        self.base_channels = base_channels
        _, _, f12, f13, f14, f15 = [
            x * self.channel_multiplier for x in self.base_channels
        ]
        _, _, f22, f23, f24, f25 = [
            x * 2 * self.channel_multiplier for x in self.base_channels
        ]
        self.attention1 = StereoAttention(channels=8, width=4, depth=f25)
        self.resid1 = ResIdentity(filters=(f15, f25), name="d5_1")
        self.resid2 = ResIdentity(filters=(f15, f25), name="d5_2")
        self.resid3 = ResIdentity(filters=(f15, f25), name="d5_3")
        self.resid4 = ResIdentity(filters=(f15, f25), name="d5_4")
        self.resup1 = ResUp(s=2, filters=(f15, f24), name="d5_5")
        self.attention2 = StereoAttention(channels=4, width=4, dilation=3)
        self.resred1 = ResReduce(f24)
        self.resid5 = ResIdentity(filters=(f14, f24), name="d4_1")
        self.resid6 = ResIdentity(filters=(f14, f24), name="d4_2")
        self.resid7 = ResIdentity(filters=(f14, f24), name="d4_3")
        self.resid8 = ResIdentity(filters=(f14, f24), name="d4_4")
        self.resid9 = ResUp(s=2, filters=(f14, f23), name="d4_5")
        self.attention3 = StereoAttention(channels=4, width=8, dilation=2)
        self.resred2 = ResReduce(f23)
        self.resid3_3 = ResIdentity(filters=(f23, f23), name="d3_3")
        self.resid3_4 = ResUp(s=2, filters=(f23, f23), name="d3_4")
        self.attention4 = StereoAttention(channels=4, width=16, dilation=2)
        self.resred3 = ResReduce(f23)
        self.resid2_3 = ResIdentity(filters=(f23, f23), name="d2_3")
        self.resup2 = ResUp(s=2, filters=(f23, f23), name="d2_4")
        self.attention5 = StereoAttention(channels=4, width=32, dilation=2)
        self.resred4 = ResReduce(f23)
        self.resid1_2 = ResIdentity(filters=(f23, f23), name="d1_2")

        """End"""

        self.head1 = tf.keras.layers.Conv2D(
            16,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="deph_conv_1",
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        n_cls = 6  # + background + depth
        self.head2 = tf.keras.layers.Conv2D(
            1 + n_cls + 1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name="deph_conv_2",
        )

        self.cal1 = ContextAdj(filter=32)
        self.cal2 = ContextAdj(filter=64)

    def call(self, inputs, training=None):
        # Trivial model:
        # x = tf.concat([inputs[0], inputs[1]], axis=-1)
        # outputs = self.trivial_conv(x)

        # depth with dummy backbone:
        x_l = inputs[0]
        x_r = inputs[1]  # another layer
        baseline = inputs[2]
        focal_length = inputs[3]

        # x_l = self.res_init(x_l)
        # x_r = self.identityR(x_r)
        # x_l = self.identityL(x_l)

        # changed to same resnet for left and right
        f_l_1, f_l_2, f_l_3, f_l_4, f_l_5 = self.resnet_lr(x_l)
        f_r_1, f_r_2, f_r_3, f_r_4, f_r_5 = self.resnet_lr(x_r)

        deep = True
        x, w, w_1 = self.attention1(f_l_5, f_r_5)
        if deep:
            x = self.resid1(x)
            x = self.resid2(x)
            x = self.resid3(x)
        x = self.resid4(x)
        x = self.resup1(x)

        x_skip = x
        x, w, w_2 = self.attention2(
            f_l_4,
            f_r_4,
            query=x,
            pw=w,
        )
        x = tf.concat([x, x_skip], axis=-1)
        x = self.resred1(x)
        if deep:
            x = self.resid5(x)
            x = self.resid6(x)
            x = self.resid7(x)
        x = self.resid8(x)
        x = self.resid9(x)

        x_skip = x
        x, w, w_3 = self.attention3(
            f_l_3,
            f_r_3,
            query=x,
            pw=w,
        )
        x = tf.concat([x, x_skip], axis=-1)
        x = self.resred2(x)

        # if deep:
        #     x = ResIdentity(filters=(f23, f23), name="d3_1")(x)
        #     x = ResIdentity(filters=(f23, f23), name="d3_2")(x)
        x = self.resid3_3(x)
        x = self.resid3_4(x)

        x_skip = x
        x, w, w_4 = self.attention4(
            f_l_2,
            f_r_2,
            query=x,
            pw=w,
        )
        x = tf.concat([x, x_skip], axis=-1)
        x = self.resred3(x)

        # if deep:
        #     x = ResIdentity(filters=(f23, f23), name="d2_1")(x)
        #     x = ResIdentity(filters=(f23, f23), name="d2_2")(x)
        x = self.resid2_3(x)
        x = self.resup2(x)

        x_skip = x
        x, w, w_5 = self.attention5(
            f_l_1,
            f_r_1,
            query=x,
            pw=w,
        )
        x = tf.concat([x, x_skip], axis=-1)
        x = self.resred4(x)

        # if deep:
        #     x = ResIdentity(filters=(f23, f23), name="d1_1")(x)
        x = self.resid1_2(x)
        # before_head = x

        """End"""

        # x = self.dummy_backbone(x)
        x = self.head1(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        outputs = self.head2(x)

        # if debug:
        #    if self.context_adjustment:
        #        adj_outputs = self.cal1(tf.cast(x_l, dtype=tf.float32), outputs)
        #        adj_outputs = self.cal2(adj_outputs, outputs)
        #        return adj_outputs, w_1, w_2, w_3, w_4, w_5, before_head, f_l_5, f_r_5, f_l_4, f_r_4
        #    else:
        #        return outputs, w_1, w_2, w_3, w_4, w_5, before_head, f_l_5, f_r_5, f_l_4, f_r_4

        if self.context_adjustment:
            raise NotImplementedError
            adj_outputs = self.cal1(tf.cast(x_l, dtype=tf.float32), outputs)
            adj_outputs = self.cal2(adj_outputs, outputs)
            return adj_outputs
        else:
            if self.use_disparity:
                disp = tf.nn.relu(outputs[..., :1])
                if self.relative_disparity:
                    disp = (
                        disp * tf.shape(inputs[0])[2]
                    )  # BHWC -> *W converts to absolute disparity

                depth = tf.math.divide_no_nan(baseline * focal_length, disp)
                depth = tf.clip_by_value(depth, 0.0, 50.0)
                outputs = tf.concat([depth, outputs[..., 1:]], axis=-1)
            return outputs
