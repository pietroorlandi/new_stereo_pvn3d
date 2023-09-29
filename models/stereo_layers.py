import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Activation,
    Conv2D,
    Add,
    UpSampling2D,
    Concatenate,
)
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l2


class ResInitial(keras.layers.Layer):
    def __init__(self, filters, name="resinit"):
        print(f"type(self): {type(self)}")
        super().__init__()
        self.f1, self.f2 = filters
        self.f1 = max(self.f1, 8)
        self.f2 = max(self.f2, 8)

        # second block # bottleneck (but size kept same with padding)
        self.conv_1 = Conv2D(
            self.f2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=l2(0.001),
            name="{}_focus_conv_2".format(name),
        )
        self.bn1 = BatchNormalization()
        self.activation1 = Activation(activations.relu)

        # third block activation after adding the input
        self.conv_2 = Conv2D(
            self.f2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=l2(0.001),
            name="{}_conv_3".format(name),
        )
        self.bn2 = BatchNormalization()
        self.activation2 = Activation(activations.relu)

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv_2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        return x


class ResIdentity(keras.layers.Layer):
    def __init__(self, filters, name="residentity"):
        super().__init__()
        self.f1, self.f2 = filters
        self.f1 = max(self.f1, 8)
        self.f2 = max(self.f2, 8)

        # second block # bottleneck (but size kept same with padding)
        self.conv_1 = Conv2D(
            self.f1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=l2(0.001),
            name="{}_conv_1".format(name),
        )

        self.bn1 = BatchNormalization()
        self.activation1 = Activation(activations.relu)

        # third block activation after adding the input
        self.conv_2 = Conv2D(
            self.f1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=l2(0.001),
            name="{}_focus_conv_2".format(name),
        )
        self.bn2 = BatchNormalization()
        self.activation2 = Activation(activations.relu)

        self.conv_3 = Conv2D(
            self.f2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=l2(0.001),
            name="{}_conv_3".format(name),
        )
        self.bn3 = BatchNormalization()
        self.add = Add()
        self.activation3 = Activation(activations.relu)

    def call(self, inputs):
        x_skip = inputs

        x = self.conv_1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)

        x = self.conv_2(x)
        x = self.bn2(x)
        x = self.activation2(x)

        x = self.conv_3(x)
        x = self.bn3(x)
        #print(f'ResIdentity - Output Shape: {x.shape}')
        #print(f'x before resid {x_skip.shape} - x after resid {x.shape}')
        outputs = self.add([x, x_skip])
        outputs = self.activation3(outputs)
        return outputs


class ResReduce(keras.layers.Layer):
    def __init__(self, depth, name="resreduce"):
        super().__init__()
        self.depth = depth
        # second block # bottleneck (but size kept same with padding)
        self.conv_1 = Conv2D(
            self.depth,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=l2(0.001),
        )
        
        self.bn1 = BatchNormalization()
        self.activation1 = Activation(activations.relu)

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn1(x)
        #print(f'ResReduce - Output Shape: {x.shape}')
        outputs = self.activation1(x)
        return outputs


class ResUp(keras.layers.Layer):
    def __init__(self, s, filters, name="resup"):
        super().__init__()
        self.f1, self.f2 = filters
        self.s = s
        self.f1 = max(self.f1, 8)
        self.f2 = max(self.f2, 8)

        # first block
        self.up1 = UpSampling2D((self.s, self.s), interpolation="bilinear")
        self.conv_1 = Conv2D(
            self.f1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=l2(0.001),
            name="{}_conv_stride_1".format(name),
        )
        self.bn1 = BatchNormalization()
        self.activation1 = Activation(activations.relu)

        # second block
        self.conv_2 = Conv2D(
            self.f1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=l2(0.001),
            name="{}_focus_conv_2".format(name),
        )
        self.bn2 = BatchNormalization()
        self.activation2 = Activation(activations.relu)

        # third block
        self.conv_3 = Conv2D(
            self.f2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=l2(0.001),
            name="{}_conv_3".format(name),
        )
        self.bn3 = BatchNormalization()

        # shortcut
        self.up_skip = UpSampling2D((self.s, self.s), interpolation="bilinear")
        self.conv_skip = Conv2D(
            self.f2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=l2(0.001),
            name="{}_shortcut".format(name),
        )
        self.bn_skip = BatchNormalization()

        # add
        self.add = Add()
        self.activation3 = Activation(activations.relu)

    def call(self, inputs):
        x_skip = inputs
        x = self.up1(inputs)
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.activation1(x)

        x = self.conv_2(x)
        x = self.bn2(x)
        x = self.activation2(x)

        x = self.conv_3(x)
        x = self.bn3(x)

        x_skip = self.up_skip(x_skip)
        x_skip = self.conv_skip(x_skip)
        x_skip = self.bn_skip(x_skip)
        #print(f'ResUp - Output Shape: {x.shape}')
        outputs = self.add([x, x_skip])
        outputs = self.activation3(outputs)
        return outputs


class ResConv(keras.layers.Layer):
    def __init__(self, s, filters, name="resconv"):
        super().__init__()
        self.f1, self.f2 = filters
        self.s = s
        self.f1 = max(self.f1, 8)
        self.f2 = max(self.f2, 8)

        # first block
        self.conv1 = Conv2D(
            self.f1,
            kernel_size=(1, 1),
            strides=(self.s, self.s),
            padding="valid",
            kernel_regularizer=l2(0.001),
            name="{}_conv_stride_1".format(name),
        )
        # when s = 2 then it is like downsizing the feature map
        self.bn1 = BatchNormalization()
        self.activation1 = Activation(activations.relu)

        # second block # bottleneck (but size kept same with padding)
        self.conv2 = Conv2D(
            self.f1,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=l2(0.001),
            name="{}_focus_conv_2".format(name),
        )
        self.bn2 = BatchNormalization()
        self.activation2 = Activation(activations.relu)

        # third block
        self.conv3 = Conv2D(
            self.f2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=l2(0.001),
            name="{}_conv_3".format(name),
        )
        self.bn3 = BatchNormalization()

        # shortcut
        self.conv_skip = Conv2D(
            self.f2,
            kernel_size=(1, 1),
            strides=(self.s, self.s),
            padding="valid",
            kernel_regularizer=l2(0.001),
            name="{}_shortcut_conv_4".format(name),
        )
        self.bn_skip = BatchNormalization()

        # add
        self.add = Add()

        self.activation3 = Activation(activations.relu)

    def call(self, inputs):
        x_skip = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x_skip = self.conv_skip(x_skip)
        x_skip = self.bn_skip(x_skip)
        #print(f'ResConv - Output Shape: {x.shape}')
        outputs = self.add([x, x_skip])
        outputs = self.activation3(outputs)
        return outputs


class EfficientAttention(keras.layers.Layer):
    def __init__(self, channels, width=5, depth=16, dilation=1, name="attention"):
        super().__init__()
        self.w = width
        self.depth = depth
        self.dilation = dilation
        self.channels = channels
        self.rolling_index = [x * self.dilation for x in range(self.w)]
        self.up_pw = UpSampling2D((2, 2), interpolation="bilinear")
        self.conv_w = Conv2D(
            self.channels,
            kernel_size=(1, width),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=l2(0.001),
        )
        self.activation_w = Activation(activations.softmax)
        self.bn = BatchNormalization()
        self.activation1 = Activation(activations.tanh)

    """ Efficient Attention layer with a query and a previous weights input. """

    def call(self, input_l, input_r, query=None, pw=None, debug=False):
        shiftmap = tf.concat(
            [
                tf.stack(
                    [tf.roll(input_l, i, axis=-2), tf.roll(input_r, -i, axis=-2)],
                    axis=-1,
                )
                for i in self.rolling_index
            ],
            axis=-1,
        )

        if pw is not None:
            pw = self.up_pw(pw)
            query = tf.einsum(
                "nijm,nijkm->nijk", pw, shiftmap
            )  # the query is the shiftmap weighted by the previous weights
            w = 3
            # roll 1 pixel
            shiftmap = tf.concat(
                [
                    tf.stack(
                        [tf.roll(query, i, axis=-2), tf.roll(query, -i, axis=-2)],
                        axis=-1,
                    )
                    for i in (0, 1)
                ],
                axis=-1,
            )

        if query is None:
            query = tf.concat(
                [input_l, input_r], axis=-1
            )  # the query is the concatenation of the two images
        weights = self.conv_w(query)
        weights = self.activation_w(weights)
        w_toprint = weights
        if weights.shape[-1] != shiftmap.shape[-1]:
            print(
                "Please change the activation parameter channels in ",
                shiftmap.shape[-1],
            )

        x = tf.einsum("nijm,nijkm->nijk", weights, shiftmap)
        x = self.bn(x)
        x = self.activation1(x)
        if pw is not None:
            weights = tf.einsum("nijl,nijm->nijlm", pw, weights)
            weights = tf.reshape(
                weights,
                (-1, weights.shape[1], weights.shape[2], weights.shape[3] * 2, 2),
            )
            weights = tf.reduce_prod(weights, axis=-1)
        return x, weights, w_toprint


class StereoAttention(keras.layers.Layer):
    def __init__(self, channels, width=5, depth=16, dilation=1, name="attention"):
        super(StereoAttention, self).__init__()
        self.w = width
        # self.query = query
        self.depth = depth
        # self.pw = previous_weights
        self.dilation = dilation
        self.channels = channels
        self.rolling_index = [x * self.dilation for x in range(self.w)]
        self.up_pw = UpSampling2D((2, 2), interpolation="bilinear")
        self.conv_w = Conv2D(
            self.channels,
            kernel_size=(1, width),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=l2(0.001),
        )
        self.activation_w = Activation(activations.softmax)
        self.bn = BatchNormalization()
        self.activation1 = Activation(activations.tanh)

    def call(self, input_l, input_r, query=None, pw=None, debug=False):
        if debug:
            print(
                "A new Attention layer begins ----------------------------------------"
            )
        # print('rolling index', self.rolling_index[0])
        # print('input r', input_r)
        # self.rolling_index = [0, 1, 2, 3]
        # print('prova', tf.concat(
        #     [tf.stack([tf.roll(input_l, i, axis=-2), tf.roll(input_r, -i, axis=-2)], axis=-1) for i in self.rolling_index], axis=-1))
        shiftmap = tf.concat(
            [
                tf.stack(
                    [tf.roll(input_l, i, axis=-2), tf.roll(input_r, -i, axis=-2)],
                    axis=-1,
                )
                for i in self.rolling_index
            ],
            axis=-1,
        )

        if debug:
            print(f"shiftmap: {shiftmap.shape}")
        if pw is not None:
            # upsample the weights
            if debug:
                print(f"previous weights: {pw.shape}")
            pw = self.up_pw(pw)
            # apply previous weights to the finer features to select the approximate alignment
            if debug:
                print(f"previous weights upsampled: {pw.shape}")
                print(f"query: {query.shape}")
            query = tf.einsum("nijm,nijkm->nijk", pw, shiftmap)

            w = 3
            # roll 1 pixel
            shiftmap = tf.concat(
                [
                    tf.stack(
                        [tf.roll(query, i, axis=-2), tf.roll(query, -i, axis=-2)],
                        axis=-1,
                    )
                    for i in (0, 1)
                ],
                axis=-1,
            )

        if query is None:
            if debug:
                print("Query is None ----------------------------------------")
            query = tf.concat([input_l, input_r], axis=-1)
        if debug:
            channels = shiftmap.shape[-1]
            print(f"channels are {channels} while input channels are {self.channels}")
        weights = self.conv_w(query)
        weights = self.activation_w(weights)
        w_toprint = weights
        if debug:
            print(f"weights: {weights.shape}")
            print(f"shiftmap: {shiftmap.shape}")
            print(
                f"weights shape is {weights.shape} while shiftmap shape is {shiftmap.shape}"
            )
        if weights.shape[-1] != shiftmap.shape[-1]:
            print(
                "Please change the activation parameter channels in ",
                shiftmap.shape[-1],
            )

        x = tf.einsum("nijm,nijkm->nijk", weights, shiftmap)
        x = self.bn(x)
        x = self.activation1(x)
        if pw is not None:
            weights = tf.einsum("nijl,nijm->nijlm", pw, weights)
            weights = tf.reshape(
                weights,
                (-1, weights.shape[1], weights.shape[2], weights.shape[3] * 2, 2),
            )
            weights = tf.reduce_prod(weights, axis=-1)
        # x = tf.squeeze(x, axis=-1)
        return x, weights, w_toprint

class DisparityAttention(tf.keras.layers.Layer):
    def __init__(self, max_disparity=3, **kwargs):
        super().__init__(**kwargs)
        self.max_disparity = max_disparity
        self.conv = tf.keras.layers.Conv2D(filters=max_disparity, kernel_size=(3, max_disparity + 2), padding="same")
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.upsample = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")

    def call(self, inputs, previous_weights=None, **kwargs):
        # Assume inputs is a list [left_image, right_image]
        left_image, right_image = inputs
        # roll right image to the left by disparity_max/2 pixels (optional to center the kernel)
        if previous_weights is not None:
            right_image = tf.roll(right_image, shift=-self.max_disparity//2, axis=2)
            #else roll -1
        else:
            right_image = tf.roll(right_image, shift=-1, axis=2)

        # add previous weights to the new list of weights and to the right image
        new_previous_weights = []
        if previous_weights is not None:
            n = len(previous_weights)+1
            attended_right = tf.zeros_like(right_image)
            for j, pw in enumerate(previous_weights):
                pw = self.upsample(pw)
                new_previous_weights.append(pw) 
                for i in range(self.max_disparity):
                    # select the weights for disparity i and apply them to the right image shifted by i pixels
                    pw_right = pw[:, :, :, i:i+1] * tf.roll(right_image, shift=-i*(2**(n-j)), axis=2)
                    right_image += pw_right

        # stack left and right images
        stacked = tf.concat([left_image, right_image], axis=-1) # (B, H, W, C)
        # Compute disparity
        disparity_map = self.conv(stacked) #(B, H, W, max_disparity)
        # Apply softmax to get the weights
        weights = self.softmax(disparity_map) 
        # add also the computed new weights, multiplied by 2 
        new_previous_weights.append(weights)

        # batch, height, width, _ = tf.shape(right_image)
        attended_right = tf.zeros_like(right_image)

        for i in range(self.max_disparity):
            # select the weights for disparity i and apply them to the right image shifted by i pixels
            weighted_right = weights[:, :, :, i:i+1] * tf.roll(right_image, shift=-i, axis=2)
            attended_right += weighted_right

        # somma previouse weights + new_weights
        x = tf.concat([left_image, attended_right], axis=-1)
        #print(f'DisparityAttention - Output Shape: {x.shape}')
        return x, new_previous_weights
        # if previous_weights is not None:
        #     print(f'previous weights {pw.shape}, current_weights {weights.shape}')
        #     new_weights = pw * weights
        #     print('new weights shape is ',new_weights.shape)
        # else:
        #     new_weights = weights
        # print('weights', tf.shape(weights))
        #weights sum with pw * 2
        
        # if previous_weights is not None: 
        #     # print('prev weights', tf.shape(previous_weights))
        #     new_weights = self.upsample(previous_weights)
        #     # print('upsampled weights', tf.shape(new_weights))
        #     new_weights = weights * new_weights# tf.einsum("nijl,nijm->nijlm", new_weights, weights)
        #     # print('upsampled weights', tf.shape(new_weights))
        #     # new_weights = tf.reshape(
        #     #     new_weights,
        #     #     (-1, new_weights.shape[1], new_weights.shape[2], new_weights.shape[3] * 2, 2),
        #     # )
        #     # new_weights = tf.reduce_prod(new_weights, axis=-1)
        # else:
        #     new_weights = weights

        # print('right_image', tf.shape(right_image))
        # print('weights', tf.shape(weights))

class ContextAdj(keras.layers.Layer):
    def __init__(self, filter, name="contextadj"):
        super().__init__()
        self.conv1 = Conv2D(filter, kernel_size=(3, 3), padding="same")
        self.conv2 = Conv2D(1, kernel_size=(3, 3), padding="same")
        self.relu = Activation(activations.relu)
        self.concat = Concatenate(axis=-1)

    def call(self, left_im, raw_dpt):
        x = self.concat([left_im, raw_dpt])
        x = self.conv1(x)
        x = self.relu(x)
        outputs = self.conv2(x)
        return outputs