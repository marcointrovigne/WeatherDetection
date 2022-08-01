"""EfficientNet models for Keras.
# Reference paper
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks]
  (https://arxiv.org/abs/1905.11946) (ICML 2019)
# Reference implementation
- [TensorFlow]
  (https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

from tensorflow.keras.models import Model
import tensorflow as tf
from __init__ import correct_pad
from __init__ import get_submodules_from_kwargs
from tensorflow.keras.layers import MaxPool2D, Dropout, Dense, Input, GlobalAveragePooling2D, \
    BatchNormalization, Activation

backend = None
layers = None
models = None
keras_utils = None

BASE_WEIGHTS_PATH = (
    'https://github.com/Callidior/keras-applications/'
    'releases/download/efficientnet/')

WEIGHTS_HASHES = {
    'b0': ('e9e877068bd0af75e0a36691e03c072c',
           '345255ed8048c2f22c793070a9c1a130'),
    'b1': ('8f83b9aecab222a9a2480219843049a1',
           'b20160ab7b79b7a92897fcb33d52cc61'),
    'b2': ('b6185fdcd190285d516936c09dceeaa4',
           'c6e46333e8cddfa702f4d8b8b6340d70'),
    'b3': ('b2db0f8aac7c553657abb2cb46dcbfbb',
           'e0cf8654fad9d3625190e30d70d0c17d'),
    'b4': ('ab314d28135fe552e2f9312b31da6926',
           'b46702e4754d2022d62897e0618edc7b'),
    'b5': ('8d60b903aff50b09c6acf8eaba098e09',
           '0a839ac36e46552a881f2975aaab442f'),
    'b6': ('a967457886eac4f5ab44139bdd827920',
           '375a35c17ef70d46f9c664b03b4437f2'),
    'b7': ('e964fd6e26e9a4c144bcb811f2a10f20',
           'd55674cc46b805f4382d18bc08ed43c1')
}

DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def swish(x):
    """
    Swish activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The Swish activation: `x * sigmoid(x)`.
    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if backend.backend() == 'tensorflow':
        try:
            return backend.tf.nn.swish(x)
        except AttributeError:
            pass

    return x * backend.sigmoid(x)


def block(inputs, activation_fn=swish, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True):
    """
    A mobile inverted residual block.
        # Arguments
            inputs: input tensor.
            activation_fn: activation function.
            drop_rate: float between 0 and 1, fraction of the input units to drop.
            name: string, block label.
            filters_in: integer, the number of input filters.
            filters_out: integer, the number of output filters.
            kernel_size: integer, the dimension of the convolution window.
            strides: integer, the stride of the convolution.
            expand_ratio: integer, scaling coefficient for the input filters.
            se_ratio: float between 0 and 1, fraction to squeeze the input filters.
            id_skip: boolean.
        # Returns
            output tensor for the block.
        """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=name + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = layers.Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, kernel_size),
                                 name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = layers.DepthwiseConv2D(kernel_size,
                               strides=strides,
                               padding=conv_pad,
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = layers.Activation(activation_fn, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        if bn_axis == 1:
            se = layers.Reshape((filters, 1, 1), name=name + 'se_reshape')(se)
        else:
            se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        se = layers.Conv2D(filters_se, 1,
                           padding='same',
                           activation=activation_fn,
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_reduce')(se)
        se = layers.Conv2D(filters, 1,
                           padding='same',
                           activation='sigmoid',
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_expand')(se)
        if backend.backend() == 'theano':
            # For the Theano backend, we have to explicitly make
            # the excitation weights broadcastable.
            se = layers.Lambda(
                lambda x: backend.pattern_broadcast(x, [True, True, True, False]),
                output_shape=lambda input_shape: input_shape,
                name=name + 'se_broadcast')(se)
        x = layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    x = layers.Conv2D(filters_out, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=name + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            x = layers.Dropout(drop_rate,
                               noise_shape=(None, 1, 1, 1),
                               name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')

    return x


def decay(epoch, lr):
    initial_lrate = lr
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def build_branch(name, dense, inputs, conv_filters, dropout, add_layer, division_layer):

    x = Dense(conv_filters)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if add_layer:
        x = Dense(conv_filters / division_layer)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(dense)(x)
    x = Activation("softmax", name=name)(x)

    return x


def EfficientNet(base_efficientnet,
                 width_coefficient,
                 depth_coefficient,
                 conv_filters,
                 dropout,
                 add_layer,
                 division_layer,
                 loss,
                 metrics,
                 lr,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 activation_fn=swish,
                 model_name='efficientnet',
                 weights=None,
                 input_tensor=None,
                 **kwargs):
    """
    Instantiates the EfficientNet architecture using given scaling coefficients.

        # Arguments
            base_efficientnet: bool, if true use original efficientnet backbone.
            width_coefficient: float, scaling coefficient for network width.
            depth_coefficient: float, scaling coefficient for network depth.
            conv_filters: integer, select number of neurons in the dense layer.
            dropout: float, dropout rate in final classifier layer.
            add_layer: bool, decide number of dense layers.
            division_layer: integer, division rate with multiple dense layers.
            loss: loss function.
            metrics: metrics for each output.
            lr: learning rate for Adam optimizer.
            blocks_args: list of dicts, parameters to construct block modules.
            drop_connect_rate: float, dropout rate at skip connections.
            depth_divisor: integer, a unit of network width.
            activation_fn: activation function.
            model_name: string, model name.
            weights: one of `None` (random initialization),
                  'imagenet' (pre-training on ImageNet),
                  or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor
                (i.e. output of `layers.Input()`)
                to use as image input for the model.
        # Returns
            A Keras model instance.
        # Raises
            ValueError: in case of invalid argument for `weights`,
                or invalid input shape.
        """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    x = img_input
    if base_efficientnet:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                                 name='stem_conv_pad')(x)
        x = layers.Conv2D(round_filters(32), 3,
                          strides=2,
                          padding='valid',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name='stem_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
        x = layers.Activation(activation_fn, name='stem_activation')(x)
    else:
        x = layers.Conv2D(round_filters(16), (7, 7), strides=2, padding='valid',
                          kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
        x = layers.Activation(activation_fn)(x)
        x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)
        x = layers.Conv2D(round_filters(32), (1, 1), padding='same', strides=(1, 1),
                          kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
        x = layers.Activation(activation_fn)(x)
        x = layers.Conv2D(round_filters(32), 3, padding='same', strides=(1, 1),
                          kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
        x = layers.Activation(activation_fn)(x)
        x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    # Build blocks
    from copy import deepcopy
    blocks_args = deepcopy(blocks_args)

    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(x, activation_fn, drop_connect_rate * b / blocks,
                      name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
            b += 1

    # Build top
    x = layers.Conv2D(round_filters(1280), 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = layers.Activation(activation_fn, name='top_activation')(x)

    x = GlobalAveragePooling2D()(x)
    daytime_branch = build_branch('daytime_output', 2, x, conv_filters, dropout, add_layer, division_layer)
    precipitation_branch = build_branch('precipitation_output', 4, x, conv_filters, dropout, add_layer, division_layer)
    fog_branch = build_branch('fog_output', 3, x, conv_filters, dropout, add_layer, division_layer)
    roadState_branch = build_branch('roadState_output', 4, x, conv_filters, dropout, add_layer, division_layer)
    sidewalkState_branch = build_branch('sidewalkState_output', 3, x, conv_filters, dropout, add_layer, division_layer)
    infrastructure_branch = build_branch('infrastructure_output', 3, x, conv_filters, dropout, add_layer,
                                         division_layer)

    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs=inputs,
                  outputs=[daytime_branch, precipitation_branch, fog_branch, roadState_branch, sidewalkState_branch,
                           infrastructure_branch],
                  name=model_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999,
                                         epsilon=1e-08)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Load weights.
    if weights == 'imagenet':
        file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
        file_hash = WEIGHTS_HASHES[model_name[-2:]][1]
        file_name = model_name + file_suff
        weights_path = keras_utils.get_file(file_name,
                                            BASE_WEIGHTS_PATH + file_name,
                                            cache_subdir='models',
                                            file_hash=file_hash)
        model.load_weights(weights_path, by_name=True)
    elif weights is not None:
        model.load_weights(weights)

    return model
