import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras.optimizers import Adam

_INPUT_IMG_SIZE = 128
_LR = 0.00006

"""
    _build_model builds a UNET model. The implementation codes are
    borrowed from the [TF official tutorial on Semantic Segmentation]
    (https://www.tensorflow.org/tutorials/images/segmentation)
"""

def build_model(input_name, label_name, num_labels) -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[_INPUT_IMG_SIZE, _INPUT_IMG_SIZE, 3], include_top=False
    )

    # Use the activations of these layers
    layer_names = [
        "block_1_expand_relu",  # 64x64
        "block_3_expand_relu",  # 32x32
        "block_6_expand_relu",  # 16x16
        "block_13_expand_relu",  # 8x8
        "block_16_project",  # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False

    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),  # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(
        shape=[_INPUT_IMG_SIZE, _INPUT_IMG_SIZE, 3], name=input_name
    )

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=num_labels,
        kernel_size=3,
        strides=2,
        padding="same",
        name=label_name,
    )  # 64x64 -> 128x128

    x = last(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer=Adam(learning_rate=_LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # by suppling "accuracy", Keras will automatically infer the appropriate variant.
        # in this case, sparse_categorical_accuracy will be chosen.
        metrics=["accuracy"],
    )
    return model


"""
    InstanceNormalization class and upsample function are
    borrowed from pix2pix in [TensorFlow Example repository](
    https://github.com/tensorflow/examples/tree/master/tensorflow_examples/models/pix2pix)
"""


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1.0, 0.02),
            trainable=True,
        )

        self.offset = self.add_weight(
            name="offset", shape=input_shape[-1:], initializer="zeros", trainable=True
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def upsample(filters, size, norm_type="batchnorm", apply_dropout=False):
    """Upsamples an input.
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_dropout: If True, adds the dropout layer
    Returns:
      Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if norm_type.lower() == "batchnorm":
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == "instancenorm":
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result