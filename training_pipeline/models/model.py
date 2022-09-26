from typing import List, Dict, Tuple

import absl
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras.optimizers import Adam
from tfx_bsl.tfxio import dataset_options
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor

_CONCRETE_INPUT = "pixel_values"
_INPUT_IMG_SIZE = 128
_TRAIN_LENGTH = 800
_EVAL_LENGTH = 200
_TRAIN_BATCH_SIZE = 64
_EVAL_BATCH_SIZE = 64
_EPOCHS = 2
_LR = 0.00006

_IMAGE_KEY = "image"
_LABEL_KEY = "label"


def INFO(text: str):
    absl.logging.info(text)


def _transformed_name(key: str) -> str:
    return key + "_xf"


"""
    _serving_preprocess, _serving_preprocess_fn, and 
    _model_exporter functions are defined to provide pre-
    processing capabilities when the model is served.
"""


def _serving_preprocess(string_input):
    decoded_input = tf.io.decode_base64(string_input)
    decoded = tf.io.decode_jpeg(decoded_input, channels=3)
    decoded = decoded / 255
    resized = tf.image.resize(decoded, size=(_INPUT_IMG_SIZE, _INPUT_IMG_SIZE))
    return resized


@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def _serving_preprocess_fn(string_input):
    decoded_images = tf.map_fn(
        _serving_preprocess, string_input, dtype=tf.float32, back_prop=False
    )
    return {_CONCRETE_INPUT: decoded_images}


def _model_exporter(model: tf.keras.Model):
    m_call = tf.function(model.call).get_concrete_function(
        tf.TensorSpec(
            shape=[None, _INPUT_IMG_SIZE, _INPUT_IMG_SIZE, 3],
            dtype=tf.float32,
            name=_CONCRETE_INPUT,
        )
    )

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_fn(string_input):
        images = _serving_preprocess_fn(string_input)
        logits = m_call(**images)
        seg_mask = tf.math.argmax(logits, -1)
        return {"seg_mask": seg_mask}

    return serving_fn


"""
    _input_fn reads TFRecord files with the given file_pattern passed down 
    from the upstream TFX component, Transform. The file patterns are inter
    nally determined by Transform component, and they are automatically acce
    ssible through fn_args.train_files and fn_args.eval_files in the run_fn 
    function. Assume the dataset is already transformed appropriately. 
"""


def _input_fn(
    file_pattern: List[str],
    data_accessor: DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    is_train: bool = False,
    batch_size: int = 200,
) -> tf.data.Dataset:
    """
    DataAccessor is responsible for accessing the data on disk, and 
    TensorFlowDatasetOptions provides options for TFXIO's TensorFlowDataset.

    the factory function tf_dataset_factory takes three inputs of List[str],
    dataset_options.TensorFlowDatasetOptions, and schema_pb2.Schema. The 
    schema_pb2.Schema holds the information how the TFRecords are structured,
    like what kind of features are accessible. In this case, there are two
    features of image_xf and label_xf, and they are the preprocessed results
    from Transform component.
    - Transform component simply preprocess the raw inputs, then returns the
    transformed output in TFRecord format. tf_dataset_factory is just a handy
    method to access TFRecord, and it is not strongly coupled with Transform 
    component.

    by giving label_key option in the TensorFlowDataset, the tf_dataset_factory
    outputs the dataset in the form of Tuple[Dict[str, Tensor], Tensor]. Here,
    the second term will hold label information, and the first term holds what
    ever the rest is in the dataset (image_xf for this case).

    then, in the modeling part, you should have input layers with the names 
    appearing in the first term Dict[str, Tensor]. For instance:
        inputs = tf.keras.layers.Input(..., name="image_xf")

    you could get rid of the label_key option, and it is totally optional. But
    then, you should have the output layer named with the label key. Otherwise,
    the model does not know which data from the Tuple to feed in the model. If
    you use label_key option, it it will be directly used in the output layer.
    """

    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_transformed_name(_LABEL_KEY)
        ),
        tf_transform_output.transformed_metadata.schema,
    )

    return dataset


def _build_model(num_labels) -> tf.keras.Model:
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
        shape=[_INPUT_IMG_SIZE, _INPUT_IMG_SIZE, 3], name=_transformed_name(_IMAGE_KEY)
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
        name=_transformed_name(_LABEL_KEY),
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


def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=True,
        batch_size=_TRAIN_BATCH_SIZE,
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=False,
        batch_size=_EVAL_BATCH_SIZE,
    )

    num_labels = 35
    model = _build_model(num_labels)

    model.fit(
        train_dataset,
        steps_per_epoch=_TRAIN_LENGTH // _TRAIN_BATCH_SIZE,
        validation_data=eval_dataset,
        validation_steps=_EVAL_LENGTH // _TRAIN_BATCH_SIZE,
        epochs=_EPOCHS,
    )

    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=_model_exporter(model)
    )
