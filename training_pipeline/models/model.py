from typing import List, Dict, Tuple

import absl
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras.optimizers import Adam
from tfx_bsl.tfxio import dataset_options
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor

from unet import build_model

_CONCRETE_INPUT = "pixel_values"

_TRAIN_LENGTH = 800
_EVAL_LENGTH = 200
_TRAIN_BATCH_SIZE = 64
_EVAL_BATCH_SIZE = 64
_EPOCHS = 2

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
    model = build_model(
        _transformed_name(_IMAGE_KEY),
        _transformed_name(_IMAGE_KEY),
        num_labels,
    )

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
