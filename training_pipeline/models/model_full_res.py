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
    _input_fn reads TFRecord files passed from the upstream 
    TFX component, Transform. Assume the dataset is already 
    transformed appropriately. 
"""


def _input_fn(
    file_pattern: List[str],
    data_accessor: DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    is_train: bool = False,
    batch_size: int = 200,
) -> tf.data.Dataset:
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
