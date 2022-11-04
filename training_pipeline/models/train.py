from typing import List

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import DataAccessor, FnArgs
from tfx_bsl.tfxio import dataset_options

from .common import IMAGE_KEY, LABEL_KEY, NUM_LABELS
from .hyperparams import EPOCHS, EVAL_BATCH_SIZE, TRAIN_BATCH_SIZE
from .signatures import (
    model_exporter,
    tf_examples_serving_signature,
    transform_features_signature,
)
from .unet import build_model
from .utils import transformed_name

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
            batch_size=batch_size, label_key=transformed_name(LABEL_KEY), shuffle=is_train
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
        batch_size=TRAIN_BATCH_SIZE,
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        is_train=False,
        batch_size=EVAL_BATCH_SIZE,
    )
    model = build_model(
        transformed_name(IMAGE_KEY), transformed_name(LABEL_KEY), NUM_LABELS
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=EPOCHS,
    )

    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures={
            "serving_default": model_exporter(model),
            "transform_features": transform_features_signature(
                model, tf_transform_output
            ),
            "from_examples": tf_examples_serving_signature(model, tf_transform_output),
        },
    )
