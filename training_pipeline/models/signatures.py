from typing import Dict

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras.applications import mobilenet_v2

from .utils import transformed_name
from .common import IMAGE_KEY, LABEL_KEY, CONCRETE_INPUT
from .hyperparams import INPUT_IMG_SIZE


def _serving_preprocess(string_input):
    """
    _serving_preprocess turns base64 encoded string data into a
    string type of Tensor. Then it is decoded as 3 channel based
    uint8 type of Tensor. Finally, it is normalized and resized
    to the size the model expects.
    """
    decoded_input = tf.io.decode_base64(string_input)
    decoded = tf.io.decode_jpeg(decoded_input, channels=3)
    resized = tf.image.resize(decoded, size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE))
    normalized = mobilenet_v2.preprocess_input(resized)
    return normalized


@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def _serving_preprocess_fn(string_input):
    """
    _serving_preprocess_fn simply iteratively applies _serving_preprocess
    function to each entry of the batch of requests. So, the output is the
    preprocessed batch of requests being ready to be fed into the model.
    """
    decoded_images = tf.map_fn(
        _serving_preprocess, string_input, dtype=tf.float32, back_prop=False
    )
    return {IMAGE_KEY: decoded_images}


def model_exporter(model: tf.keras.Model):
    """
    model_exporter will be assigned to the "serving_defaults" signature.
    that means, when clients sends requests to the endpoint of this model
    hosted on TF Serving, the serving_fn which model_exporter returns will
    be the first point where the request payloads are going to be accepted.
    """
    m_call = tf.function(model.call).get_concrete_function(
        tf.TensorSpec(
            shape=[None, INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3],
            dtype=tf.float32,
            name=IMAGE_KEY,
        )
    )

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_fn(string_input):
        """
        serving_fn simply preprocesses the request payloads with the
        _serving_preprocess_fn, then the preprocessed data will be fed
        into the model. The model outputs predictions of the request p
        ayloads, then it will be returned back to the client after app
        lying postprocess of tf.math.argmax to the outputs(logits)
        """
        images = _serving_preprocess_fn(string_input)
        logits = m_call(**images)
        seg_mask = tf.math.argmax(logits, -1)
        return {"seg_mask": seg_mask}

    return serving_fn

"""
    Note that transform_features_signature and tf_examples_serving_signature
    functions exist only for model evaluation purposes with Evaluator component.
"""


def transform_features_signature(
    tf_transform_output: tft.TFTransformOutput
):
    """
    transform_features_signature simply returns a function that transforms
    any data of the type of tf.Example which is denoted as the type of sta
    ndard_artifacts.Examples in TFX. The purpose of this function is to ap
    ply Transform Graph obtained from Transform component to the data prod
    uced by ImportExampleGen. This function will be used in the Evaluator
    component, so the raw evaluation inputs from ImportExampleGen can be a
    pporiately transformed that the model could understand.
    """

    # basically, what Transform component emits is a SavedModel that knows
    # how to transform data. transform_features_layer() simply returns the
    # layer from the Transform.
    tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")]
    )
    def serve_tf_examples_fn(serialized_tf_examples):
        """
        raw_feature_spec returns a set of feature maps(dict) for the input
        TFRecords based on the knowledge that Transform component has lear
        ned(learn doesn't mean training here). By using this information,
        the raw data from ImportExampleGen could be parsed with tf.io.parse
        _example utility function.

        Then, it is passed to the model.tft_layer, so the final output we
        get is the transformed data of the raw input.
        """
        feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = tft_layer(parsed_features)

        return transformed_features

    return serve_tf_examples_fn


def tf_examples_serving_signature(model, tf_transform_output):
    """
    tf_examples_serving_signature simply returns a function that performs
    data transformation(preprocessing) and model prediction in a sequential
    manner. How data transformation is done is idential to the process of
    transform_features_signature function.
    """

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")]
    )
    def serve_tf_examples_fn(
        serialized_tf_example: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_tf_example, feature_spec)

        outputs = model(parsed_features)

        return {LABEL_KEY: outputs}

    return serve_tf_examples_fn
