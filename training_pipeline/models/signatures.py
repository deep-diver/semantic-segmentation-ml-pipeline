from typing import Dict

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras.applications import mobilenet_v2

from .utils import transformed_name
from .common import LABEL_KEY, CONCRETE_INPUT
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
    return {CONCRETE_INPUT: decoded_images}


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
            name=CONCRETE_INPUT,
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
