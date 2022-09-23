import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2

_RAW_IMG_SIZE = 256
_INPUT_IMG_SIZE = 128

_IMAGE_KEY = "image"
_IMAGE_SHAPE_KEY = "image_shape"
_LABEL_KEY = "label"
_LABEL_SHAPE_KEY = "label_shape"


def _transformed_name(key: str) -> str:
    return key + "_xf"


# output should have the same keys as inputs
def preprocess(inputs):
    images = tf.reshape(inputs[_IMAGE_KEY], [_RAW_IMG_SIZE, _RAW_IMG_SIZE, 3])
    labels = tf.reshape(inputs[_LABEL_KEY], [_RAW_IMG_SIZE, _RAW_IMG_SIZE, 1])

    return {
        _IMAGE_KEY: images,
        _IMAGE_SHAPE_KEY: inputs[_IMAGE_SHAPE_KEY],
        _LABEL_KEY: labels,
        _LABEL_SHAPE_KEY: inputs[_LABEL_SHAPE_KEY],
    }


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
      inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
      Map from string feature key to transformed feature operations.
    """
    # print(inputs)
    outputs = {}

    features = tf.map_fn(preprocess, inputs)

    features[_IMAGE_KEY] = tf.image.resize(features[_IMAGE_KEY], [_INPUT_IMG_SIZE, _INPUT_IMG_SIZE])
    features[_LABEL_KEY] = tf.image.resize(features[_LABEL_KEY], [_INPUT_IMG_SIZE, _INPUT_IMG_SIZE])

    image_features = mobilenet_v2.preprocess_input(features[_IMAGE_KEY])

    outputs[_transformed_name(_IMAGE_KEY)] = image_features
    outputs[_transformed_name(_LABEL_KEY)] = features[_LABEL_KEY]

    return outputs