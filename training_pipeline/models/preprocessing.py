import tensorflow as tf
from tensorflow.keras.applications import mobilenet

_IMAGE_KEY = "image"
_LABEL_KEY = "label"
_IMAGE_SIZE = (128, 128)

def _transformed_name(key: str) -> str:
    return key + "_xf"

def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
      inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}

    image_features = tf.map_fn(
        lambda x: tf.io.decode_png(x[0], channels=3),
        inputs[_IMAGE_KEY],
        fn_output_signature=(tf.uint8),
    )
    image_features = tf.image.resize(image_features, _IMAGE_SIZE)
    image_features = mobilenet.preprocess_input(image_features)

    label_features = tf.map_fn(
        lambda x: tf.io.decode_png(x[0], channels=1),
        inputs[_LABEL_KEY],
        fn_output_signature=(tf.uint8),
    )
    label_features = tf.image.resize(label_features, _IMAGE_SIZE)

    outputs[_transformed_name(_IMAGE_KEY)] = image_features
    outputs[_transformed_name(_LABEL_KEY)] = label_features

    return outputs
