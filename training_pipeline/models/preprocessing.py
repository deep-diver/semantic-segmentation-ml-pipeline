import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from .utils import transformed_name
from .common import IMAGE_KEY, LABEL_KEY

def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
      inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
      Map from string feature key to transformed feature operations.
    """
    # print(inputs)
    outputs = {}

    image_features = mobilenet_v2.preprocess_input(inputs[IMAGE_KEY])

    outputs[IMAGE_KEY] = image_features
    outputs[LABEL_KEY] = inputs[LABEL_KEY]

    return outputs
