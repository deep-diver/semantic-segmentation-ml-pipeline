import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from .utils import transformed_name
from .common import IMAGE_KEY, IMAGE_SHAPE_KEY, LABEL_KEY, LABEL_SHAPE_KEY
from .hyperparams import INPUT_IMG_SIZE

# output should have the same keys as inputs
def preprocess(inputs):
    image_shape = inputs[IMAGE_SHAPE_KEY]
    label_shape = inputs[LABEL_SHAPE_KEY]

    images = tf.reshape(inputs[IMAGE_KEY], [image_shape[0], image_shape[1], 3])
    labels = tf.reshape(inputs[LABEL_KEY], [label_shape[0], label_shape[1], 1])

    return {
        IMAGE_KEY: images,
        IMAGE_SHAPE_KEY: inputs[IMAGE_SHAPE_KEY],
        LABEL_KEY: labels,
        LABEL_SHAPE_KEY: inputs[LABEL_SHAPE_KEY],
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

    features[IMAGE_KEY] = tf.image.resize(
        features[IMAGE_KEY], [INPUT_IMG_SIZE, INPUT_IMG_SIZE]
    )
    features[LABEL_KEY] = tf.image.resize(
        features[LABEL_KEY], [INPUT_IMG_SIZE, INPUT_IMG_SIZE]
    )

    image_features = mobilenet_v2.preprocess_input(features[IMAGE_KEY])

    outputs[transformed_name(IMAGE_KEY)] = image_features
    outputs[transformed_name(LABEL_KEY)] = features[LABEL_KEY]

    return outputs
