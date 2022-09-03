from typing import List, Dict, Tuple

import os
import json
import datetime
import absl
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tfx.components.trainer.fn_args_utils import FnArgs

class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
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

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result    

_CONCRETE_INPUT = "pixel_values"
_IMAGE_SHAPE = (128, 128)
_TRAIN_BATCH_SIZE = 64
_EVAL_BATCH_SIZE = 64
_EPOCHS = 2
_LR = 0.00006

def INFO(text: str):
    absl.logging.info(text)

def _serving_normalize_img(
    img, mean=feature_extractor.image_mean, std=feature_extractor.image_std
):
    # Scale to the value range of [0, 1] first and then normalize.
    img = img / 255
    mean = tf.constant(mean)
    std = tf.constant(std)
    return (img - mean) / std

def _serving_preprocess(string_input):
    decoded_input = tf.io.decode_base64(string_input)
    decoded = tf.io.decode_jpeg(decoded_input, channels=3)
    decoded = decoded / 255
    resized = tf.image.resize(decoded, size=_IMAGE_SHAPE)
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
            shape=[None, 128, 128, 3], dtype=tf.float32, name=_CONCRETE_INPUT
        )
    )

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_fn(string_input):
        images = _serving_preprocess_fn(string_input)
        logits = m_call(**images)

        return {"logits": logits}

    return serving_fn

def _parse_tfr(proto):
    feature_description = {
        "image": tf.io.VarLenFeature(tf.float32),
        "image_shape": tf.io.VarLenFeature(tf.int64),
        "label": tf.io.VarLenFeature(tf.float32),
        "label_shape": tf.io.VarLenFeature(tf.int64),
    }
    rec = tf.io.parse_single_example(proto, feature_description)
    image_shape = tf.sparse.to_dense(rec["image_shape"])
    image = tf.reshape(tf.sparse.to_dense(rec["image"]), image_shape)
    label_shape = tf.sparse.to_dense(rec["label_shape"])
    label = tf.reshape(tf.sparse.to_dense(rec["label"]), label_shape)

    return {"pixel_values": image, "labels": label}

def _preprocess(example_batch):
    images = example_batch["pixel_values"]
    images = tf.transpose(images, perm=[0, 2, 3, 1]) # (batch_size, height, width, num_channels)
    labels = tf.expand_dims(example_batch["labels"], -1) # Adds extra dimension, otherwise tf.image.resize won't work.
    labels = tf.transpose(labels, perm=[0, 1, 2, 3]) # So, that TF can evaluation the shapes.

    images = tf.image.resize(images, _IMAGE_SHAPE)
    labels = tf.image.resize(labels, _IMAGE_SHAPE)

    labels = tf.squeeze(labels, -1)
    
    return images, labels

def _input_fn(
    file_pattern: List[str],
    batch_size: int = 32,
    is_train: bool = False,
) -> tf.data.Dataset:
    INFO(f"Reading data from: {file_pattern}")

    dataset = tf.data.TFRecordDataset(
        tf.io.gfile.glob(file_pattern[0] + ".gz"),
        num_parallel_reads=tf.data.AUTOTUNE,
        compression_type="GZIP"
    ).map(_parse_tfr, num_parallel_calls=tf.data.AUTOTUNE)

    if is_train:
        dataset = dataset.shuffle(batch_size * 2)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.map(_preprocess)
    return dataset

def _build_model(num_labels:int) -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False

    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),   # 32x32 -> 64x64
    ]    

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

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
        filters=num_labels, kernel_size=3, strides=2,
        padding='same',
        name="labels")  #64x64 -> 128x128

    x = last(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def run_fn(fn_args: FnArgs):
    train_dataset = _input_fn(
        fn_args.train_files,
        is_train=True,
        batch_size=_TRAIN_BATCH_SIZE,
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        is_train=False,
        batch_size=_EVAL_BATCH_SIZE,
    )

    num_labels = 35
    model = _build_model(num_labels)

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=_EPOCHS,
    )

    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures=_model_exporter(model)
    )
