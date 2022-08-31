from typing import List, Dict, Tuple

import os
import json
import datetime
import absl
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tfx.components.trainer.fn_args_utils import FnArgs

from transformers import (
    SegformerFeatureExtractor,
    TFSegformerForSemanticSegmentation,
    create_optimizer,
)
from datasets import load_metric
from transformers.keras_callbacks import KerasMetricCallback

from huggingface_hub import cached_download, hf_hub_url

_CONCRETE_INPUT = "pixel_values"
_IMAGE_SHAPE = (512, 512)
_TRAIN_BATCH_SIZE = 64
_EVAL_BATCH_SIZE = 64
_EPOCHS = 2
_LR = 0.00006

feature_extractor = SegformerFeatureExtractor()

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
    resized = tf.image.resize(decoded, size=_IMAGE_SHAPE)
    normalized = _serving_normalize_img(resized)
    normalized = tf.transpose(
        normalized, (2, 0, 1)
    )  # Since HF models are channel-first.
    return normalized

@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def _serving_preprocess_fn(string_input):
    decoded_images = tf.map_fn(
        _serving_preprocess, string_input, dtype=tf.float32, back_prop=False
    )
    return {_CONCRETE_INPUT: decoded_images}


def _model_exporter(model: tf.keras.Model):
    m_call = tf.function(model.call).get_concrete_function(
        tf.TensorSpec(
            shape=[None, 3, 512, 512], dtype=tf.float32, name=_CONCRETE_INPUT
        )
    )

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_fn(string_input):
        images = _serving_preprocess_fn(string_input)
        logits = m_call(**images).logits

        # Transpose to have the shape (batch_size, height/4, width/4, num_labels)
        logits = tf.transpose(logits, [0, 2, 3, 1])

        upsampled_logits = tf.image.resize(
            logits,
            images.size[
                ::-1
            ],  # We reverse the shape of `image` because `image.size` returns width and height.
        )

        pred_seg = tf.math.argmax(upsampled_logits, axis=-1)[0]
        return {"pred_seg": pred_seg}

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

    images = tf.transpose(images, perm=[0, 3, 1, 2]) # (batch_size, num_channels, height, width)
    labels = tf.squeeze(labels, -1)

    return {"pixel_values": images, "labels": labels}

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

def _get_label_info() -> Tuple[Dict, Dict, int]:
    hf_dataset_identifier = "segments/sidewalk-semantic"
    repo_id = f"datasets/{hf_dataset_identifier}"
    filename = "id2label.json"

    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename)), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    num_labels = len(id2label)

    return id2label, label2id, num_labels

def _build_metric_callback(eval_dataset, num_labels) -> KerasMetricCallback:
    metric = load_metric("mean_iou")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # logits are of shape (batch_size, num_labels, height, width), so
        # we first transpose them to (batch_size, height, width, num_labels)
        logits = tf.transpose(logits, perm=[0, 2, 3, 1])
        # scale the logits to the size of the label
        logits_resized = tf.image.resize(
            logits,
            size=tf.shape(labels)[1:],
            method="bilinear",
        )
        # compute the prediction labels and compute the metric
        pred_labels = tf.argmax(logits_resized, axis=-1)
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=-1,
            reduce_labels=feature_extractor.reduce_labels,
        )
        return {"val_" + k: v for k, v in metrics.items()}


    metric_callback = KerasMetricCallback(
        metric_fn=compute_metrics,
        eval_dataset=eval_dataset,
        batch_size=_EVAL_BATCH_SIZE,
        label_cols=["labels"],
    )

    return metric_callback

def _build_model(id2label, label2id, num_labels) -> tf.keras.Model:
    model_checkpoint = "nvidia/mit-b0"
    model = TFSegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # Will ensure the segmentation specific components are reinitialized.
    )

    optimizer = Adam(learning_rate=_LR)
    model.compile(optimizer)
    model.summary(print_fn=INFO)

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

    id2label, label2id, num_labels = _get_label_info()

    model = _build_model(id2label, label2id, num_labels)
    metric_callback = _build_metric_callback(eval_dataset, num_labels)

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        callbacks=[metric_callback],
        epochs=_EPOCHS,
    )
    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
       # signatures=_model_exporter(model)
    )
