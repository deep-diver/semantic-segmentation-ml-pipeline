"""
Script to generate TFRecord shards from the Sidewalks dataset as shown in this
blog post: https://huggingface.co/blog/fine-tune-segformer.

The recommended way to obtain TFRecord shards is via an Apache Beam
Pipeline with an execution runner like Dataflow. Example:
https://github.com/GoogleCloudPlatform/practical-ml-vision-book/blob/master/05_create_dataset/jpeg_to_tfrecord.py.

Usage:
    python create_tfrecords --batch_size 16

References:

    * https://github.com/sayakpaul/TF-2.0-Hacks/blob/master/Cats_vs_Dogs_TFRecords.ipynb
    * https://www.tensorflow.org/tutorials/images/segmentation
"""

import argparse
import math
import os
from typing import Tuple

import datasets
import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image

RESOLUTION = 256


def load_sidewalks_dataset(args):
    hf_dataset_identifier = "segments/sidewalk-semantic"
    ds = datasets.load_dataset(hf_dataset_identifier)

    ds = ds.shuffle(seed=1)
    ds = ds["train"].train_test_split(test_size=args.split, seed=args.seed)
    train_ds = ds["train"]
    val_ds = ds["test"]

    return train_ds, val_ds


def resize_img(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(image, (RESOLUTION, RESOLUTION))
    label = tf.image.resize(label[..., None], (RESOLUTION, RESOLUTION))
    label = tf.squeeze(label, -1)
    return image, label


def process_image(image: Image, label: Image) -> Tuple[tf.Tensor, tf.Tensor]:
    image = np.array(image)
    label = np.array(label)

    image = tf.convert_to_tensor(image)
    label = tf.convert_to_tensor(label)

    image, label = resize_img(image, label)
    image, label = normalize_img(image, label)
    return image, label


def _bytestring_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def create_tfrecord(image: Image, label: Image):
    image, label = process_image(image, label)

    image = tf.io.serialize_tensor(image)
    label = tf.io.serialize_tensor(label)

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image": _bytestring_feature([image.numpy()]),
                "label": _bytestring_feature([label.numpy()]),
            }
        )
    ).SerializeToString()


def write_tfrecords(root_dir, dataset, split, batch_size):
    print(f"Preparing TFRecords for split: {split}.")

    for step in tqdm.tnrange(int(math.ceil(len(dataset) / batch_size))):
        temp_ds = dataset[step * batch_size : (step + 1) * batch_size]
        shard_size = len(temp_ds["pixel_values"])
        filename = os.path.join(
            root_dir, "{}-{:02d}-{}.tfrec".format(split, step, shard_size)
        )

        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size):
                image = temp_ds["pixel_values"][i]
                label = temp_ds["label"][i]
                example = create_tfrecord(image, label)
                out_file.write(example)
            print("Wrote file {} containing {} records".format(filename, shard_size))


def main(args):
    train_ds, val_ds = load_sidewalks_dataset(args)
    print("Dataset loaded from HF.")

    if not os.path.exists(args.root_tfrecord_dir):
        os.makedirs(args.root_tfrecord_dir, exist_ok=True)

    write_tfrecords(args.root_tfrecord_dir, train_ds, "train", args.batch_size)
    write_tfrecords(args.root_tfrecord_dir, val_ds, "val", args.batch_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", help="Train and test split.", default=0.2, type=float
    )
    parser.add_argument(
        "--seed",
        help="Seed to be used while performing train-test splits.",
        default=2022,
        type=int,
    )
    parser.add_argument(
        "--root_tfrecord_dir",
        help="Root directory where the TFRecord shards will be serialized.",
        default="sidewalks-tfrecords",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        help="Number of samples to process in a batch before serializing a single TFRecord shard.",
        default=32,
        type=int,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
