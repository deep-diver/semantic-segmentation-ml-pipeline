"""
Script to generate TFRecord shards from the Pets dataset as shown in this
tutorial: https://keras.io/examples/vision/oxford_pets_image_segmentation/.

The recommended way to obtain TFRecord shards is via an Apache Beam
Pipeline with an execution runner like Dataflow. Example:
https://github.com/GoogleCloudPlatform/practical-ml-vision-book/blob/master/05_create_dataset/jpeg_to_tfrecord.py.

Usage:
    python create_tfrecords_pets.py --batch_size 64
    python create_tfrecords_pets.py --resize 128 # without --resize flag, no resizing is applied

References:

    * https://github.com/GoogleCloudPlatform/practical-ml-vision-book/blob/master/05_create_dataset/05_split_tfrecord.ipynb
    * https://www.tensorflow.org/tutorials/images/segmentation
    * https://keras.io/examples/vision/oxford_pets_image_segmentation/
"""

import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image

RESOLUTION = 128
SEED = 2022


def load_paths(args) -> Tuple[List[str], List[str]]:
    input_img_paths = sorted(
        [
            os.path.join(args.input_dir, fname)
            for fname in os.listdir(args.input_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(args.target_dir, fname)
            for fname in os.listdir(args.target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )
    return input_img_paths, target_img_paths


def resize_img(
    image: tf.Tensor, label: tf.Tensor, resize: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(image, (resize, resize))
    label = tf.image.resize(label[..., None], (resize, resize))
    label = tf.squeeze(label, -1)
    label -= 1
    return image, label


def process_image(
    image_path: str, label_path: str, resize: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    image = Image.open(image_path).convert("RGB")
    label = Image.open(label_path).convert("L")

    image = np.array(image)
    label = np.array(label)

    image = tf.convert_to_tensor(image)
    label = tf.convert_to_tensor(label)

    if resize:
        image, label = resize_img(image, label, resize)

    return image, label


def split_paths(img_paths: List[str], target_paths: List[str], split: float):
    val_samples = int(img_paths * split)

    random.Random(SEED).shuffle(img_paths)
    random.Random(SEED).shuffle(target_paths)

    train_img_paths = img_paths[:-val_samples]
    train_target_paths = target_paths[:-val_samples]
    val_img_paths = img_paths[-val_samples:]
    val_target_paths = target_paths[-val_samples:]

    return train_img_paths, train_target_paths, val_img_paths, val_target_paths


def prepare_tf_dataset(
    img_paths: List[str], target_paths: List[str], batch_size: int
) -> tf.data.Dataset:
    tf_dataset = tf.data.Dataset.from_tensor_slices((img_paths, target_paths))
    return tf_dataset.batch(batch_size)


def get_tf_datasets(args) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    input_img_paths, target_img_paths = load_paths(args)
    train_img_paths, train_target_paths, val_img_paths, val_target_paths = split_paths(
        input_img_paths, target_img_paths, args.split
    )
    train_ds = prepare_tf_dataset(train_img_paths, train_target_paths, args.batch_size)
    val_ds = prepare_tf_dataset(val_img_paths, val_target_paths, args.batch_size)
    return train_ds, val_ds


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tfrecord(image: Image, label: Image, resize: int):
    image, label = process_image(image, label, resize)
    image_dims = image.shape
    label_dims = label.shape

    image = tf.reshape(image, [-1])  # flatten to 1D array
    label = tf.reshape(label, [-1])  # flatten to 1D array

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image": _float_feature(image.numpy()),
                "image_shape": _int64_feature(
                    [image_dims[0], image_dims[1], image_dims[2]]
                ),
                "label": _float_feature(label.numpy()),
                "label_shape": _int64_feature([label_dims[0], label_dims[1]]),
            }
        )
    ).SerializeToString()


def write_tfrecords(root_dir, dataset, split, resize):
    print(f"Preparing TFRecords for split: {split}.")

    for shard, (image_paths, label_paths) in enumerate(tqdm(dataset)):
        shard_size = image_paths.numpy().shape[0]
        filename = os.path.join(
            root_dir, "{}-{:02d}-{}.tfrec".format(split, shard, shard_size)
        )

        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size):
                img_path = image_paths["pixel_values"][i]
                label_path = label_paths["label"][i]
                example = create_tfrecord(img_path, label_path, resize)
                out_file.write(example)
            print("Wrote file {} containing {} records".format(filename, shard_size))


def main(args):
    train_ds, val_ds = get_tf_datasets(args)
    print("TensorFlow datasets loaded.")

    if not os.path.exists(args.root_tfrecord_dir):
        os.makedirs(args.root_tfrecord_dir, exist_ok=True)

    write_tfrecords(
        args.root_tfrecord_dir, train_ds, "train", args.batch_size, args.resize
    )
    write_tfrecords(args.root_tfrecord_dir, val_ds, "val", args.batch_size, args.resize)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", help="Train and test split.", default=0.2, type=float)
    parser.add_argument(
        "--input_dir",
        help="Path to the directory containing all images.",
        default="images/",
        type=str,
    )
    parser.add_argument(
        "--target_dir",
        help="Path to the directory containing all targets.",
        default="annotations/trimaps/",
        type=str,
    )
    parser.add_argument(
        "--root_tfrecord_dir",
        help="Root directory where the TFRecord shards will be serialized.",
        default="pets-tfrecords",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        help="Number of samples to process in a batch before serializing a single TFRecord shard.",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--resize",
        help="Width and height size the image will be resized to. No resizing will be applied when this isn't set.",
        type=int,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
