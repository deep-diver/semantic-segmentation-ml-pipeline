There are two scripts to create TFRecords for [sidewalk dataset](https://huggingface.co/datasets/segments/sidewalk-semantic) hosted in 🤗 Dataset Hub. They are basically same except for one thing. `create_tfrecords.py` creates TFRecords with the features encoded in numeric data type. On the other hand, `create_tfrecords_str.py` creates TFRecords with the features encoded in string data type. 
- This is to demonstrate how to create such a dataset because string encoding is better in terms of compression.
- However, we use the dataset created from `create_tfrecords.py` since we couldn't find a way to extract string encoded dataset in `Transform` component yet. We will explore the solution afterwards.

`create_tfrecords.py` supports some flags:
- `--split`: train and test split. The default value is `0.2`.
- `--seed`: seed to be used while performing train-test splits. The default value is `2022`.
- `--root_tfrecord_dir`: root directory where the TFRecord shards will be serialized. The default value is `sidewalks-tfrecords`.
- `--batch_size`: number of samples to process in a batch before serializing a single TFRecord shard. The default value is `32.
- `--resize`: width and height size the image will be resized to. The default value is `None` (set if noe value is given), and that means no resizing will be applied.

we created two different datasets with the following CLIs:
```console
# full resolution dataset
$ python create_tfrecords.py

# lowered resolution dataset
$ python create_tfrecords.py --resize 256
```

In Google Cloud environment, the default machine couldn't not handle a large size of dataset, so we make `256` sized dataset. The original size dataset is there to experiment that how `ImportExampleGen` and `Transform` components could delegate their jobs to **Dataflow**. Each component could be run with and without **Dataflow** integration independently. We are enabling **Dataflow** for `Transform` and `ImportExampleGen` at the same time because the fact that `ImportExampleGen` couldn't not handle the full resolution dataset suggests that `Transform` can't handle it too without **Dataflow**.

`create_tfrecords_pets.py` creates TFRecords for the [pets dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/). To run it first download
the prerequisites:

```bash
curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```