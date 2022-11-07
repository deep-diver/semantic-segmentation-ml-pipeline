Files in this directory are used to create TFRecords based on [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) and [Sidewalk Dataset](https://huggingface.co/datasets/segments/sidewalk-semantic). We have built and verified ML pipeline with **Sidewalk Dataset** first, but we moved to use **Oxford-IIIT Pet Dataset** to acheive good enough results with the basic UNet architecture. However, you could generate and test TFRecords from **Sidewalk Dataset** if your model is enough to handle more complex data. Below organizes the files into two sub-categories based on the different dataset.
- **Oxford-IIIT Pet Dataset**
  - `create_tfrecords_str.py` creates TFRecords with the features encoded in numeric data type such as `float` and `int`.
  - To run `create_tfrecords_str.py`, first download the prerequisites:
    ```bash
    wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    tar xf images.tar.gz
    tar xf annotations.tar.gz
    ```

- **Sidewalk Dataset**
  - `create_tfrecords.py` creates TFRecords with the features encoded in numeric data type such as `float` and `int`. 
  - `create_tfrecords_str.py` creates TFRecords with the features encoded in string data type. This is to demonstrate how to create such a dataset because string encoding is better in terms of compression.

`create_tfrecords_pets.py`, `create_tfrecords.py`, and `create_tfrecords_str.py` supports common flags:
- `--split`: train and test split. The default value is `0.2`.
- `--seed`: seed to be used while performing train-test splits. The default value is `2022`.
- `--root_tfrecord_dir`: root directory where the TFRecord shards will be serialized. The default value is `sidewalks-tfrecords`.
- `--batch_size`: number of samples to process in a batch before serializing a single TFRecord shard. The default value is `32.
- `--resize`: width and height size the image will be resized to. The default value is `None` (set if noe value is given), and that means no resizing will be applied.

---

### NOTE on Sidewalk Dataset

We created two different TFRecords in different image sizes from **Sidewalk Dataset** with the following CLIs:
```console
# full resolution dataset
$ python create_tfrecords.py

# lowered resolution dataset
$ python create_tfrecords.py --resize 256
```

In Google Cloud environment, the default machine couldn't not handle a large size of dataset, so we make `256` sized dataset. The original size dataset is there to experiment that how `ImportExampleGen` and `Transform` components could delegate their jobs to **Dataflow**. Each component could be run with and without **Dataflow** integration independently. We are enabling **Dataflow** for `Transform` and `ImportExampleGen` at the same time because the fact that `ImportExampleGen` couldn't not handle the full resolution dataset suggests that `Transform` can't handle it too without **Dataflow**.