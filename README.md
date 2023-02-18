![Python](https://img.shields.io/badge/python-3.9-blue.svg) [![TFX](https://img.shields.io/badge/TFX-1.9.1-orange)](https://www.tensorflow.org/tfx)
[![Validity Check for Training Pipeline](https://github.com/deep-diver/semantic-segmentation-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/deep-diver/semantic-segmentation-ml-pipeline/actions/workflows/ci.yml) [![Trigger Training Pipeline](https://github.com/deep-diver/semantic-segmentation-ml-pipeline/actions/workflows/cd-training-pipeline.yml/badge.svg)](https://github.com/deep-diver/semantic-segmentation-ml-pipeline/actions/workflows/cd-training-pipeline.yml)

# Semantic Sementation model within ML pipeline

<p align="center">
  <img src="https://i.ibb.co/7kF5kCK/semantic-seg-overview.png" width="80%" />
</p>

This repository shows how to build a Machine Learning Pipeline for [Semantic Segmentation](https://paperswithcode.com/task/semantic-segmentation) with [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx) and various GCP products such as [Vertex Pipeline](https://cloud.google.com/vertex-ai/docs/pipelines), [Vertex Training](https://cloud.google.com/vertex-ai/docs/training/custom-training), [Vertex Endpoint](https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api). Also, the ML pipeline contains a custom TFX component that is integrated with Hugging Face 🤗 Hub - `HFPusher`. `HFPusher` pushes a trained model to 🤗 Model Hub and, optionally `Gradio` application to 🤗 Space Hub with the latest model out of the box.

**NOTE**: We use U-NET based TensorFlow model from the [official tutorial](https://www.tensorflow.org/tutorials/images/segmentation). Since we implement an ML pipeline, U-NET like model could be a good starting point. Other SOTA models like [SegFormer from 🤗 `Transformers`](https://huggingface.co/transformers/v4.12.5/model_doc/segformer.html) or [DeepLabv3+](https://keras.io/examples/vision/deeplabv3_plus/) will be explored later.

**NOTE**: The aim of this project is not to serve the most SoTA segmentation model. Our main focus is to demonstrate how to build an end-to-end ML pipeline for semantic segmentation task instead.

**Update 17/02/2023**: This project received the [#TFCommunitySpotlight award](https://twitter.com/TensorFlow/status/1626629821022208020).

**Update 18/01/2023**: We published a blogpost on the TensorFlow blog discussing this project: [End-to-End Pipeline for Segmentation with TFX, Google Cloud, and Hugging Face](https://blog.tensorflow.org/2023/01/end-to-end-pipeline-for-segmentation-tfx-google-cloud-hugging-face.html).

# Project structure


```bash
project
│
└───notebooks
│   │   gradio_demo.ipynb 
│   │   inference_from_SavedModel.ipynb # test inference w/ Vertex Endpoint
│   │   parse_tfrecords_pets.ipynb # test TFRecord parsing
│   │   tfx_pipeline.ipynb # build TFX pipeline within a notebook
│
└───tfrecords
│   │   create_tfrecords_pets.py # script to create TFRecords of PETS dataset
│
└───training_pipeline
    └───apps # Gradio app template codebase    
    └───models # contains files related to model    
    └───pipeline # definition of TFX pipeline
```

Inside `training_pipeline` the entrypoints for the pipeline runners are defined in
`kubeflow_runner.py` and `local_runner.py`.

# Instructions

The TFX pipeline is designed to be run on both of local and GCP environments. 

## On local environment

```
$ cd training_pipeline
$ tfx pipeline create --pipeline-path=local_runner.py \
                      --engine=local
$ tfx pipeline compile --pipeline-path=local_runner.py \
                       --engine=local
$ tfx run create --pipeline-name=segformer-training-pipeline \ 
                 --engine=local
```

## On Vertex AI environment

There are two ways to run TFX pipeline on GCP environment(Vertex AI). 

First, you can run it manually with the following CLIs. In this case, you should replace `GOOGLE_CLOUD_PROJECT` to your GCP project ID in `training_pipeline/pipeline/configs.py` beforehand.

```
$ cd training_pipeline
$ tfx pipeline create --pipeline-path=kubeflow_runner.py \
                      --engine=vertex
$ tfx pipeline compile --pipeline-path=kubeflow_runner.py \
                       --engine=vertex
$ tfx run create --pipeline-name=segformer-training-pipeline \ 
                 --engine=vertex \ 
                 --project=$GCP_PROJECT_ID \
                 --regeion=$GCP_REGION
```

## Using GitHub Actions

You can use `workflow_dispatch` feature of GitHub Action to run the pipeline on Vertex AI environment as well. In this case, go to the action tab, then select `Trigger Training Pipeline` on the left pane, then `Run workflow` on the branch of your choice. The GCP project ID in the input parameters will automatically replace the `GOOGLE_CLOUD_PROJECT` in `training_pipeline/pipeline/configs.py`. Also it will be injected to the `tfx run create` CLI.

![](https://i.ibb.co/MkTWLZS/dispatch.png)

For further understading about how GitHub Action is implemented, please refer to [its README document](.github/workflows/README.md).

# To-do

- [X] Notebook to prepare input dataset in `TFRecord` format
- [X] Upload the input dataset into the GCS bucket
- [X] Implement and include [UNet](https://www.tensorflow.org/tutorials/images/segmentation) model in the pipeline
- [X] Implement Gradio app template
- [X] Make a complete TFX pipeline with `ExampleGen`, `SchemaGen`, `Resolver`, `Trainer`, `Evaluator`, and `Pusher` components
- [X] Add necessary configurations to the [configs.py](https://github.com/deep-diver/semantic-segmentation-ml-pipeline/blob/main/training_pipeline/pipeline/configs.py)
- [X] Add `HFPusher` component to the TFX pipeline
- [X] Replace `SchemaGen` with `ImportSchemaGen` for better TFRecords parsing capability
- [X] (Optional) Integrate `Dataflow` in `ImportExampleGen` to handle a large amount of dataset. This feature is included in the code as a reference, but it is not used after we switched the Sidewalk to PETS dataset.

## Misc notes

### On the use of two different datasets

Initially, we started our work with the [Sidewalks dataset](https://huggingface.co/datasets/segments/sidewalk-semantic). This
dataset contains different stuff and things and is also very high-resolution in nature. To keep the runtime of our pipeline
faster and to experiment quicker, we settled with a shallow UNet architecture (from [this tutorial](https://www.tensorflow.org/tutorials/images/segmentation)). This is why, we also downsampled the Sidewalks dataset quite a bit (128x128, 256x256, etc.). But this 
led to poor quality models. 

To circumvent around this, we used the [PETS dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) where the effects of downsampling
weren't that visible compared to Sidewalks. 

But do note that the approaches showcases in our pipeline can easily be extended to high-resolution segmentation datasets and different
model architectures (as long as they can be serialized as a `SavedModel`).

## Acknowledgements

We are thankful to the ML Developer Programs team at Google that provided GCP support.
