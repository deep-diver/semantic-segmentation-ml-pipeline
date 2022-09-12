![Python](https://img.shields.io/badge/python-3.9-blue.svg) [![TFX](https://img.shields.io/badge/TFX-1.9.1-orange)](https://www.tensorflow.org/tfx)
[![Validity Check for Training Pipeline](https://github.com/deep-diver/semantic-segmentation-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/deep-diver/semantic-segmentation-ml-pipeline/actions/workflows/ci.yml) [![Trigger Training Pipeline](https://github.com/deep-diver/semantic-segmentation-ml-pipeline/actions/workflows/cd-traing-pipeline.yml/badge.svg)](https://github.com/deep-diver/semantic-segmentation-ml-pipeline/actions/workflows/cd-traing-pipeline.yml)

# Semantic Sementation model within ML pipeline

<p align="center">
  <img src="https://i.ibb.co/CtSJLQq/semantic-segmentation-ml-pipeline.png" width="70%" />
</p>

This repository shows how to build a Machine Learning Pipeline for [Semantic Segmentation task](https://paperswithcode.com/task/semantic-segmentation) with [TensorFlow Extended(TFX)](https://www.tensorflow.org/tfx) and various GCP products such as [Vertex Pipeline](https://cloud.google.com/vertex-ai/docs/pipelines), [Vertex Training](https://cloud.google.com/vertex-ai/docs/training/custom-training), [Vertex Endpoint](https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api). Also, the ML pipeline contains few custom components integrated with 🤗 Hub. `HFModelPusher` pushes trained model to 🤗 Model Repositry, and `HFSpacePusher` creates a `Gradio` application with latest model out of the box.

**NOTE**: We use U-NET based TensorFlow model from the [official tutorial](https://www.tensorflow.org/tutorials/images/segmentation). Since we implement an ML pipeline, U-NET like model could be a good starting point. Other SOTA models like [SegFormer from 🤗 `Transformers`](https://huggingface.co/transformers/v4.12.5/model_doc/segformer.html) will be explored later.

**NOTE**: The aim of this project is not to serve a fully fine-tuned model. Our main focus is to demonstrate how to build ML pipeline for semantic segmentation task instead.

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

Or, you can use `workflow_dispatch` feature of GitHub Action. In this case, go to the action tab, then select `Trigger Training Pipeline` on the left pane, then `Run workflow` on the branch of your choice. The GCP project ID in the input parameters will automatically replace the `GOOGLE_CLOUD_PROJECT` in `training_pipeline/pipeline/configs.py`. Also it will be injected to the `tfx run create` CLI.

![](https://i.ibb.co/MkTWLZS/dispatch.png)

# To-do

- [X] Notebook to prepare input dataset in `TFRecord` format
- [X] Upload the input dataset into the GCS bucket
- [ ] (Optional) Add additional Service Account Key as a GitHub Action Secret if collaborators want to run ML pipeline on the GCP with their own GCP account. Each word of the secret key should be separated with underscore. For example, `GCP-ML-172005` should be `GCP_ML_172005`.
- [X] Modify [modeling part](https://github.com/deep-diver/segformer-in-ml-pipeline/blob/main/training_pipeline/models/model.py) to train TensorFlow based Unet model.
- [ ] Modify [Gradio app part](https://github.com/deep-diver/segformer-in-ml-pipeline/tree/main/training_pipeline/apps/gradio/semantic_segmentation). Initial version is copied from [segformer-tf-transformers](https://github.com/deep-diver/segformer-tf-transformers) repository.
- [ ] Modify [Pipeline part](https://github.com/deep-diver/segformer-in-ml-pipeline/blob/main/training_pipeline/pipeline/pipeline.py). May need to remove some optional components such as `StatisticsGen`, `SchemaGen`, `Transform`, `Tuner`, and `Evaluator`.
- [ ] Modify [configs.py](https://github.com/deep-diver/segformer-in-ml-pipeline/blob/main/training_pipeline/pipeline/configs.py) to reflect changes.
- [ ] (Optional) Add custom TFX component to dynamically inject hyperameters to search with `Tuner`.

## Acknowledgements

We are thankful to the ML Developer Programs team at Google that provided GCP support.
