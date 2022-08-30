# Semantic Sementation model within ML pipeline

This repository shows how to build a Machine Learning Pipeline for [Semantic Segmentation task](https://paperswithcode.com/task/semantic-segmentation) with [TensorFlow eXtended(TFX)](https://www.tensorflow.org/tfx) and various GCP products such as [Vertex Pipeline](https://cloud.google.com/vertex-ai/docs/pipelines), [Vertex Training](https://cloud.google.com/vertex-ai/docs/training/custom-training), [Vertex Endpoint](https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api). Also, the ML pipeline contains few custom components integrated with 🤗 Hub. `HFModelPusher` pushes trained model to 🤗 Model Repositry, and `HFSpacePusher` creates a `Gradio` application with latest model out of the box.

# To-do

- [ ] Notebook to prepare input dataset in `TFRecord` format
- [ ] Upload the input dataset into the GCS bucket
- [ ] (Optional) Add additional Service Account Key as a GitHub Action Secret if collaborators want to run ML pipeline on the GCP with their own GCP account. Each word of the secret key should be separated with underscore. For example, `GCP-ML-172005` should be `GCP_ML_172005`.
- [ ] Modify [Dockerfile](https://github.com/deep-diver/segformer-in-ml-pipeline/blob/main/training_pipeline/Dockerfile) to install 🤗 `Transformers` dependency.
- [ ] Modify [modeling part](https://github.com/deep-diver/segformer-in-ml-pipeline/blob/main/training_pipeline/models/model.py) to use 🤗 `SegFormer`. Initial version is written for image classification task.
- [ ] Modify [Gradio app part](https://github.com/deep-diver/segformer-in-ml-pipeline/tree/main/training_pipeline/apps/gradio/semantic_segmentation). Initial version is copied from [segformer-tf-transformers](https://github.com/deep-diver/segformer-tf-transformers) repository.
- [ ] Modify [Pipeline part](https://github.com/deep-diver/segformer-in-ml-pipeline/blob/main/training_pipeline/pipeline/pipeline.py). May need to remove some optional components such as `StatisticsGen`, `SchemaGen`, `Transform`, `Tuner`, and `Evaluator`.
- [ ] Modify [configs.py](https://github.com/deep-diver/segformer-in-ml-pipeline/blob/main/training_pipeline/pipeline/configs.py) to reflect changes.
- [ ] (Optional) Add custom TFX component to dynamically inject hyperameters to search with `Tuner`.

## Formatting

We use [black](https://pypi.org/project/black/) to lint Python source files. In order to apply it effortlessly, it is recommended to use [pre-commit](https://pre-commit.com/) hook, so run the following command before any contributions.

```shell
$ pre-commit install --hook-type pre-push
```

## Acknowledgements

We are thankful to the ML Developer Programs team at Google that provided GCP support.
