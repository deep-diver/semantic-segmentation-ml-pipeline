# Semantic Sementation model within ML pipeline

This repository shows how to build a Machine Learning Pipeline for [Semantic Segmentation task](https://paperswithcode.com/task/semantic-segmentation) with [TensorFlow eXtended(TFX)](https://www.tensorflow.org/tfx) and various GCP products such as [Vertex Pipeline](https://cloud.google.com/vertex-ai/docs/pipelines), [Vertex Training](https://cloud.google.com/vertex-ai/docs/training/custom-training), [Vertex Endpoint](https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api). Also, the ML pipeline contains few custom components integrated with 🤗 Hub. `HFModelPusher` pushes trained model to 🤗 Model Repositry, and `HFSpacePusher` creates a `Gradio` application with latest model out of the box.

# To-do

- [ ] Notebook to prepare input dataset in `TFRecord` format
- [ ] Notebook to run TFX pipeline on Google Cloud Platform
  - Write a module file containing whole model training codes
  - Write a module file containing Gradio app template codes

## Acknowledgements

We are thankful to the ML Developer Programs team at Google that provided GCP support.
