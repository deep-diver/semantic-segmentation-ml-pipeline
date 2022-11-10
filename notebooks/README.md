Files in `notebooks` contains Jupyter Notebooks showing some of the verifications of a part of work of the entire workflow. All files could be categorized in two based on the dataset being used. 

In this project, we used two different datasets for validation purposes:

* [Sidewalk Dataset](https://huggingface.co/datasets/segments/sidewalk-semantic)
* [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

### Jupyter Notebooks for handling Oxford-IIIT Pet Dataset

- `gradio_demo_pets.ipynb`: demonstrate how to build Gradio app that runs the model hosted on 🤗 Model Hub.
- `parse_tfrecords_pets.ipynb`: verify that generated TFRecord files are not broken and they contains information encoded correctly.
- `tfx_pipeline_pets.ipynb`: demonstrate how to build TFX pipeline with InteractiveContext that runs the pipeline on Jupyter Notebook.
- `inference_SavedModel_VertexEndpoint.ipynb`: demonstrate how to run inference a test image data locally after downloading a SavedModel emitted as an intermediate result from the pipeline. It also demonstrates how to run inference the same test image data on Vertex AI's Endpoint.

### Jupyter Notebooks for handling Sidewalk Dataset
- `parse_tfrecords_sidewalks.ipynb`: verify that generated TFRecord files are not broken and they contains information encoded correctly.
- `unet_training_sidewalks.ipynb`: demonstrate how to train Unet model with Sidewalk Dataset. Note that we didn't utilize TFX in this notebook. 
- `tfx_pipeline_sidewalks.ipynb`: demonstrate how to build TFX pipeline with InteractiveContext that runs the pipeline on Jupyter Notebook.