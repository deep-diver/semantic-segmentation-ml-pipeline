Files in `notebooks` contains Jupyter Notebooks showing some of the verifications of a part of work of the entire workflow. All files could be categorized in two based on the different datasets. This is because we have verified ML pipeline based on [Sidewalk Dataset](https://huggingface.co/datasets/segments/sidewalk-semantic) first, and moved to [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) later.

### Jupyter Notebooks for handling Oxford-IIIT Pet Dataset

- `gradio_demo_pets.ipynb`
- `parse_tfrecords_pets.ipynb`
- `tfx_pipeline_pets.ipynb`
- `inference_SavedModel_VertexEndpoint.ipynb`

### Jupyter Notebooks for handling Sidewalk Dataset
- `parse_tfrecords_sidewalks.ipynb`
- `unet_training_sidewalks.ipynb`
- `tfx_pipeline_sidewalks.ipynb`