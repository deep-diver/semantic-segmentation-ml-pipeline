## Preprocessing and Modeling modules

- `train.py`: the `run_fn` function inside this file is designated in the `run_fn` argument of the `Trainer` component. `run_fn` function basically constructs `tf.data` pipeline for training and validation dataset, builds a model, train the model, and export the trained model. 
  - `unet.py`: the model used in this project is a **U-NET** variant borrowed from [TensorFlow's official tutorial](https://www.tensorflow.org/tutorials/images/segmentation). the `build_model` function returns the U-NET implementation based on Keras, and it is used in the `train.py`.
  - `signatures.py`: a signature let us add custom functions to a model when it is exported as `SavedModel`. This file defines a set of signatures for the model. `model_exporter` defines a signature that handles requests when the model is deployed, `transform_features_signature` defines a signature that applies preprocessing logics from the `Transform` component to the input, and `tf_examples_serving_signature` defines a signature that gets predictions from the model after applying the preprocessing logics from the `Transform` component. 
    - NOTE: difference between `model_exporter` and `tf_examples_serving_signature` is whether the preprocessing logics from the `Transform` component is applied.
    - NOTE: the results of `Transform` component can be accessed by `FnArgs.transform_output` when `transform_graph` argument is set within `Trainer` component. Then, we can simply call `transform_output.transform_features_layer()` to preprocess dataset according to how `Transform` logics are defined.


- `preprocessing.py`: the `preprocessing_fn` function inside this file is designated in the `preprocessing_fn` argument of the `Transform` component. `preprocessing_fn` function basically transforms the raw data from the upstream component, `ImportExampleGen` to be fit into the format that the model expects.

## Commons modules

The other files contains a set of common functions and variables used in the `train.py` and `preprocessing.py` modules. 

- `common.py`: it defines a set of keys to extract features from TFRecords
- `utils.py`: it defines utility functions. 
- `hyperparams.py`: it defines a set of hyper-parameters used in the model.