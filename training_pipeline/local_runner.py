import os
from absl import logging

from tfx import v1 as tfx
from tfx.orchestration.data_types import RuntimeParameter
from pipeline import configs
from pipeline import local_pipeline

# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
# NOTE: It is recommended to have a separated OUTPUT_DIR which is *outside* of
#       the source code structure. Please change OUTPUT_DIR to other location
#       where we can store outputs of the pipeline.
OUTPUT_DIR = "."

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
# - Metadata will be written to SQLite database in METADATA_PATH.
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, "tfx_pipeline_output", configs.PIPELINE_NAME)
METADATA_PATH = os.path.join(
    OUTPUT_DIR, "tfx_metadata", configs.PIPELINE_NAME, "metadata.db"
)

# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model")

# Specifies data file directory. DATA_PATH should be a directory containing CSV
# files for CsvExampleGen in this example. By default, data files are in the
# `data` directory.
# NOTE: If you upload data files to GCS(which is recommended if you use
#       Kubeflow), you can use a path starting "gs://YOUR_BUCKET_NAME/path" for
#       DATA_PATH. For example,
#       DATA_PATH = 'gs://bucket/penguin/csv/'.
# TODO(step 4): Specify the path for your data.
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def run():
    """Define a pipeline."""

    tfx.orchestration.LocalDagRunner().run(
        local_pipeline.create_pipeline(
            pipeline_name=configs.PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_path=DATA_PATH,
            modules={
                "preprocessing_fn": configs.PREPROCESSING_FN,
                "training_fn": configs.TRAINING_FN,
                "tuner_fn": configs.TUNER_FN,
            },
            train_args=tfx.proto.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
            eval_args=tfx.proto.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
            serving_model_dir=SERVING_MODEL_DIR,
            metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
                METADATA_PATH
            ),
        )
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
