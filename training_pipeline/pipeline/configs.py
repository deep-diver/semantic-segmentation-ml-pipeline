import os  # pylint: disable=unused-import
import tfx
import tfx.extensions.google_cloud_ai_platform.constants as vertex_const
import tfx.extensions.google_cloud_ai_platform.trainer.executor as vertex_training_const
import tfx.extensions.google_cloud_ai_platform.tuner.executor as vertex_tuner_const
import tensorflow_model_analysis as tfma

PIPELINE_NAME = "segformer-training-pipeline"

try:
    import google.auth  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    try:
        _, GOOGLE_CLOUD_PROJECT = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        GOOGLE_CLOUD_PROJECT = "gcp-ml-172005"
except ImportError:
    GOOGLE_CLOUD_PROJECT = "gcp-ml-172005"

GOOGLE_CLOUD_REGION = "us-central1"

GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + "-complete-mlops"
PIPELINE_IMAGE = f"gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}"

OUTPUT_DIR = os.path.join("gs://", GCS_BUCKET_NAME)
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, "tfx_pipeline_output", PIPELINE_NAME)

DATA_PATH = "gs://sidewalks-tfx-lowres/sidewalks-tfrecords/"

PREPROCESSING_FN = "models.preprocessing.preprocessing_fn"
TRAINING_FN = "models.train.run_fn"
TUNER_FN = "models.train.tuner_fn"
CLOUD_TUNER_FN = "models.train.tuner_fn"

GRADIO_APP_PATH = "apps.gradio.img_classifier"
MODEL_HUB_REPO_PLACEHOLDER = "$MODEL_REPO_ID"
MODEL_HUB_URL_PLACEHOLDER = "$MODEL_REPO_URL"
MODEL_VERSION_PLACEHOLDER = "$MODEL_VERSION"

TRAIN_NUM_STEPS = 160
EVAL_NUM_STEPS = 4

EXAMPLE_GEN_BEAM_ARGS = None
TRANSFORM_BEAM_ARGS = None

"""
EVAL_CONFIGS is to configuration for the Evaluator component
to define how it is going to evalua the model performance.

tfma.ModelSpec
    signature_name is one of the signature in the SavedModel from
    Trainer component, and it will be used to make predictions on 
    given data. preprocessing_function_names allows us to include
    a set of transformation(preprocessing) signatures in the Saved
    Model from Trainer component. 

    label_key and prediction_key will be used to compare the ground
    truth and prediction results.

slicing_specs
    we use the entire dataset to evaluate the model performance. If
    you want to evaluate the model based on different slices of data
    set, you should prepare TFRecords to have multiple features which
    of each corresponds to each slices(or categories), then write the
    slicing_specs options accordingly.
"""
EVAL_CONFIGS = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(
            # the names in the signature_name preprocessing_function_names
            # are defined in the `signatures` parameter when model.save()
            # you can find how it is done in models/train.py
            signature_name="from_examples",
            preprocessing_function_names=["transform_features"],
            label_key="label_xf",
            prediction_key="label_xf",
        )
    ],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(
                    class_name="SparseCategoricalAccuracy",
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={"value": 0.55}
                        ),
                        # Change threshold will be ignored if there is no
                        # baseline model resolved from MLMD (first run).
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value": -1e-3},
                        ),
                    ),
                )
            ]
        )
    ],
)

GCP_AI_PLATFORM_TRAINING_ARGS = {
    vertex_const.ENABLE_VERTEX_KEY: True,
    vertex_const.VERTEX_REGION_KEY: GOOGLE_CLOUD_REGION,
    vertex_training_const.TRAINING_ARGS_KEY: {
        "project": GOOGLE_CLOUD_PROJECT,
        "worker_pool_specs": [
            {
                "machine_spec": {
                    "machine_type": "n1-standard-4",
                    "accelerator_type": "NVIDIA_TESLA_K80",
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": PIPELINE_IMAGE,
                },
            }
        ],
    },
    "use_gpu": True,
}

fullres_data = os.environ.get("FULL_RES_DATA", "false")

if fullres_data.lower() == "true":
    DATA_PATH = "gs://sidewalks-tfx-fullres/sidewalks-tfrecords/"

    DATAFLOW_SERVICE_ACCOUNT = "csp-gde-dataflow@gcp-ml-172005.iam.gserviceaccount.com"
    DATAFLOW_MACHINE_TYPE = "n1-standard-4"
    DATAFLOW_MAX_WORKERS = 4
    DATAFLOW_DISK_SIZE_GB = 100

    EXAMPLE_GEN_BEAM_ARGS = [
        "--runner=DataflowRunner",
        "--project=" + GOOGLE_CLOUD_PROJECT,
        "--region=" + GOOGLE_CLOUD_REGION,
        "--service_account_email=" + DATAFLOW_SERVICE_ACCOUNT,
        "--machine_type=" + DATAFLOW_MACHINE_TYPE,
        "--experiments=use_runner_v2",
        "--max_num_workers=" + str(DATAFLOW_MAX_WORKERS),
        "--disk_size_gb=" + str(DATAFLOW_DISK_SIZE_GB),
    ]

    TRANSFORM_BEAM_ARGS = [
        "--runner=DataflowRunner",
        "--project=" + GOOGLE_CLOUD_PROJECT,
        "--region=" + GOOGLE_CLOUD_REGION,
        "--service_account_email=" + DATAFLOW_SERVICE_ACCOUNT,
        "--machine_type=" + DATAFLOW_MACHINE_TYPE,
        "--experiments=use_runner_v2",
        "--max_num_workers=" + str(DATAFLOW_MAX_WORKERS),
        "--disk_size_gb=" + str(DATAFLOW_DISK_SIZE_GB),
        "--worker_harness_container_image=" + PIPELINE_IMAGE,
    ]

    GCP_AI_PLATFORM_TRAINING_ARGS[vertex_training_const.TRAINING_ARGS_KEY][
        "worker_pool_specs"
    ] = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-8",
                "accelerator_type": "NVIDIA_TESLA_V100",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": PIPELINE_IMAGE,
            },
        }
    ]


GCP_AI_PLATFORM_SERVING_ARGS = {
    vertex_const.ENABLE_VERTEX_KEY: True,
    vertex_const.VERTEX_REGION_KEY: GOOGLE_CLOUD_REGION,
    vertex_const.VERTEX_CONTAINER_IMAGE_URI_KEY: "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
    vertex_const.SERVING_ARGS_KEY: {
        "project_id": GOOGLE_CLOUD_PROJECT,
        "deployed_model_display_name": PIPELINE_NAME.replace("-", "_"),
        "endpoint_name": "prediction-" + PIPELINE_NAME.replace("-", "_"),
        "traffic_split": {"0": 100},
        "machine_type": "n1-standard-4",
        "min_replica_count": 1,
        "max_replica_count": 1,
    },
}

HF_MODEL_RELEASE_ARGS = {
    "HF_MODEL_RELEASE": {
        "ACCESS_TOKEN": "$HF_ACCESS_TOKEN",
        "USERNAME": "chansung",
        "REPONAME": PIPELINE_NAME,
    }
}

HF_SPACE_RELEASE_ARGS = {
    "HF_SPACE_RELEASE": {
        "ACCESS_TOKEN": "$HF_ACCESS_TOKEN",
        "USERNAME": "chansung",
        "REPONAME": PIPELINE_NAME,
        "APP_PATH": GRADIO_APP_PATH,
        "MODEL_HUB_REPO_PLACEHOLDER": MODEL_HUB_REPO_PLACEHOLDER,
        "MODEL_HUB_URL_PLACEHOLDER": MODEL_HUB_URL_PLACEHOLDER,
        "MODEL_VERSION_PLACEHOLDER": MODEL_VERSION_PLACEHOLDER,
    }
}
