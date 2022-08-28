import os
from absl import logging

from tfx import v1 as tfx
from tfx import proto
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner as runner
from tfx.orchestration.data_types import RuntimeParameter
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2

from pipeline import configs
from pipeline import pipeline

"""
RuntimeParameter could be injected with TFX CLI
: 
--runtime-parameter output-config='{}' \
--runtime-parameter input-config='{"splits": [{"name": "train", "pattern": "span-[12]/train/*.tfrecord"}, {"name": "val", "pattern": "span-[12]/test/*.tfrecord"}]}' 
  
OR it could be injected programatically
: 
  import json
  from kfp.v2.google import client

  pipelines_client = client.AIPlatformClient(
      project_id=GOOGLE_CLOUD_PROJECT, region=GOOGLE_CLOUD_REGION,
  )
  _ = pipelines_client.create_run_from_job_spec(
      PIPELINE_DEFINITION_FILE,
      enable_caching=False,
      parameter_values={
          "input-config": json.dumps(
              {
                  "splits": [
                      {"name": "train", "pattern": "span-[12]/train/*.tfrecord"},
                      {"name": "val", "pattern": "span-[12]/test/*.tfrecord"},
                  ]
              }
          ),
          "output-config": json.dumps({}),
      },
  )          
"""


def run():
    runner_config = runner.KubeflowV2DagRunnerConfig(
        default_image=configs.PIPELINE_IMAGE
    )

    runner.KubeflowV2DagRunner(
        config=runner_config,
        output_filename=configs.PIPELINE_NAME + "_pipeline.json",
    ).run(
        pipeline.create_pipeline(
            input_config=RuntimeParameter(
                name="input-config",
                default='{"input_config": {"splits": [{"name":"train", "pattern":"span-1/train/*"}, {"name":"eval", "pattern":"span-1/test/*"}]}}',
                ptype=str,
            ),
            output_config=RuntimeParameter(
                name="output-config",
                default="{}",
                ptype=str,
            ),
            pipeline_name=configs.PIPELINE_NAME,
            pipeline_root=configs.PIPELINE_ROOT,
            data_path=configs.DATA_PATH,
            modules={
                "preprocessing_fn": configs.PREPROCESSING_FN,
                "training_fn": configs.TRAINING_FN,
                "cloud_tuner_fn": configs.CLOUD_TUNER_FN,
            },
            train_args=trainer_pb2.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
            eval_args=trainer_pb2.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
            tuner_args=tuner_pb2.TuneArgs(
                num_parallel_trials=configs.NUM_PARALLEL_TRIALS
            ),
            ai_platform_training_args=configs.GCP_AI_PLATFORM_TRAINING_ARGS,
            ai_platform_tuner_args=configs.GCP_AI_PLATFORM_TUNER_ARGS,
            ai_platform_serving_args=configs.GCP_AI_PLATFORM_SERVING_ARGS,
            gh_release_args=configs.GH_RELEASE_ARGS,
            hf_model_release_args=configs.HF_MODEL_RELEASE_ARGS,
            hf_space_release_args=configs.HF_SPACE_RELEASE_ARGS,
        )
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
