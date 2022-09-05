from typing import Any, Dict, List, Optional, Text

from tfx import v1 as tfx

from ml_metadata.proto import metadata_store_pb2
from tfx.proto import example_gen_pb2

from tfx.components import ImportExampleGen
from tfx.extensions.google_cloud_ai_platform.trainer.component import (
    Trainer as VertexTrainer,
)
from tfx.extensions.google_cloud_ai_platform.pusher.component import (
    Pusher as VertexPusher,
)
from pipeline.components.pusher.HFModelPusher.component import Pusher as HFModelPusher
from pipeline.components.pusher.HFSpacePusher.component import Pusher as HFSpacePusher
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2
from tfx.proto import trainer_pb2


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    modules: Dict[Text, Text],
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    ai_platform_serving_args: Optional[Dict[Text, Any]] = None,
    hf_model_release_args: Optional[Dict[Text, Any]] = None,
    hf_space_release_args: Optional[Dict[Text, Any]] = None,
) -> tfx.dsl.Pipeline:
    components = []

    input_config = example_gen_pb2.Input(
        splits=[
            example_gen_pb2.Input.Split(name="train", pattern="train-00-*.tfrec"),
            example_gen_pb2.Input.Split(name="eval", pattern="val-00-*.tfrec"),
        ]
    )
    example_gen = ImportExampleGen(input_base=data_path, input_config=input_config)
    components.append(example_gen)

    trainer_args = {
        "run_fn": modules["training_fn"],
        "examples": example_gen.outputs["examples"],
        "train_args": train_args,
        "eval_args": eval_args,
        "custom_config": ai_platform_training_args,
    }
    trainer = VertexTrainer(**trainer_args)
    components.append(trainer)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False,
        metadata_connection_config=metadata_connection_config,
    )

