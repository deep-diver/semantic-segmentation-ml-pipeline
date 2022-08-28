from typing import Any, Dict, List, Optional, Text

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx

from ml_metadata.proto import metadata_store_pb2
from tfx.proto import example_gen_pb2

import absl
import tensorflow_model_analysis as tfma
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ImportExampleGen
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Tuner
from tfx.extensions.google_cloud_ai_platform.trainer.component import (
    Trainer as VertexTrainer,
)
from tfx.extensions.google_cloud_ai_platform.pusher.component import (
    Pusher as VertexPusher,
)
from tfx.components import Transform
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.orchestration.data_types import RuntimeParameter

from components.pusher.GHReleasePusher.component import Pusher as GHPusher


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    modules: Dict[Text, Text],
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    serving_model_dir: Text,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
) -> tfx.dsl.Pipeline:
    components = []

    input_config = example_gen_pb2.Input(
        splits=[
            example_gen_pb2.Input.Split(name="train", pattern="train/*.tfrecord"),
            example_gen_pb2.Input.Split(name="eval", pattern="test/*.tfrecord"),
        ]
    )
    example_gen = ImportExampleGen(input_base=data_path, input_config=input_config)
    components.append(example_gen)

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
    components.append(statistics_gen)

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True
    )
    components.append(schema_gen)

    #   example_validator = tfx.components.ExampleValidator(
    #       statistics=statistics_gen.outputs['statistics'],
    #       schema=schema_gen.outputs['schema'])
    #   components.append(example_validator)

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        preprocessing_fn=modules["preprocessing_fn"],
    )
    components.append(transform)

    tuner = Tuner(
        tuner_fn=modules["tuner_fn"],
        examples=transform.outputs["transformed_examples"],
        schema=schema_gen.outputs["schema"],
        transform_graph=transform.outputs["transform_graph"],
        train_args=train_args,
        eval_args=eval_args,
    )
    components.append(tuner)

    trainer_args = {
        "run_fn": modules["training_fn"],
        "transformed_examples": transform.outputs["transformed_examples"],
        "schema": schema_gen.outputs["schema"],
        "hyperparameters": tuner.outputs["best_hyperparameters"],
        "transform_graph": transform.outputs["transform_graph"],
        "train_args": train_args,
        "eval_args": eval_args,
    }
    trainer = Trainer(**trainer_args)
    components.append(trainer)

    model_resolver = resolver.Resolver(
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")
    components.append(model_resolver)

    # Uses TFMA to compute evaluation statistics over features of a model and
    # perform quality validation of a candidate model (compare to a baseline).
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="label_xf", prediction_key="label_xf")],
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

    evaluator = Evaluator(
        examples=transform.outputs["transformed_examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )
    components.append(evaluator)

    pusher_args = {
        "model": trainer.outputs["model"],
        "model_blessing": evaluator.outputs["blessing"],
        "push_destination": tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    }
    pusher = Pusher(**pusher_args)  # pylint: disable=unused-variable
    components.append(pusher)

    pusher_args = {
        "model": trainer.outputs["model"],
        "model_blessing": evaluator.outputs["blessing"],
        "custom_config": {
            "GH_RELEASE": {
                "ACCESS_TOKEN": "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",
                "USERNAME": "deep-diver",
                "REPONAME": "PyGithubTest",
                "BRANCH": "main",
                "ASSETNAME": "saved_model.tar.gz",
            }
        },
    }

    gh_pusher = GHPusher(**pusher_args).with_id("gh_release_pusher")
    components.append(gh_pusher)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False,
        metadata_connection_config=metadata_connection_config,
    )
