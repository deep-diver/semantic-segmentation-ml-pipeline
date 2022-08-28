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
from tfx.extensions.google_cloud_ai_platform.tuner.component import Tuner as VertexTuner
from pipeline.components.pusher.GHReleasePusher.component import Pusher as GHPusher
from pipeline.components.pusher.HFModelPusher.component import Pusher as HFModelPusher
from pipeline.components.pusher.HFSpacePusher.component import Pusher as HFSpacePusher
from tfx.components import Transform
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2
from tfx.proto import trainer_pb2
from tfx.proto import tuner_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.orchestration.data_types import RuntimeParameter


def create_pipeline(
    input_config: RuntimeParameter,
    output_config: RuntimeParameter,
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    modules: Dict[Text, Text],
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    tuner_args: tuner_pb2.TuneArgs,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    ai_platform_tuner_args: Optional[Dict[Text, Text]] = None,
    ai_platform_serving_args: Optional[Dict[Text, Any]] = None,
    gh_release_args: Optional[Dict[Text, Any]] = None,
    hf_model_release_args: Optional[Dict[Text, Any]] = None,
    hf_space_release_args: Optional[Dict[Text, Any]] = None,
) -> tfx.dsl.Pipeline:
    components = []

    example_gen = ImportExampleGen(
        input_base=data_path, input_config=input_config, output_config=output_config
    )
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

    tuner = VertexTuner(
        tuner_fn=modules["cloud_tuner_fn"],
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        train_args=train_args,
        eval_args=eval_args,
        tune_args=tuner_args,
        custom_config=ai_platform_tuner_args,
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
        "custom_config": ai_platform_training_args,
    }
    trainer = VertexTrainer(**trainer_args)
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
        "custom_config": ai_platform_serving_args,
    }
    # pusher = VertexPusher(**pusher_args)  # pylint: disable=unused-variable
    # components.append(pusher)

    # pusher_args["custom_config"] = gh_release_args
    # gh_pusher = GHPusher(**pusher_args).with_id("GHReleasePusher")
    # components.append(gh_pusher)

    pusher_args["custom_config"] = hf_model_release_args
    hf_model_pusher = HFModelPusher(**pusher_args).with_id("HFModelPusher")
    components.append(hf_model_pusher)

    space_pusher_args = {
        "model": hf_model_pusher.outputs["pushed_model"],
        "custom_config": hf_space_release_args,
    }
    hf_space_pusher = HFSpacePusher(**space_pusher_args).with_id("HFSpacePusher")
    components.append(hf_space_pusher)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata_connection_config,
    )
