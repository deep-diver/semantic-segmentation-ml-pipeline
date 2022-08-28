from typing import Any, Dict, Optional

from tfx import types
from tfx.types import standard_artifacts
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from pipeline.components.pusher.HFSpacePusher import executor
from pipeline.components.pusher.HFSpacePusher import component_spec

from tfx.utils import json_utils


class Pusher(base_component.BaseComponent):
    SPEC_CLASS = component_spec.HFSpacePusherSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(
        self,
        model: types.Channel = None,
        custom_config: Optional[Dict[str, Any]] = None,
    ):

        pushed_space = types.Channel(type=standard_artifacts.PushedModel)

        spec = component_spec.HFSpacePusherSpec(
            hf_model=model,
            custom_config=json_utils.dumps(custom_config),
            pushed_space=pushed_space,
        )
        super().__init__(spec=spec)
