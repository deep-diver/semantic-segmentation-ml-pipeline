from typing import Any, Dict, Optional

from tfx import types
from tfx.components.pusher import component as pusher_component
from tfx.dsl.components.base import executor_spec
from pipeline.components.pusher.HFModelPusher import executor


class Pusher(pusher_component.Pusher):
    """Component for pushing model to Cloud AI Platform serving."""

    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(
        self,
        model: Optional[types.Channel] = None,
        model_blessing: Optional[types.Channel] = None,
        infra_blessing: Optional[types.Channel] = None,
        custom_config: Optional[Dict[str, Any]] = None,
    ):
        """Construct a Pusher component.
        Args:
          model: An optional Channel of type `standard_artifacts.Model`, usually
            produced by a Trainer component, representing the model used for
            training.
          model_blessing: An optional Channel of type
            `standard_artifacts.ModelBlessing`, usually produced from an Evaluator
            component, containing the blessing model.
          infra_blessing: An optional Channel of type
            `standard_artifacts.InfraBlessing`, usually produced from an
            InfraValidator component, containing the validation result.
          custom_config: A dict which contains the deployment job parameters to be
            passed to Cloud platforms.
        """
        super().__init__(
            model=model,
            model_blessing=model_blessing,
            infra_blessing=infra_blessing,
            custom_config=custom_config,
        )
