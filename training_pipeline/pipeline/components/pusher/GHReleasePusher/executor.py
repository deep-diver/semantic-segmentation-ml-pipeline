import time
from typing import Any, Dict, List

from google.api_core import client_options
from googleapiclient import discovery
from tfx import types
from tfx.components.pusher import executor as tfx_pusher_executor
from pipeline.components.pusher.GHReleasePusher import constants
from pipeline.components.pusher.GHReleasePusher import runner
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import deprecation_utils
from tfx.utils import io_utils
from tfx.utils import json_utils
from tfx.utils import name_utils
from tfx.utils import telemetry_utils


from tfx.dsl.io import fileio

# Keys for custom_config.
_CUSTOM_CONFIG_KEY = "custom_config"


class Executor(tfx_pusher_executor.Executor):
    """Deploy a model to Google Cloud AI Platform serving."""

    def Do(
        self,
        input_dict: Dict[str, List[types.Artifact]],
        output_dict: Dict[str, List[types.Artifact]],
        exec_properties: Dict[str, Any],
    ):
        """Overrides the tfx_pusher_executor.
        Args:
          input_dict: Input dict from input key to a list of artifacts, including:
            - model_export: exported model from trainer.
            - model_blessing: model blessing path from evaluator.
          output_dict: Output dict from key to a list of artifacts, including:
            - model_push: A list of 'ModelPushPath' artifact of size one. It will
              include the model in this push execution if the model was pushed.
          exec_properties: Mostly a passthrough input dict for
            tfx.components.Pusher.executor.  The following keys in `custom_config`
            are consumed by this class:
                CONFIG = {
                    "USERNAME": "deep-diver",
                    "REPONAME": "PyGithubTest",
                    "ASSETNAME": "saved_model.tar.gz",
                }
        Raises:
          ValueError:
            If one of USERNAME, REPONAME, ASSETNAME, TAG is not in exec_properties.custom_config.
            If Serving model path does not start with gs://.
          RuntimeError: if the GitHub Release job failed.
        """
        self._log_startup(input_dict, output_dict, exec_properties)

        custom_config = json_utils.loads(
            exec_properties.get(_CUSTOM_CONFIG_KEY, "null")
        )

        if custom_config is not None and not isinstance(custom_config, Dict):
            raise ValueError(
                "custom_config in execution properties needs to be a dict."
            )

        gh_release_args = custom_config.get(constants.GH_RELEASE_KEY)
        if not gh_release_args:
            raise ValueError("'GH_RELEASE' is missing in 'custom_config'")
        model_push = artifact_utils.get_single_instance(
            output_dict[standard_component_specs.PUSHED_MODEL_KEY]
        )
        if not self.CheckBlessing(input_dict):
            self._MarkNotPushed(model_push)
            return

        # Deploy the model.
        io_utils.copy_dir(src=self.GetModelPath(input_dict), dst=model_push.uri)
        model_path = model_push.uri

        executor_class_path = name_utils.get_full_name(self.__class__)
        with telemetry_utils.scoped_labels(
            {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}
        ):
            job_labels = telemetry_utils.make_labels_dict()

        model_name = f"v{int(time.time())}"
        pushed_model_path = runner.release_model_for_github(
            model_path=model_path,
            model_version_name=model_name,
            gh_release_args=gh_release_args,
        )
        self._MarkPushed(model_push, pushed_destination=pushed_model_path)
