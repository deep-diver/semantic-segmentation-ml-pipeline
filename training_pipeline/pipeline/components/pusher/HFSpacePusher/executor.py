import time
from absl import logging

from typing import Any, Dict, List, Optional

from google.api_core import client_options
from googleapiclient import discovery
from tfx import types
from pipeline.components.pusher.HFSpacePusher import constants
from pipeline.components.pusher.HFSpacePusher import runner
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils
from tfx.utils import name_utils
from tfx.utils import telemetry_utils
from tfx.dsl.components.base import base_executor
from tfx import types

# Keys for custom_config.
_CUSTOM_CONFIG_KEY = "custom_config"
_PUSHED_REPO_ID = "pushed_repo_id"
_PUSHED_VERSION = "pushed_version"
_PUSHED_REPO_URL = "pushed_repo_url"


class Executor(base_executor.BaseExecutor):
    def Do(
        self,
        input_dict: Dict[str, List[types.Artifact]],
        output_dict: Dict[str, List[types.Artifact]],
        exec_properties: Dict[str, Any],
    ):
        self._log_startup(input_dict, output_dict, exec_properties)

        custom_config = json_utils.loads(
            exec_properties.get(_CUSTOM_CONFIG_KEY, "null")
        )

        if custom_config is not None and not isinstance(custom_config, Dict):
            raise ValueError(
                "custom_config in execution properties needs to be a dict."
            )

        hf_release_args = custom_config.get(constants.HF_SPACE_RELEASE_KEY)
        if not hf_release_args:
            raise ValueError("'HF_SPACE_RELEASE_KEY' is missing in 'custom_config'")

        logging.warning(input_dict)

        pushed_hf_model = artifact_utils.get_single_instance(
            input_dict[constants.HF_MODEL_KEY]
        )

        space_to_push = artifact_utils.get_single_instance(
            output_dict[constants.PUSHED_SPACE_KEY]
        )

        if pushed_hf_model.get_int_custom_property("pushed") == 0:
            space_to_push.set_int_custom_property("pushed", 0)
            return

        model_repo_id = pushed_hf_model.get_string_custom_property(_PUSHED_REPO_ID)
        model_repo_url = pushed_hf_model.get_string_custom_property(_PUSHED_REPO_URL)
        model_version = pushed_hf_model.get_string_custom_property(_PUSHED_VERSION)

        space_repo_id, space_url = runner.release_model_for_hf_space(
            model_repo_id=model_repo_id,
            model_repo_url=model_repo_url,
            model_version=model_version,
            hf_release_args=hf_release_args,
        )

        space_to_push.set_int_custom_property("pushed", 1)
        space_to_push.set_string_custom_property(_PUSHED_REPO_URL, space_url)
        space_to_push.set_string_custom_property(_PUSHED_REPO_ID, space_repo_id)
