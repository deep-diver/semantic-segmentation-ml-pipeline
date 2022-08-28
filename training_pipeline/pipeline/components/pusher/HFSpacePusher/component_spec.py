from pipeline.components.pusher.HFSpacePusher import constants
from tfx.types.standard_artifacts import PushedModel
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter


class HFSpacePusherSpec(ComponentSpec):
    """Pusher component spec."""

    PARAMETERS = {
        constants.CUSTOM_CONFIG_KEY: ExecutionParameter(type=str),
    }
    INPUTS = {
        constants.HF_MODEL_KEY: ChannelParameter(type=PushedModel),
    }
    OUTPUTS = {
        constants.PUSHED_SPACE_KEY: ChannelParameter(type=PushedModel),
    }
