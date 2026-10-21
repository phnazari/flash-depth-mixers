from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from depth_mixer.models.attnres.configuration_attnres import AttnResConfig
from depth_mixer.models.attnres.modeling_attnres import AttnResForCausalLM, AttnResModel

AutoConfig.register(AttnResConfig.model_type, AttnResConfig, exist_ok=True)
AutoModel.register(AttnResConfig, AttnResModel, exist_ok=True)
AutoModelForCausalLM.register(AttnResConfig, AttnResForCausalLM, exist_ok=True)

__all__ = ['AttnResConfig', 'AttnResForCausalLM', 'AttnResModel']
