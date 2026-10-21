from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from depth_mixer.models.ldm.configuration_ldm import LDMConfig
from depth_mixer.models.ldm.modeling_ldm import LDMForCausalLM, LDMModel

AutoConfig.register(LDMConfig.model_type, LDMConfig, exist_ok=True)
AutoModel.register(LDMConfig, LDMModel, exist_ok=True)
AutoModelForCausalLM.register(LDMConfig, LDMForCausalLM, exist_ok=True)

__all__ = ['LDMConfig', 'LDMForCausalLM', 'LDMModel']
