from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from depth_mixer.models.mhc.configuration_mhc import MHCConfig
from depth_mixer.models.mhc.modeling_mhc import MHCForCausalLM, MHCModel

AutoConfig.register(MHCConfig.model_type, MHCConfig, exist_ok=True)
AutoModel.register(MHCConfig, MHCModel, exist_ok=True)
AutoModelForCausalLM.register(MHCConfig, MHCForCausalLM, exist_ok=True)

__all__ = ['MHCConfig', 'MHCForCausalLM', 'MHCModel']
