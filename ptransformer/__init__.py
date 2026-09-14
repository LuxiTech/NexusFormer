# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_ptransformer import PTransformerConfig
from .modeling_ptransformer import (
    PTransformerForCausalLM, PTransformerModel)

'''AutoConfig.register(PTransformerConfig.model_type, PTransformerConfig)
AutoModel.register(PTransformerConfig, PTransformerModel)
AutoModelForCausalLM.register(PTransformerConfig, PTransformerForCausalLM)'''


__all__ = ['PTransformerConfig', 'PTransformerForCausalLM', 'PTransformerModel']
