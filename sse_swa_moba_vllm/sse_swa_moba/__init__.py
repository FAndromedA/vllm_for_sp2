from vllm import ModelRegistry
from transformers import AutoConfig, AutoModel

def register_model():
    from .modeling_sse_swa_moba import SseSwaMobaForCausalLM
    from .configuration_SseSwaMoba import SseSwaMobaConfig

    AutoConfig.register("sse_swa_moba", SseSwaMobaConfig)
    ModelRegistry.register_model(
        "SseSwaMobaForCausalLM",
        "SpikingBrain.sse_swa_moba.modeling_sse_swa_moba:SseSwaMobaForCausalLM",
    )