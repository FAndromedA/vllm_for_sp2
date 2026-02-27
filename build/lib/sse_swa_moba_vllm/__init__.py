

def register_model():

    from .sse_swa_moba import register_model
    register_model()
    register_attention_backends()


def register_attention_backends():
    
    from .attention.moba_attn import MobaSseFlashAttentionBackend
    # Registration is handled by the @register_backend decorator on the class itself, so no additional code is needed here.
    # but this function serves as a central place to ensure the module is imported and the backend is registered when this function is called.