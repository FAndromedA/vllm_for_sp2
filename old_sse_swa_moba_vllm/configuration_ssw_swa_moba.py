from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Union
import warnings
from pydantic import ConfigDict

@dataclass
class SSESWAMoBAConfig:
    """
    vLLM compatible configuration for SSE-SWA-MoBA model.
    
    This class is designed to be used with vLLM's configuration system and 
    provides compatibility with both vLLM native format and HuggingFace Transformers format.
    """
    
    # Core model configuration
    model_type: str = "sse_swa_moba"
    attn_mode: str = "chunk"
    hidden_size: int = 256 # 2048
    expand_v: float = 1.0
    use_output_gate: bool = True
    use_short_conv: bool = False
    allow_neg_eigval: bool = False
    conv_size: int = 4
    head_dim: int = 64 # 256
    num_heads: int = 4 # 8
    num_v_heads: Optional[int] = None
    num_sparse_partition: int = 4
    num_writer: int = 2
    num_reader: int = 2
    linear_attn_type: str = "gdn"
    sse_implementation: str = "varlen"
    sse_qk_relu: bool = False
    aux_loss_coef: float = 0.01
    max_position_embeddings: int = 2048
    hidden_ratio: Optional[int] = 4
    intermediate_size: Optional[int] = None
    hidden_act: str = "swish"
    num_hidden_layers: int = 3 # 24
    norm_eps: float = 1e-6
    # mini configuration
    attn: Optional[Dict] = {
        "layers": [
            1, 2
        ],
        "full_layers": [2],
        "num_heads": 32,
        "num_kv_heads": 8,
        "qkv_bias": False,
        "qk_norm": True,
        "window_size": 64,
        "rope_theta": 5000000,
        "moba_chunk_size": 512,
        "moba_topk": 4
    }
    swa_dropout: float = 0.5
    use_cache: bool = True
    initializer_range: float = 0.02
    
    # Tokenization and generation
    vocab_size: int = 32000
    pad_token_id: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    
    # Optimization and fusion
    fuse_norm: bool = True
    fuse_swiglu: bool = True
    fuse_cross_entropy: bool = False
    fuse_linear_cross_entropy: bool = False
    use_l2warp: bool = False
    
    # vLLM specific configurations
    model_config: Any = field(default_factory=dict)
    trust_remote_code: bool = False
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    swap_space: float = 4
    cpu_offload_gb: float = 0
    
    # For compatibility with vLLM's config system
    model_config_dict: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Post initialization validation and setup.
        """
        # Validate conflicting fusion flags
        if self.fuse_cross_entropy and self.fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        
        # Warning for linear cross entropy fusion
        if self.fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )
        
        # Validate attention configuration
        if self.attn is not None:
            if not isinstance(self.attn, dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in self.attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in self.attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            
            # Set default values for attention config
            self.attn['num_kv_heads'] = self.attn.get('num_kv_heads', self.attn['num_heads'])
            self.attn['qkv_bias'] = self.attn.get('qkv_bias', False)
            self.attn['window_size'] = self.attn.get('window_size', None)
            self.attn['moba_chunk_size'] = self.attn.get('moba_chunk_size', 1024)
            self.attn['moba_topk'] = self.attn.get('moba_topk', 4)
    
    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "SSESWAMoBAConfig":
        """
        Convert a HuggingFace Transformers config to vLLM compatible config.
        
        Args:
            hf_config: HuggingFace Transformers configuration object
            
        Returns:
            SSESWAMoBAConfig: vLLM compatible configuration
        """
        config_dict = {}
        
        # Copy all attributes from HF config
        for key in hf_config.__dict__:
            if not key.startswith('_'):  # Skip private attributes
                config_dict[key] = getattr(hf_config, key)
        
        # Create vLLM config
        return cls(**config_dict)
    
    def to_vllm_engine_args(self) -> Dict[str, Any]:
        """
        Convert to vLLM EngineArgs compatible dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary compatible with vLLM's EngineArgs
        """
        return {
            "model": "sse_swa_moba",  # This should be the actual model name/path
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "swap_space": self.swap_space,
            "cpu_offload_gb": self.cpu_offload_gb,
            "model_config": {
                "hf_config": self,
                "model_type": self.model_type,
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_heads,
                "head_dim": self.head_dim,
                "vocab_size": self.vocab_size,
                "max_position_embeddings": self.max_position_embeddings,
                # Add other model-specific configurations
                "attn_mode": self.attn_mode,
                "expand_v": self.expand_v,
                "use_output_gate": self.use_output_gate,
                "use_short_conv": self.use_short_conv,
                "conv_size": self.conv_size,
                "num_v_heads": self.num_v_heads,
                "num_sparse_partition": self.num_sparse_partition,
                "num_writer": self.num_writer,
                "num_reader": self.num_reader,
                "linear_attn_type": self.linear_attn_type,
                "sse_implementation": self.sse_implementation,
                "sse_qk_relu": self.sse_qk_relu,
                "attn": self.attn,
            }
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration dictionary for vLLM model loading.
        
        Returns:
            Dict[str, Any]: Model configuration for vLLM
        """
        return {
            "architectures": ["SSESWAMoBAModel"],
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            # Model-specific parameters
            "attn_mode": self.attn_mode,
            "expand_v": self.expand_v,
            "use_output_gate": self.use_output_gate,
            "use_short_conv": self.use_short_conv,
            "conv_size": self.conv_size,
            "num_v_heads": self.num_v_heads,
            "num_sparse_partition": self.num_sparse_partition,
            "num_writer": self.num_writer,
            "num_reader": self.num_reader,
            "linear_attn_type": self.linear_attn_type,
            "sse_implementation": self.sse_implementation,
            "sse_qk_relu": self.sse_qk_relu,
            "attn": self.attn,
        }


# For vLLM model registration
def register_sse_swa_moba_model():
    """
    Register the SSE-SWA-MoBA model with vLLM's model registry.
    """
    try:
        from vllm import ModelRegistry
        
        # This is a placeholder - in practice, you would import your actual model class
        class SSESWAMoBAModel:
            def __init__(self, config: SSESWAMoBAConfig):
                self.config = config
        
        ModelRegistry.register_model("sse_swa_moba", SSESWAMoBAModel)
        print("SSE-SWA-MoBA model registered with vLLM")
    except ImportError:
        warnings.warn("vLLM is not installed. Model registration skipped.")
    except Exception as e:
        warnings.warn(f"Failed to register model with vLLM: {e}")


# Example usage
if __name__ == "__main__":
    # Create a default vLLM config
    vllm_config = SSESWAMoBAConfig()
    print("Default vLLM Config:")
    print(vllm_config)
    
    # Convert to EngineArgs
    engine_args = vllm_config.to_vllm_engine_args()
    print("\nEngine Args:")
    print(engine_args)
    
    # Get model config
    model_config = vllm_config.get_model_config()
    print("\nModel Config:")
    print(model_config)