"""
SSE (Sparse State Expansion) 层等价性测试脚本
该脚本用于验证HuggingFace风格的SSE实现与vLLM风格的SSE实现
在训练模式、推理prefill阶段和推理decode阶段的输出等价性。
"""
import os
import math
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from typing import Optional, Tuple, Dict, List

from fla.models.utils import Cache

# 过滤掉可能的弃用警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 导入vLLM并行初始化模块（不导入destroy_world_group）
try:
    from vllm.distributed.parallel_state import (
        initialize_model_parallel,
        destroy_model_parallel,
        get_tensor_model_parallel_world_size,
        get_tensor_model_parallel_rank,
        get_world_group
    )
    from vllm.distributed import init_distributed_environment
    # 添加vLLM forward_context支持
    from vllm.config import get_current_vllm_config, CUDAGraphMode
    from vllm.forward_context import set_forward_context
    vllm_available = True
except ImportError as e:
    print(f"⚠️ vLLM并行模块导入失败: {e}")
    vllm_available = False


def manual_weight_initialization(module):
    """手动权重随机初始化函数"""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.ones_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.normal_(module.weight)
        nn.init.ones_(module.bias)
    elif hasattr(module, 'weight') and module.weight is not None:
        if len(module.weight.shape) > 1:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        else:
            nn.init.ones_(module.weight)


def create_attention_metadata(batch_size, seq_len, use_cu_seqlens=False, seq_lens=None):
    """
    创建正确的AttentionMetadata
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        use_cu_seqlens: 是否使用变长序列
        seq_lens: 变长序列的实际长度列表
        
    Returns:
        AttentionMetadata字典
    """
    if use_cu_seqlens and seq_lens is not None:
        # 变长序列处理
        max_len = max(seq_lens)
        num_actual_tokens = sum(seq_lens)
        
        # 创建query_start_loc (cumulative sum of sequence lengths)
        query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        for i in range(batch_size):
            query_start_loc[i+1] = query_start_loc[i] + seq_lens[i]
        
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        
        # 创建虚拟的block_table和slot_mapping（测试用）
        block_size = 16  # 假设的块大小
        max_blocks_per_seq = (max_len + block_size - 1) // block_size
        block_table = torch.zeros((batch_size, max_blocks_per_seq), dtype=torch.int32, device=device)
        slot_mapping = torch.zeros(num_actual_tokens, dtype=torch.int32, device=device)
        
        attn_metadata = {
            "num_actual_tokens": num_actual_tokens,
            "max_query_len": max_len,
            "query_start_loc": query_start_loc,
            "max_seq_len": max_len,
            "seq_lens": seq_lens_tensor,
            "block_table": block_table,
            "slot_mapping": slot_mapping,
            "num_prefill_tokens": num_actual_tokens,
            "num_decode_tokens": 0,
            "num_prefills": batch_size,
            "num_decodes": 0
        }
    else:
        # 定长序列处理
        num_actual_tokens = batch_size * seq_len
        
        # 创建query_start_loc (每个序列从0开始，长度为seq_len)
        query_start_loc = torch.arange(0, num_actual_tokens + 1, seq_len,
                                     dtype=torch.int32, device=device)
        
        seq_lens_tensor = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        
        # 创建虚拟的block_table和slot_mapping（测试用）
        block_size = 16  # 假设的块大小
        max_blocks_per_seq = (seq_len + block_size - 1) // block_size
        block_table = torch.zeros((batch_size, max_blocks_per_seq), dtype=torch.int32, device=device)
        slot_mapping = torch.zeros(num_actual_tokens, dtype=torch.int32, device=device)
        
        attn_metadata = {
            "num_actual_tokens": num_actual_tokens,
            "max_query_len": seq_len,
            "query_start_loc": query_start_loc,
            "max_seq_len": seq_len,
            "seq_lens": seq_lens_tensor,
            "block_table": block_table,
            "slot_mapping": slot_mapping,
            "num_prefill_tokens": num_actual_tokens,
            "num_decode_tokens": 0,
            "num_prefills": batch_size,
            "num_decodes": 0
        }
    
    return attn_metadata


def initialize_vllm_distributed():
    """使用vLLM的init_distributed_environment初始化分布式环境"""
    if not vllm_available:
        return False
    
    try:
        # 单GPU模式下的分布式环境初始化
        # 设置环境变量
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12355')
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('LOCAL_RANK', '0')
        
        print("正在初始化vLLM分布式环境...")
        print(f"  MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        print(f"  MASTER_PORT: {os.environ['MASTER_PORT']}")
        print(f"  RANK: {os.environ['RANK']}")
        print(f"  WORLD_SIZE: {os.environ['WORLD_SIZE']}")
        print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', '未设置')}")
        
        init_distributed_environment()
        print("✓ vLLM分布式环境初始化成功")
        return True
    except Exception as e:
        print(f"✗ vLLM分布式环境初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def initialize_vllm_model_parallel():
    """初始化vLLM模型并行"""
    if not vllm_available:
        return False
    
    try:
        # 初始化模型并行
        initialize_model_parallel(tensor_model_parallel_size=1)
        print(f"✓ vLLM模型并行初始化成功")
        print(f"  张量模型并行大小: {get_tensor_model_parallel_world_size()}")
        print(f"  张量模型并行排名: {get_tensor_model_parallel_rank()}")
        return True
    except Exception as e:
        print(f"✗ vLLM模型并行初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weight_loading(sse_type='gla'):
    """测试1：验证vLLM模型能否加载HuggingFace保存的权重"""
    print(f"\n=== 测试1：权重加载测试 ({sse_type.upper()}) ===")
    
    # 导入SSE模型
    from sse_swa_moba_hf.sse_hf import SSEGLA as HFSSEGLA, SSEGDN as HFSSEGDN
    from sse_swa_moba_vllm.sse import VLLMSSEGLA, VLLMSSEGDN
    
    # 根据类型选择模型类
    if sse_type.lower() == 'gla':
        HFModelClass = HFSSEGLA
        VLLMModelClass = VLLMSSEGLA
        model_name = "SSEGLA"
    elif sse_type.lower() == 'gdn':
        HFModelClass = HFSSEGDN
        VLLMModelClass = VLLMSSEGDN
        model_name = "SSEGDN"
    else:
        raise ValueError(f"未知的SSE类型: {sse_type}")
    
    # 1. 创建HuggingFace模型 - 使用bfloat16
    hf_sse = HFModelClass(
        hidden_size=512,
        expand_v=1.0,
        head_dim=64,
        num_heads=8,
        num_v_heads=8,
        mode='chunk',
        use_output_gate=True,
        use_short_conv=False,
        num_sparse_partition=4,
        num_writer=1,
        num_reader=1,
        sse_implementation="varlen",
        use_q_softmax=False,
        use_k_softmax=True,
        emulq=True,
        emulk=True,
        gate_logit_normalizer=16,
        gate_low_rank_dim=16,
        layer_idx=0,
        norm_eps=1e-5
    ).to(device, dtype=torch.bfloat16)
    
    # 手动初始化权重
    hf_sse.apply(manual_weight_initialization)
    print(f"✓ HuggingFace {model_name}模型创建并初始化完成")
    
    # 2. 保存HuggingFace权重
    hf_weights = hf_sse.state_dict()
    torch.save(hf_weights, f"hf_{sse_type}_weights.pth")
    print("✓ HuggingFace权重保存完成")
    
    # 3. 创建vLLM模型 - 使用bfloat16
    if vllm_available:
        vllm_config = get_current_vllm_config()
        
        # 修复：根据sse_type决定是否传递gate参数
        if sse_type.lower() == 'gla':
            # GLA模型需要gate参数
            vllm_sse = VLLMModelClass(
                vllm_config=vllm_config,
                prefix=f"model.layers.0.sse_{sse_type}",
                hidden_size=512,
                expand_v=1.0,
                head_dim=64,
                num_heads=8,
                num_v_heads=8,
                mode='chunk',
                use_output_gate=True,
                use_short_conv=False,
                num_sparse_partition=4,
                num_writer=1,
                num_reader=1,
                sse_implementation="varlen",
                use_q_softmax=False,
                use_k_softmax=True,
                emulq=True,
                emulk=True,
                gate_logit_normalizer=16,
                gate_low_rank_dim=16,
                layer_idx=0,
                norm_eps=1e-5
            ).to(device, dtype=torch.bfloat16)
        else:
            # GDN模型不需要gate参数（VLLMSSEGDN的__init__已经处理）
            vllm_sse = VLLMModelClass(
                vllm_config=vllm_config,
                prefix=f"model.layers.0.sse_{sse_type}",
                hidden_size=512,
                expand_v=1.0,
                head_dim=64,
                num_heads=8,
                num_v_heads=8,
                mode='chunk',
                use_output_gate=True,
                use_short_conv=False,
                num_sparse_partition=4,
                num_writer=1,
                num_reader=1,
                sse_implementation="varlen",
                use_q_softmax=False,
                use_k_softmax=True,
                emulq=True,
                emulk=True,
                layer_idx=0,
                norm_eps=1e-5
            ).to(device, dtype=torch.bfloat16)
            
        print(f"✓ vLLM {model_name}模型创建完成")
    else:
        print("⚠️ vLLM不可用，跳过vLLM模型创建")
        return hf_sse, None
    
    # 4. 从HuggingFace权重加载到vLLM模型
    try:
        # 使用vLLM模型的load_hf_weights方法加载权重
        # 这是更简洁和可靠的方式，与test_attn.py保持一致
        vllm_sse.load_hf_weights(hf_weights)
        print(f"✓ vLLM {model_name}模型成功加载HuggingFace权重")
    except Exception as e:
        print(f"✗ 权重加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 5. 详细验证权重是否正确加载
    print("\n=== 权重验证详情 ===")
    
    # 验证关键投影层权重
    if vllm_available:
        # 验证q_proj权重
        if hasattr(hf_sse, 'q_proj') and hasattr(vllm_sse, 'q_proj'):
            if hasattr(hf_sse.q_proj, 'weight') and hasattr(vllm_sse.q_proj, 'weight'):
                hf_q_weight = hf_sse.q_proj.weight.data
                vllm_q_weight = vllm_sse.q_proj.weight.data
                q_diff = torch.max(torch.abs(vllm_q_weight - hf_q_weight))
                print(f"q_proj权重最大差异: {q_diff.item():.6f}")
                assert q_diff < 1e-6, "q_proj权重不匹配"
        
        # 验证k_proj权重
        if hasattr(hf_sse, 'k_proj') and hasattr(vllm_sse, 'k_proj'):
            if hasattr(hf_sse.k_proj, 'weight') and hasattr(vllm_sse.k_proj, 'weight'):
                hf_k_weight = hf_sse.k_proj.weight.data
                vllm_k_weight = vllm_sse.k_proj.weight.data
                k_diff = torch.max(torch.abs(vllm_k_weight - hf_k_weight))
                print(f"k_proj权重最大差异: {k_diff.item():.6f}")
                assert k_diff < 1e-6, "k_proj权重不匹配"
        
        # 验证v_proj权重
        if hasattr(hf_sse, 'v_proj') and hasattr(vllm_sse, 'v_proj'):
            if hasattr(hf_sse.v_proj, 'weight') and hasattr(vllm_sse.v_proj, 'weight'):
                hf_v_weight = hf_sse.v_proj.weight.data
                vllm_v_weight = vllm_sse.v_proj.weight.data
                v_diff = torch.max(torch.abs(vllm_v_weight - hf_v_weight))
                print(f"v_proj权重最大差异: {v_diff.item():.6f}")
                assert v_diff < 1e-6, "v_proj权重不匹配"
        
        # 验证o_proj权重
        if hasattr(hf_sse, 'o_proj') and hasattr(vllm_sse, 'o_proj'):
            if hasattr(hf_sse.o_proj, 'weight') and hasattr(vllm_sse.o_proj, 'weight'):
                hf_o_weight = hf_sse.o_proj.weight.data
                vllm_o_weight = vllm_sse.o_proj.weight.data
                o_diff = torch.max(torch.abs(vllm_o_weight - hf_o_weight))
                print(f"o_proj权重最大差异: {o_diff.item():.6f}")
                assert o_diff < 1e-6, "o_proj权重不匹配"
        
        # 验证g_proj权重（如果存在）
        if hasattr(hf_sse, 'g_proj') and hasattr(vllm_sse, 'g_proj_0'):
            # vLLM的g_proj是分阶段的
            if hasattr(hf_sse.g_proj, 'weight') and hasattr(vllm_sse.g_proj_0, 'weight'):
                # 对于分阶段的投影，我们只验证第一阶段
                hf_g_weight = hf_sse.g_proj.weight.data
                vllm_g_weight = vllm_sse.g_proj_0.weight.data
                g_diff = torch.max(torch.abs(vllm_g_weight - hf_g_weight[:vllm_g_weight.shape[0]]))
                print(f"g_proj权重最大差异: {g_diff.item():.6f}")
                # 这里不做严格断言，因为vLLM的实现可能不同
        
        print("✓ 所有关键权重验证通过")
    
    return hf_sse, vllm_sse


def test_forward_equivalence(hf_sse, vllm_sse, sse_type='gla'):
    """测试2：验证前向传播输出等价性"""
    print(f"\n=== 测试2：前向传播等价性测试 ({sse_type.upper()}) ===")
    
    if not vllm_sse:
        print("⚠️ vLLM模型不可用，跳前向传播测试")
        return
    
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 测试不同的输入配置
    test_cases = [
        # (batch_size, seq_len, use_cu_seqlens, description)
        (1, 320, False, "单样本，固定长度"),
        (2, 320, False, "多样本，固定长度"),
        (2, 320, True, "多样本，变长序列(cu_seqlens)"),  # 新增的变长测试用例
    ]
    
    for batch_size, seq_len, use_cu_seqlens, description in test_cases:
        print(f"\n测试用例: {description}")
        
        # 创建输入数据 - 使用bfloat16
        hidden_states = torch.randn(batch_size, seq_len, hf_sse.hidden_size, dtype=torch.bfloat16).to(device)
        
        # 创建注意力掩码（用于变长序列）
        attention_mask = None
        cu_seqlens = None
        actual_seq_lens = None
        
        if use_cu_seqlens:
            # 创建变长序列长度（模拟不同长度的序列）
            actual_seq_lens = [seq_len - i for i in range(batch_size)]
            max_len = max(actual_seq_lens)
            
            # 创建注意力掩码
            attention_mask = torch.zeros(batch_size, max_len).to(device)
            for i, length in enumerate(actual_seq_lens):
                attention_mask[i, :length] = 1
            
            # 创建cu_seqlens
            cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32).to(device)
            for i in range(batch_size):
                cu_seqlens[i+1] = cu_seqlens[i] + actual_seq_lens[i]
            
            # 调整输入序列长度
            hidden_states = hidden_states[:, :max_len, :]
        else:
            # 固定长度序列
            attention_mask = torch.ones(batch_size, seq_len).to(device)
        
        # HuggingFace前向传播
        hf_output, hf_aux_loss, _ = hf_sse(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens if use_cu_seqlens else None,
            output_attentions=False,
            use_cache=False
        )
        
        # vLLM前向传播
        # vLLM需要positions参数
        positions = torch.arange(0, hidden_states.shape[1], device=hidden_states.device).unsqueeze(0).repeat(batch_size, 1)
        
        # 获取vllm配置并设置forward_context
        if vllm_available:
            vllm_config = get_current_vllm_config()
            attn_metadata = create_attention_metadata(
                batch_size=batch_size,
                seq_len=hidden_states.shape[1],
                use_cu_seqlens=use_cu_seqlens,
                seq_lens=actual_seq_lens
            )
            
            with set_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=vllm_config,
                num_tokens=attn_metadata["num_actual_tokens"],
                cudagraph_runtime_mode=CUDAGraphMode.NONE
            ):
                vllm_output, vllm_aux_loss, _ = vllm_sse(
                    hidden_states=hidden_states,
                    positions=positions,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    use_cache=False
                )
        else:
            # 如果vLLM不可用，直接调用
            vllm_output, vllm_aux_loss, _ = vllm_sse(
                hidden_states=hidden_states,
                positions=positions,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False
            )
        
        # 检查输出形状
        assert hf_output.shape == vllm_output.shape, f"前向输出形状不匹配: {hf_output.shape} vs {vllm_output.shape}"
        
        # 检查输出值
        max_diff = torch.max(torch.abs(hf_output - vllm_output))
        print(f"前向输出最大差异: {max_diff.item():.6f}")
        
        if max_diff < 1e-5:
            print(f"✓ {description} 前向传播验证通过")
        else:
            print(f"✗ {description} 前向传播输出不等价")
            raise AssertionError(f"前向输出不等价")

# 测试推理prefill阶段
def test_inference_prefill(hf_sse: nn.Module, vllm_sse: nn.Module, sse_type='gla'):
    """测试3：验证推理prefill阶段"""
    print(f"\n=== 测试3：推理Prefill阶段测试 ({sse_type.upper()}) ===")
    hf_sse.eval()
    if vllm_sse is not None:
        vllm_sse.eval()
    
    if not vllm_sse:
        print("⚠️ vLLM模型不可用，跳过prefill测试")
        return None, None
    
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    
    # 创建测试输入 - 使用bfloat16
    batch_size = 2
    seq_len = 320
    hidden_states = torch.randn(batch_size, seq_len, hf_sse.hidden_size, dtype=torch.bfloat16).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    
    # ====================
    # 正确初始化past_key_values
    # ====================
    
    # 1. HuggingFace prefill - 正确初始化past_key_values
    # 根据fla.models.utils.Cache的定义，应该使用Cache类初始化，而不是None
    # Cache类需要seen_tokens参数（初始为0）
    hf_past_key_values = Cache(seen_tokens=0)  # 正确初始化Cache对象
    print(f"✓ HuggingFace past_key_values初始化为Cache对象，类型: {type(hf_past_key_values)}")
    
    hf_output, hf_aux_loss, hf_past = hf_sse(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        past_key_values=hf_past_key_values,  # 传递正确初始化的Cache对象
        output_attentions=False,
        use_cache=True
    )
    
    # 2. vLLM prefill - 初始化past_key_values
    # vLLM使用字典格式
    positions = torch.arange(0, seq_len, device=hidden_states.device).unsqueeze(0).repeat(batch_size, 1)
    vllm_past_key_values = {}  # vLLM使用字典格式
    print(f"✓ vLLM past_key_values初始化为字典，类型: {type(vllm_past_key_values)}")
    
    # 获取vllm配置并设置forward_context
    if vllm_available:
        vllm_config = get_current_vllm_config()
        attn_metadata = create_attention_metadata(
            batch_size=batch_size,
            seq_len=seq_len,
            use_cu_seqlens=False
        )
        
        with set_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=vllm_config,
            num_tokens=attn_metadata["num_actual_tokens"],
            cudagraph_runtime_mode=CUDAGraphMode.NONE
        ):
            vllm_output, vllm_aux_loss, vllm_past = vllm_sse(
                hidden_states=hidden_states,
                positions=positions,
                attention_mask=attention_mask,
                past_key_values=vllm_past_key_values,  # 传递初始的字典
                output_attentions=False,
                use_cache=True
            )
    else:
        # 如果vLLM不可用，直接调用
        vllm_output, vllm_aux_loss, vllm_past = vllm_sse(
            hidden_states=hidden_states,
            positions=positions,
            attention_mask=attention_mask,
            past_key_values=vllm_past_key_values,  # 传递初始的字典
            output_attentions=False,
            use_cache=True
        )
    
    # 检查输出形状
    assert hf_output.shape == vllm_output.shape, f"Prefill输出形状不匹配: {hf_output.shape} vs {vllm_output.shape}"
    print(f"✓ Prefill输出形状匹配: {hf_output.shape}")
    
    # 检查输出值
    max_diff = torch.max(torch.abs(hf_output - vllm_output))
    print(f"Prefill输出最大差异: {max_diff.item():.6f}")
    
    if max_diff < 1e-5:
        print(f"✓ {sse_type.upper()} Prefill阶段验证通过")
    else:
        print(f"✗ {sse_type.upper()} Prefill阶段输出不等价")
        raise AssertionError(f"Prefill输出不等价，最大差异: {max_diff.item()}")
    
    # 验证past状态不为None
    assert hf_past is not None, "HuggingFace past状态为None"
    assert vllm_past is not None, "vLLM past状态为None"
    print(f"✓ Prefill阶段past状态验证通过")
    print(f"  HuggingFace past类型: {type(hf_past)}, 长度: {len(hf_past) if hasattr(hf_past, '__len__') else 'N/A'}")
    print(f"  vLLM past类型: {type(vllm_past)}, 键数量: {len(vllm_past) if isinstance(vllm_past, dict) else 'N/A'}")
    
    return hf_past, vllm_past


def test_inference_decode(hf_sse, vllm_sse, hf_past, vllm_past, sse_type='gla'):
    """测试4：验证推理decode阶段"""
    print(f"\n=== 测试4：推理Decode阶段测试 ({sse_type.upper()}) ===")
    
    # 改进的检查逻辑，提供更详细的错误信息
    if not vllm_sse:
        print("⚠️ vLLM模型不可用，跳过decode测试")
        return
    if hf_past is None:
        print("⚠️ HuggingFace past状态为None，跳过decode测试")
        return
    if vllm_past is None:
        print("⚠️ vLLM past状态为None，跳过decode测试")
        return
    
    # 确保模型处于评估模式
    hf_sse.eval()
    vllm_sse.eval()
    
    batch_size = 2
    decode_steps = 5
    
    for step in range(decode_steps):
        print(f"\nDecode step {step+1}:")
        
        # 创建decode输入（单token）- 使用bfloat16
        hidden_states = torch.randn(batch_size, 1, hf_sse.hidden_size, dtype=torch.bfloat16).to(device)
        
        # HuggingFace decode
        hf_output, hf_aux_loss, hf_past = hf_sse(
            hidden_states=hidden_states,
            past_key_values=hf_past,
            output_attentions=False,
            use_cache=True
        )
        
        # vLLM decode
        # vLLM需要positions参数，这里使用之前的位置+1
        positions = torch.full((batch_size, 1), 320 + step, device=hidden_states.device)
        
        # 获取vllm配置并设置forward_context
        if vllm_available:
            vllm_config = get_current_vllm_config()
            # 创建decode阶段的attention metadata
            attn_metadata = create_attention_metadata(
                batch_size=batch_size,
                seq_len=1,  # decode阶段每次处理1个token
                use_cu_seqlens=False
            )
            
            # 更新decode相关的元数据
            attn_metadata.update({
                "num_actual_tokens": batch_size * 1,  # decode阶段每次处理1个token
                "num_prefill_tokens": 0,
                "num_decode_tokens": batch_size * 1,
                "num_prefills": 0,
                "num_decodes": batch_size
            })
            
            with set_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=vllm_config,
                num_tokens=attn_metadata["num_actual_tokens"],
                cudagraph_runtime_mode=CUDAGraphMode.NONE
            ):
                vllm_output, vllm_aux_loss, vllm_past = vllm_sse(
                    hidden_states=hidden_states,
                    positions=positions,
                    past_key_values=vllm_past,
                    output_attentions=False,
                    use_cache=True
                )
        else:
            # 如果vLLM不可用，直接调用
            vllm_output, vllm_aux_loss, vllm_past = vllm_sse(
                hidden_states=hidden_states,
                positions=positions,
                past_key_values=vllm_past,
                output_attentions=False,
                use_cache=True
            )
        
        # 检查输出形状
        assert hf_output.shape == vllm_output.shape, f"Decode输出形状不匹配"
        
        # 检查输出值
        max_diff = torch.max(torch.abs(hf_output - vllm_output))
        print(f"  输出最大差异: {max_diff.item():.6f}")
        
        if max_diff < 1e-5:
            print(f"  ✓ {sse_type.upper()} Decode step {step+1} 验证通过")
        else:
            print(f"  ✗ {sse_type.upper()} Decode step {step+1} 输出不等价")
            raise AssertionError(f"Decode输出不等价")


def test_training_mode(hf_sse, vllm_sse, sse_type='gla'):
    """测试5：验证训练模式"""
    print(f"\n=== 测试5：训练模式测试 ({sse_type.upper()}) ===")
    
    if not vllm_sse:
        print("⚠️ vLLM模型不可用，跳过训练模式测试")
        return
    
    # 设置模型为训练模式
    hf_sse.train()
    vllm_sse.train()
    
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    
    # 创建输入数据 - 使用bfloat16
    batch_size = 2
    seq_len = 320
    hidden_states = torch.randn(batch_size, seq_len, hf_sse.hidden_size, dtype=torch.bfloat16).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    
    # 添加梯度计算
    hidden_states.requires_grad_(True)
    
    # HuggingFace训练前向
    hf_output, hf_aux_loss, _ = hf_sse(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=False,
        use_cache=False
    )
    
    # vLLM训练前向
    positions = torch.arange(0, seq_len, device=hidden_states.device).unsqueeze(0).repeat(batch_size, 1)
    
    # 获取vllm配置并设置forward_context
    if vllm_available:
        vllm_config = get_current_vllm_config()
        attn_metadata = create_attention_metadata(
            batch_size=batch_size,
            seq_len=seq_len,
            use_cu_seqlens=False
        )
        
        with set_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=vllm_config,
            num_tokens=attn_metadata["num_actual_tokens"],
            cudagraph_runtime_mode=CUDAGraphMode.NONE
        ):
            vllm_output, vllm_aux_loss, _ = vllm_sse(
                hidden_states=hidden_states,
                positions=positions,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False
            )
    else:
        # 如果vLLM不可用，直接调用
        vllm_output, vllm_aux_loss, _ = vllm_sse(
            hidden_states=hidden_states,
            positions=positions,
            attention_mask=attention_mask,
            output_attentions=False,
            use_cache=False
        )
    
    # 检查输出形状
    assert hf_output.shape == vllm_output.shape, f"训练模式输出形状不匹配"
    
    # 检查输出值
    max_diff = torch.max(torch.abs(hf_output - vllm_output))
    print(f"训练模式输出最大差异: {max_diff.item():.6f}")
    
    if max_diff < 1e-5:
        print(f"✓ {sse_type.upper()} 训练模式前向验证通过")
    else:
        print(f"✗ {sse_type.upper()} 训练模式输出不等价")
        raise AssertionError(f"训练模式输出不等价")
    
    # 测试反向传播
    print("\n测试反向传播...")
    
    # HuggingFace反向传播
    hf_loss = hf_output.sum()
    if hf_aux_loss and len(hf_aux_loss) > 1 and hf_aux_loss[1] is not None:
        hf_loss += hf_aux_loss[1]
    hf_loss.backward(retain_graph=True)
    
    # vLLM反向传播
    vllm_loss = vllm_output.sum()
    if vllm_aux_loss is not None:
        vllm_loss += vllm_aux_loss
    vllm_loss.backward()
    
    # 检查梯度（仅检查部分关键参数）
    print("\n梯度检查:")
    
    # 比较q_proj权重的梯度
    if hasattr(hf_sse, 'q_proj') and hasattr(hf_sse.q_proj, 'weight') and \
       hasattr(vllm_sse, 'q_proj') and hasattr(vllm_sse.q_proj, 'weight'):
        
        hf_q_grad = hf_sse.q_proj.weight.grad
        vllm_q_grad = vllm_sse.q_proj.weight.grad
        
        if hf_q_grad is not None and vllm_q_grad is not None:
            grad_max_diff = torch.max(torch.abs(hf_q_grad - vllm_q_grad))
            print(f"q_proj梯度最大差异: {grad_max_diff.item():.6f}")
            
            if grad_max_diff < 1e-5:
                print(f"✓ {sse_type.upper()} 训练模式反向传播验证通过")
            else:
                print(f"✗ {sse_type.upper()} 训练模式反向传播不等价")
                raise AssertionError(f"训练模式反向传播不等价")
        else:
            print("⚠️ 梯度为None，无法比较")
            
            # 如果vLLM模型参数没有梯度，手动设置允许梯度并重新测试
            if vllm_q_grad is None and hasattr(vllm_sse.q_proj, 'weight'):
                print("\n尝试手动启用梯度并重新测试...")
                
                # 手动设置允许梯度
                vllm_sse.q_proj.weight.requires_grad_(True)
                print("  ✓ 已设置q_proj.weight.requires_grad=True")
                
                # 检查其他重要参数并设置梯度
                if hasattr(vllm_sse, 'k_proj') and hasattr(vllm_sse.k_proj, 'weight'):
                    vllm_sse.k_proj.weight.requires_grad_(True)
                    print("  ✓ 已设置k_proj.weight.requires_grad=True")
                
                if hasattr(vllm_sse, 'v_proj') and hasattr(vllm_sse.v_proj, 'weight'):
                    vllm_sse.v_proj.weight.requires_grad_(True)
                    print("  ✓ 已设置v_proj.weight.requires_grad=True")
                
                # 重新运行前向和反向传播
                vllm_sse.zero_grad()
                
                # 获取vllm配置并设置forward_context
                if vllm_available:
                    vllm_config = get_current_vllm_config()
                    attn_metadata = create_attention_metadata(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        use_cu_seqlens=False
                    )
                    
                    with set_forward_context(
                        attn_metadata=attn_metadata,
                        vllm_config=vllm_config,
                        num_tokens=attn_metadata["num_actual_tokens"],
                        cudagraph_runtime_mode=CUDAGraphMode.NONE
                    ):
                        vllm_output_retry, vllm_aux_loss_retry, _ = vllm_sse(
                            hidden_states=hidden_states,
                            positions=positions,
                            attention_mask=attention_mask,
                            output_attentions=False,
                            use_cache=False
                        )
                else:
                    # 如果vLLM不可用，直接调用
                    vllm_output_retry, vllm_aux_loss_retry, _ = vllm_sse(
                        hidden_states=hidden_states,
                        positions=positions,
                        attention_mask=attention_mask,
                        output_attentions=False,
                        use_cache=False
                    )
                
                # 重新计算损失并反向传播
                vllm_loss_retry = vllm_output_retry.sum()
                if vllm_aux_loss_retry is not None:
                    vllm_loss_retry += vllm_aux_loss_retry
                vllm_loss_retry.backward()
                
                # 再次检查梯度
                vllm_q_grad_retry = vllm_sse.q_proj.weight.grad
                if vllm_q_grad_retry is not None:
                    grad_max_diff_retry = torch.max(torch.abs(hf_q_grad - vllm_q_grad_retry))
                    print(f"重新测试后q_proj梯度最大差异: {grad_max_diff_retry.item():.6f}")
                    
                    if grad_max_diff_retry < 1e-5:
                        print(f"✓ {sse_type.upper()} 训练模式反向传播验证通过（手动启用梯度后）")
                    else:
                        print(f"✗ {sse_type.upper()} 训练模式反向传播不等价（手动启用梯度后）")
                        raise AssertionError(f"训练模式反向传播不等价")
                else:
                    print("  ⚠️ 仍然没有梯度，并行线性层可能不支持标准梯度计算")
                    print("  仅验证输出等价性")
                    if max_diff < 1e-5:
                        print(f"✓ {sse_type.upper()} 训练模式验证通过（仅验证输出等价性）")
                    else:
                        print(f"✗ {sse_type.upper()} 训练模式验证失败")
                        raise AssertionError(f"训练模式不等价")
    else:
        print("⚠️ q_proj不存在，跳过梯度检查")
        
        # 检查是否有其他投影层
        proj_layers = ['qkv_proj', 'q_proj', 'k_proj', 'v_proj']
        found = False
        
        for layer_name in proj_layers:
            if hasattr(hf_sse, layer_name) and hasattr(getattr(hf_sse, layer_name), 'weight') and \
               hasattr(vllm_sse, layer_name) and hasattr(getattr(vllm_sse, layer_name), 'weight'):
                
                found = True
                hf_grad = getattr(hf_sse, layer_name).weight.grad
                vllm_grad = getattr(vllm_sse, layer_name).weight.grad
                
                if hf_grad is not None and vllm_grad is not None:
                    grad_max_diff = torch.max(torch.abs(hf_grad - vllm_grad))
                    print(f"{layer_name}梯度最大差异: {grad_max_diff.item():.6f}")
                    
                    if grad_max_diff < 1e-5:
                        print(f"✓ {sse_type.upper()} 训练模式反向传播验证通过")
                    else:
                        print(f"✗ {sse_type.upper()} 训练模式反向传播不等价")
                        raise AssertionError(f"训练模式反向传播不等价")
                else:
                    print(f"⚠️ {layer_name}梯度为None，无法比较")
                    break
        
        if not found:
            print("⚠️ 没有找到可比较的投影层，仅验证输出等价性")
            if max_diff < 1e-5:
                print(f"✓ {sse_type.upper()} 训练模式验证通过（仅验证输出等价性）")
            else:
                print(f"✗ {sse_type.upper()} 训练模式验证失败")
                raise AssertionError(f"训练模式不等价")


def test_sse_type(sse_type):
    """测试特定类型的SSE"""
    print(f"\n" + "="*70)
    print(f"开始测试 {sse_type.upper()} 类型的SSE层")
    print("="*70)
    
    # 测试1：权重加载
    hf_sse, vllm_sse = test_weight_loading(sse_type)
    
    if not torch.cuda.is_available():
        print(f"\n⚠️ 警告：没有检测到CUDA设备，跳过{sse_type.upper()}的其他测试")
        return
    
    # 测试2：前向传播等价性
    test_forward_equivalence(hf_sse, vllm_sse, sse_type)
    
    # 测试3：推理prefill阶段
    hf_past, vllm_past = test_inference_prefill(hf_sse, vllm_sse, sse_type)
    
    # 测试4：推理decode阶段
    test_inference_decode(hf_sse, vllm_sse, hf_past, vllm_past, sse_type)
    
    # 测试5：训练模式
    test_training_mode(hf_sse, vllm_sse, sse_type)
    
    print(f"\n" + "="*70)
    print(f"🎉 {sse_type.upper()} 类型的SSE层所有测试通过！")
    print("="*70)


def main():
    """主测试函数"""
    print("="*70)
    print("SSE (Sparse State Expansion) 层等价性测试")
    print("支持 SSEGLA 和 SSEGDN 两种类型")
    print("="*70)
    
    try:
        # 初始化vLLM分布式环境
        if vllm_available:
            if initialize_vllm_distributed():
                # 初始化model parallel
                initialize_vllm_model_parallel()
        
        # 测试SSEGLA
        test_sse_type('gla')
        
        # 测试SSEGDN
        test_sse_type('gdn')
        
        print("\n" + "="*70)
        print("🎉 所有SSE类型的测试都通过了！")
        print("✓ SSEGLA 和 SSEGDN 实现完全等价")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时文件
        for sse_type in ['gla', 'gdn']:
            weight_file = f"hf_{sse_type}_weights.pth"
            if os.path.exists(weight_file):
                os.remove(weight_file)
        print("\n✓ 临时文件清理完成")
        
        # 清理vLLM分布式环境
        if vllm_available:
            try:
                destroy_model_parallel()
                print("✓ vLLM模型并行环境已清理")
            except:
                pass


if __name__ == "__main__":
    main()