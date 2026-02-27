import torch
import os
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig

from sse_swa_moba_hf.configuration_sse_swa_moba_hf import SSESWAMoBAConfig
from sse_swa_moba_hf.modeling_sse_swa_moba_hf import SSESWAMoBAForCausalLM, SSESWAMoBAModel

AutoConfig.register(SSESWAMoBAConfig.model_type, SSESWAMoBAConfig, exist_ok=True)
AutoModel.register(SSESWAMoBAConfig, SSESWAMoBAModel, exist_ok=True)
AutoModelForCausalLM.register(SSESWAMoBAConfig, SSESWAMoBAForCausalLM, exist_ok=True)

def save_model_param_names(model_name_or_path, output_file="weights_pureswa1.txt"):
    """
    使用AutoModel加载模型（支持.bin格式权重），提取所有参数名称并保存到指定文件
    
    Args:
        model_name_or_path (str): 模型名称（如bert-base-chinese）或包含.bin权重的文件夹路径
        output_file (str): 输出参数名称的文件路径，默认是当前目录的 weights.txt
    """
    try:
        # 加载模型配置（避免下载预训练权重）
        config = AutoConfig.from_pretrained(model_name_or_path)
        
        # 从指定路径加载模型（自动识别.bin权重文件）
        # map_location='cpu' 确保无GPU也能运行
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
        )
        
        # 提取所有可训练参数的名称
        # named_parameters() 只返回可训练参数，named_parameters(recurse=True) 递归获取所有层
        param_names = [name for name, _ in model.named_parameters(recurse=True)]
        
        # 提取模型所有参数（包括不可训练的，如LayerNorm的running_mean等）
        # 如需包含所有参数，可替换为：param_names = [name for name, _ in model.named_parameters()] + [name for name, _ in model.named_buffers()]
        
        # 将参数名称写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, name in enumerate(param_names, 1):
                f.write(f"{name}\n")
        
        print(f"✅ 成功！共提取到 {len(param_names)} 个参数名称")
        print(f"📄 参数名称已保存到：{os.path.abspath(output_file)}")

        model.save_pretrained('/mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_moba_gdn_u1to3_pureSwa_1.7b_dense_lr3en5_min0p1_bsz64_ep1_aux1en3_pt_data_800k/modeling2')
        
    except Exception as e:
        print(f"❌ 加载模型失败：{e}")
        print("\n请检查：")
        print("1. 输入的模型名称/路径是否正确（如bert-base-chinese）")
        print("2. 文件夹中是否包含pytorch_model.bin和config.json文件")
        print("3. 已安装最新版transformers：pip install --upgrade transformers")


# 主程序入口
if __name__ == "__main__":
    # 输入模型名称（如bert-base-chinese）或包含.bin权重的文件夹路径
    model_path = input("请输入模型名称/包含.bin权重的文件夹路径：").strip()
    
    # 调用函数执行保存操作
    save_model_param_names(model_path)

# /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k_aux1en4/modeling
