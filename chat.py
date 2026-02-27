import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig

from sse_swa_moba_hf.configuration_sse_swa_moba_hf import SSESWAMoBAConfig
from sse_swa_moba_hf.modeling_sse_swa_moba_hf import SSESWAMoBAForCausalLM, SSESWAMoBAModel

AutoConfig.register(SSESWAMoBAConfig.model_type, SSESWAMoBAConfig, exist_ok=True)
AutoModel.register(SSESWAMoBAConfig, SSESWAMoBAModel, exist_ok=True)
AutoModelForCausalLM.register(SSESWAMoBAConfig, SSESWAMoBAForCausalLM, exist_ok=True)

# =====================
# 1. 修改成你的模型路径
# =====================
# model_name = "Qwen/Qwen2-1.5B-Instruct"   # 也可以换成本地路径
model_name = input("输入路径：")
# =====================
# 2. 加载 tokenizer
# =====================
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# =====================
# 3. 加载模型
# =====================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

# =====================
# 4. 构造对话历史
# =====================
messages = [
]

def chat():
    print("输入 exit 退出")
    print(model.device)
    # while True:
    if True:
        # user_input = input("\nUser: ")
        user_input = "你好，请介绍一下你自己。"
        # if user_input.lower() == "exit":
        #     break
        
        messages.append({"role": "user", "content": user_input})
        
        # 使用 chat template 构造输入
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        print(len(input_ids), input_ids, tokenizer.decode(input_ids[0], skip_special_tokens=True))
        torch.cuda.set_device(model.device)
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=128,
                # do_sample=True,
                temperature=0.7,
                # top_p=0.9
            )
        
        # 只取新生成的部分
        response_ids = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        print(f"\nAssistant: {response}")
        
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    chat()