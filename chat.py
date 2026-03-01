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
# tokenizer = AutoTokenizer.from_pretrained(
#     model_name,
#     trust_remote_code=True
# )

PATH = '/mnt/jfzn/models/Qwen3-4B-Thinking-2507'

# import fla
# PATH = '/mnt/jfzn/sbh/train_exp/transformer-1B-100B'

tokenizer = AutoTokenizer.from_pretrained(PATH,
                                        padding_side='left',
                                        truncation_side='left',
                                        trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =====================
# 3. 加载模型
# =====================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda()

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
        # user_input = "你好，请介绍一下你自己。"
        prompts = [
            # "What is the best thing to do in San Francisco? The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\nQuestion: What is the best thing to do in San Francisco?",
            # "Explain java fx basic concepts",
            # "Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
            "你好，请介绍一下你自己。"
        ]
        # if user_input.lower() == "exit":
        #     break
        
        # messages.append({"role": "user", "content": user_input})
        
        # 使用 chat template 构造输入
        conversation_prompts = [
            [{'role' : 'user', 'content' : prompt}] for prompt in prompts
        ]
        prompts = [
            tokenizer.apply_chat_template(conversation_prompt, tokenize = False,  \
            add_generation_prompt=True) for conversation_prompt in conversation_prompts
        ]

        for prompt in prompts :
            print(prompt)
            print('==============')
        inputs = tokenizer(prompt, padding=True, truncation=True, max_length=65536, return_tensors='pt').to('cuda')
        print("input shape: ", inputs.input_ids.shape)

        # print(len(input_ids), input_ids, tokenizer.decode(input_ids[0], skip_special_tokens=True))
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
            do_sample=True,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range(len(prompts)):
            print("=====Begin======")
            print(response[i])
            print("======End======")


if __name__ == "__main__":
    chat()