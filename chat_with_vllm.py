import os
from openai import OpenAI
# vLLM OpenAI 兼容后端的地址
VLLM_BASE_URL = "http://localhost:8711/v1"
# vLLM 不需要真正的 key，这里随便填一个占位
FAKE_API_KEY = "vllm-api-key"
client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=FAKE_API_KEY,
)
# 这里要和你启动 vLLM 时的 --model 一致
MODEL_NAME = "SSE_SWA_MOBA"
def main():
    print("多轮对话 Demo（使用 vLLM 后端）。输入 'exit' 或 'quit' 结束。\n")
    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": "你是一个乐于助人的人工智能助手，请尽量用中文回答。",
        }
    ]
    while True:
        user_input = input("用户: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("对话结束。")
            break
        if not user_input:
            continue
        # 将用户输入加入历史
        messages.append({"role": "user", "content": user_input})
        try:
            # 调用 vLLM 的 ChatCompletion（OpenAI 兼容）
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7,
                max_tokens=8192,
                extra_body={'repetition_penalty': 1.1},
            )
            assistant_msg = response.choices[0].message.content
            print(f"助手: {assistant_msg}\n")
            # 将回复加入历史，形成多轮对话
            messages.append({"role": "assistant", "content": assistant_msg})
        except Exception as e:
            print(f"调用 vLLM 出错: {e}")
            break
if __name__ == "__main__":
    main()