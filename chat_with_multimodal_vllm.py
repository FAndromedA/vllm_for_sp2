import argparse
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


def _to_data_url(image_path: str) -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _build_user_content(text: str, image: str | None) -> Any:
    if not image:
        return text

    image_url = image if image.startswith(("http://", "https://", "data:")) else _to_data_url(image)
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM OpenAI 接口多模态聊天客户端")
    parser.add_argument("--base-url", default="http://localhost:8711/v1", help="vLLM OpenAI API 地址")
    parser.add_argument("--api-key", default="vllm-api-key", help="占位 API key")
    parser.add_argument("--model", default="SPB2VL", help="模型名（与服务端 --served-model-name 一致）")
    parser.add_argument("--system", default="你是一个乐于助人的人工智能助手，请尽量用中文回答。", help="system prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    parser.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print("多模态多轮对话（vLLM OpenAI 兼容接口）")
    print("输入格式:")
    print("  纯文本: 你好")
    print("  图文: /img /abs/path/to/example.jpg 这张图里有什么")
    print("退出: exit 或 quit\n")

    messages: list[dict[str, Any]] = [{"role": "system", "content": args.system}]
    while True:
        raw = input("用户: ").strip()
        if raw.lower() in {"exit", "quit"}:
            print("对话结束。")
            break
        if not raw:
            continue

        image: str | None = None
        text = raw
        if raw.startswith("/img "):
            parts = raw.split(maxsplit=2)
            if len(parts) < 3:
                print("格式错误。请使用: /img <图片路径或URL> <问题>")
                continue
            image = parts[1]
            text = parts[2]

        try:
            user_content = _build_user_content(text, image)
        except Exception as e:
            print(f"图片处理失败: {e}")
            continue

        messages.append({"role": "user", "content": user_content})
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                extra_body={"repetition_penalty": 1.1},
            )
            assistant_msg = response.choices[0].message.content
            print(f"助手: {assistant_msg}\n")
            messages.append({"role": "assistant", "content": assistant_msg})
        except Exception as e:
            print(f"调用 vLLM 出错: {e}")
            break


if __name__ == "__main__":
    main()