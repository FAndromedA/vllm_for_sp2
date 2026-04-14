# vllm_for_sp2 (`sse_swa_moba_vllm`)

这是一个 **vLLM 插件式扩展包**：用于将自定义模型结构注册进 vLLM，并提供/覆盖对应的 attention backend（当前将 `FLASH_ATTN` 指向自定义实现）。

- **包名**：`sse_swa_moba_vllm`
- **vLLM 插件入口**：`vllm.general_plugins = sse_swa_moba_model = "sse_swa_moba_vllm:register_model"`（见 `pyproject.toml`）
- **注册的模型**（见 `sse_swa_moba_vllm/sse_swa_moba/__init__.py`）：
  - `SSESWAMoBAForCausalLM`
  - `SPB2VLForConditionalGeneration`（多模态）

> 适用场景：你有一份 `transformers`/HF 形式的 checkpoint（`--trust-remote-code`），希望用 vLLM 进行 OpenAI 兼容服务、长上下文吞吐/延迟评测等。

---

## 环境与依赖

该仓库依赖 GPU 环境（CUDA）以及 vLLM + Triton/FlashAttention 生态。`requirements.txt` 中包含：

- `torch>=2.10.0`
- `transformers>=4.57.0`
- `vllm==0.17.1`
- `triton==3.6.0`
- `flash_attn==2.8.3`
- `flash-linear-attention==0.4.0`
- `flash_moba==2.0.0`

**建议**：在隔离环境（conda/venv）中安装，并确保 CUDA / 驱动与 `torch`、`flash_attn` 编译环境匹配。

---

## 安装

在仓库根目录执行（推荐可编辑安装，方便改代码立即生效）：

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

可选：快速验证插件入口是否可导入：

```bash
python -c "import sse_swa_moba_vllm; sse_swa_moba_vllm.register_model()"
```

正常情况下你会看到类似：

- `vllm plugin register model successfully.`
- `vllm plugin register attn_backend successfully.`

---

## 快速开始：启动 vLLM OpenAI 兼容服务

### 启动（文本模型）

下面命令来自仓库的实践记录（`notes.md`），按需调整 `--model` 路径、并行度与端口：

```bash
vllm serve /path/to/your/model \
  --served-model-name SSE_SWA_MOBA \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 8 \
  --block-size 128 \
  --dtype bfloat16 \
  --port 8711 \
  --trust-remote-code \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-block-size 8 \
  --max_num_batched_tokens 4096 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

> 说明：参数组合与模型结构/显存有关；`notes.md` 里保留了多组历史命令，遇到 OOM/性能问题可以从那里挑选更保守的配置（例如降低 `--gpu-memory-utilization`、`--max-model-len` 等）。

### 启动（多模态模型）

如果你的模型是多模态（如 `SPB2VL`），可以额外限制每个 prompt 的多模态输入数量：

```bash
vllm serve /path/to/your/mm/model \
  --served-model-name SPB2VL \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 8 \
  --block-size 128 \
  --dtype bfloat16 \
  --port 8711 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 16, "video": 2}' \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-block-size 8 \
  --max_num_batched_tokens 4096 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

---

## 调用方式（OpenAI 兼容）

### curl 调用

Chat Completions：

```bash
curl http://localhost:8711/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "SSE_SWA_MOBA",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "你好，请介绍一下你自己。"}
        ],
        "temperature": 0.7,
        "max_tokens": 128
      }'
```

Completions（用于一些基准脚本/流式测 TTFT）：

```bash
curl http://localhost:8711/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SSE_SWA_MOBA",
    "prompt": "你好，请介绍一下你自",
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

### Python 客户端（文本多轮）

仓库提供了一个最小可用的多轮示例：`chat_with_vllm.py`

```bash
python chat_with_vllm.py
```

它使用 `OpenAI(base_url="http://localhost:8711/v1")` 调用 vLLM，并通过 `extra_body` 传入 `repetition_penalty`。

### Python 客户端（多模态多轮）

仓库提供：`chat_with_multimodal_vllm.py`，支持：

- 纯文本：`你好`
- 图文：`/img /abs/path/to/example.jpg 这张图里有什么`

```bash
python chat_with_multimodal_vllm.py --model SPB2VL
```

也可改 `--base-url`、`--max-tokens` 等参数。

---

## Benchmark：TTFT / TPOT（长上下文）

脚本：`benchmark_vllm_ttft_tpot.py`

功能：

- **TTFT**：从发起请求到收到首个非空流式 token 的时间
- **TPOT（不含首 token）**：\((total\_latency - ttft) / (generated\_tokens - 1)\)

示例（服务启动后执行）：

```bash
python benchmark_vllm_ttft_tpot.py \
  --base-url http://127.0.0.1:8711 \
  --model SSE_SWA_MOBA \
  --model-path /path/to/your/model \
  --runs-per-len 5 \
  --warmup-runs 3 \
  --gen-tokens 100 \
  --force-min-tokens \
  --lengths "1k,2k,4k,8k,16k,32k,64k,128k" \
  --ignore-eos
```

输出：

- 控制台打印汇总表
- `benchmark_results/` 下生成 `*.json` 与 `*.csv`

> 提示：脚本内部为长上下文设置了 `max_model_len = 524288`，如果你要测到 128K/256K/512K，请确保服务端 `--max-model-len` 足够大，否则请求会失败或被截断。

---

## 目录结构（高层）

- `sse_swa_moba_vllm/`：vLLM 插件包（模型注册、attention backend、layer/ops 实现）
  - `__init__.py`：插件入口 `register_model()`；并注册 attention backend override
  - `sse_swa_moba/`：HF config / vLLM registry glue / 模型实现
  - `attention/`、`layers/`：核心算子与模块
- `sse_swa_moba_hf/`：HF 侧的配置与建模代码（用于 `transformers` 加载/对齐）
- `chat_with_vllm.py`：连接 vLLM 的 OpenAI 兼容聊天 demo（文本）
- `chat_with_multimodal_vllm.py`：多模态聊天 demo
- `benchmark_vllm_ttft_tpot.py`：TTFT/TPOT 基准脚本
- `notes.md`：实验命令与参数备忘（serve/bench/lm-eval）

> `build/lib/` 与 `*.pyc` 属于构建产物/缓存，一般不作为源码入口（建议在实际发布或提交前清理/忽略）。

---

## 常见问题（FAQ）

### 1) vLLM 没有识别到自定义模型/报 “unknown model”

- 确认你已执行 `pip install -e .` 或安装了本包
- 运行验证：

```bash
python -c "import sse_swa_moba_vllm; sse_swa_moba_vllm.register_model()"
```

### 2) FlashAttention / Triton 安装失败

这通常与 CUDA、编译工具链、`torch` 版本不匹配相关。建议：

- 使用与 CUDA 版本匹配的 `torch` 轮子
- 确保 `nvcc`、`gcc/g++`、Python 开发头文件齐全
- 在全新环境中重装依赖以避免 ABI 冲突

### 3) 长上下文 OOM 或吞吐不稳定

优先从这些参数入手降低压力：

- 降低 `--max-model-len`、`--max_num_batched_tokens`
- 降低 `--gpu-memory-utilization`
- 调整并行度 `--tensor-parallel-size` / `--pipeline-parallel-size`

仓库的 `notes.md` 里保留了多组可参考的启动/评测命令。

---

## License

仓库目前未包含明确的 LICENSE 文件；如需开源发布，请补充许可证并在此处更新说明。

