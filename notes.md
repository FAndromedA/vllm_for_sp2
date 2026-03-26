nohup vllm serve /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k_aux1en4/modeling \
	--tensor-parallel-size 4 \
	--max-model-len 65536 \
	--served-model-name SSE_SWA_MOBA \
	--gpu-memory-utilization 0.65 \
	--block-size 128 \
	--dtype bfloat16 \
	--port 8711 \
	--trust-remote-code \
	> _vllm_serve_sse_swa_moba.log 2>&1 &

TORCHDYNAMO_VERBOSE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k_aux1en4/modeling --tensor-parallel-size 4 --max-model-len 65536 --served-model-name SSE_SWA_MOBA --gpu-memory-utilization 0.65 --block-size 128 --dtype bfloat16 --port 8711 --trust-remote-code --enforce-eager > _vllm_serve_sse_swa_moba.log 2>&1


curl http://localhost:8711/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "SSE_SWA_MOBA",
        "messages": [
          {"role": "user", "content": "我想要去中国旅游，帮我规划详尽一个7天的旅游计划"}
        ],
        "temperature": 0.7,
        "max_tokens": 8192,
        "top_p": 0.9,
        "repetition_penalty": 1.1
      }'

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

curl http://localhost:8711/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SSE_SWA_MOBA",
    "prompt": "你好，请介绍一下你自",
    "max_tokens": 128,
    "temperature": 0.2
  }'

source /mnt/jfzn/miniconda3/bin/activate
conda activate v2_vllm
cd /mnt/jfzn/zjh/vllm_for_sp2/

--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

TORCHDYNAMO_VERBOSE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k_aux1en4/modeling --max-model-len 65536 --served-model-name SSE_SWA_MOBA --gpu-memory-utilization 0.65 --block-size 128 --dtype bfloat16 --port 8711 --trust-remote-code --enforce-eager > _vllm_serve_sse_swa_moba.log 2>&1

TORCHDYNAMO_VERBOSE=1 nohup vllm serve /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling --max-model-len 65536 --served-model-name SSE_SWA_MOBA --gpu-memory-utilization 0.65 --tensor-parallel-size 4 --pipeline-parallel-size 2 --block-size 128 --dtype bfloat16 --port 8711 --trust-remote-code --enforce-eager > _vllm_serve_sse_swa_moba.log 2>&1

===
vllm 0.13.0

TORCHDYNAMO_VERBOSE=1 nohup vllm serve /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling --max-model-len 65536 --served-model-name SSE_SWA_MOBA --gpu-memory-utilization 0.85 --tensor-parallel-size 8 --block-size 128 --dtype bfloat16 --port 8711 --enable-chunked-prefill --trust-remote-code --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' > _vllm_serve_sse_swa_moba.log 2>&1

====
vllm 0.17.1
nohup vllm serve /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling --max-model-len 16384 --served-model-name SSE_SWA_MOBA --gpu-memory-utilization 0.85 --tensor-parallel-size 8 --block-size 128 --dtype bfloat16 --port 8711 --enable-prefix-caching --mamba-cache-mode align --mamba-block-size 8 --max_num_batched_tokens 4096 --trust-remote-code --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' > _vllm2_serve_sse_swa_moba.log 2>&1

====
QWEN: nohup vllm serve /mnt/jfzn/models/Qwen3-4B-Thinking-2507 --max-model-len 4194304 --no-enable-chunked-prefill --served-model-name Qwen --gpu-memory-utilization 0.9 --tensor-parallel-size 8 --block-size 128 --dtype bfloat16 --port 8711 --trust-remote-code --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' --no-enable-prefix-caching --rope-scaling '{"rope_type":"yarn","factor":16.0,"original_max_position_embeddings":262144}' > _vllm_serve_qwen.log 2>&1

====
pureSWA: TORCHDYNAMO_VERBOSE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_moba_gdn_u1to3_pureSwa_1.7b_dense_lr3en5_min0p1_bsz64_ep1_aux1en3_pt_data_800k/modeling2/ --max-model-len 65536 --served-model-name SSE_SWA_MOBA --gpu-memory-utilization 0.65 --block-size 128 --dtype bfloat16 --port 8711 --trust-remote-code --enforce-eager > _vllm_serve_sse_swa_moba_pureSWA.log 2>&1

/mnt/jfzn/pyq/ColossalAI-dev/checkpoints/spb2_5b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_offdistill70k/modeling

=== 
vllm 0.13.0 if 0.17.1 replace the input-len and output-len with random-input-len and random-output-len

nohup vllm bench throughput \
  --model /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 132000 \
  --gpu-memory-utilization 0.85 \
  --block-size 128 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --input-len 131072 \
  --output-len 128 \
  --num-prompts 1000 > bench2_sp2_throughput.log 2>&1 &

nohup vllm bench throughput \
  --model /mnt/jfzn/models/Qwen3-4B-Thinking-2507/ \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 70000 \
  --gpu-memory-utilization 0.85 \
  --block-size 128 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --input-len 65536 \
  --output-len 128 \
  --num-prompts 1000 > bench2_qwen_new_throughput.log 2>&1 &

nohup vllm serve /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling --max-model-len 4194304 --served-model-name SSE_SWA_MOBA --gpu-memory-utilization 0.8 --max-num-seqs 2048 --no-enable-chunked-prefill --tensor-parallel-size 8 --block-size 128 --dtype bfloat16 --port 8711 --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' > _vllm_serve_sse_swa_moba.log 2>&1

python benchmark_vllm_ttft_tpot.py \
  --base-url http://127.0.0.1:8711 \
  --model SSE_SWA_MOBA \
  --model-path /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling \
  --runs-per-len 5 \
  --warmup-runs 3 \
  --gen-tokens 100 \
  --force-min-tokens \
  --lengths "1m" \
  --ignore-eos

lm-eval:

http://github.com/EleutherAI/lm-evaluation-harness/blob/d800e04dcb1ce96791d8b2926cf0cc7703d58457/lm_eval/_cli/run.py

source /mnt/jfzn/miniconda3/bin/activate
conda activate zjh_lm_eval
cd /mnt/jfzn/zjh/V2_dev_bench
export HF_ENDPOINT=https://hf-mirror.com 
export http_proxy="http://cloudml:gP1dY0uI0o@10.119.176.202:3128"
export https_proxy="http://cloudml:gP1dY0uI0o@10.119.176.202:3128"

nohup lm_eval \
  --model vllm \
  --model_args '{
    "pretrained": "/mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling",
    "tensor_parallel_size": 8,
    "gpu_memory_utilization": 0.8,
    "max_model_len": 524288,
    "dtype": "bfloat16",
    "block_size": 128,
    "enable_chunked_prefill": false,
    "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}
  }' \
  --tasks bbh_cot_fewshot \
  --num_fewshot 3 \
  --batch_size 2 \
  --output_path lm_eval_results/bbh > /mnt/jfzn/zjh/V2_dev_bench/logs/bbh_cot3_sp2_vllm.log 2>&1 &

nohup lm_eval \
  --model vllm \
  --model_args '{
    "pretrained": "/mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling",
    "tensor_parallel_size": 8,
    "gpu_memory_utilization": 0.8,
    "max_model_len": 65536,
    "dtype": "bfloat16",
    "block_size": 128,
    "enable_chunked_prefill": false,
    "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}
  }' \
  --tasks gsm8k_cot \
  --num_fewshot 4 \
  --batch_size 32 \
  --log_samples \
  --output_path lm_eval_results/gsm8k_cot_nochat > /mnt/jfzn/zjh/V2_dev_bench/logs/gsm8k_cot_sp2_vllm.log 2>&1 &


"enforce_eager": true

HF_ALLOW_CODE_EVAL=1 nohup lm_eval \
  --model vllm \
  --model_args '{
    "pretrained": "/mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling",
    "tensor_parallel_size": 8,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 8192,
    "max_num_batched_tokens": 8192,
    "dtype": "bfloat16",
    "block_size": 128,
    "enable_chunked_prefill": false,
    "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}
  }' \
  --tasks humaneval_instruct \
  --num_fewshot 0 \
  --batch_size auto \
  --confirm_run_unsafe_code \
  --apply_chat_template \
  --log_samples \
  --output_path lm_eval_results/humaneval > /mnt/jfzn/zjh/V2_dev_bench/logs/humaneval_sp2_vllm.log 2>&1 &

nohup lm_eval \
  --model vllm \
  --model_args '{
    "pretrained": "/mnt/jfzn/oylt/checkpoints/sp2_deepscaler/global_step_10/",
    "tensor_parallel_size": 8,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 131072,
    "max_num_batched_tokens": 131072,
    "dtype": "bfloat16",
    "block_size": 128,
    "enable_chunked_prefill": false,
    "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}
  }' \
  --tasks aime25 \
  --num_fewshot 0 \
  --batch_size 15 \
  --apply_chat_template \
  --log_samples \
  --output_path lm_eval_results/aime25 > /mnt/jfzn/zjh/V2_dev_bench/logs/aime25_sp2_deepscaler_gs10_vllm.log 2>&1 &

nohup lm_eval \
  --model vllm \
  --model_args '{
    "pretrained": "/mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling",
    "tensor_parallel_size": 8,
    "gpu_memory_utilization": 0.50,
    "max_model_len": 32768,
    "max_num_batched_tokens": 32768,
    "dtype": "bfloat16",
    "block_size": 128,
    "enable_chunked_prefill": false,
    "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}
  }' \
  --tasks mmlu \
  --batch_size auto \
  --log_samples \
  --output_path lm_eval_results/mmlu > /mnt/jfzn/zjh/V2_dev_bench/logs/mmlu_sp2_vllm.log 2>&1 &

nohup lm_eval \
  --model vllm \
  --model_args '{
    "pretrained": "/mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling",
    "tensor_parallel_size": 8,
    "gpu_memory_utilization": 0.75,
    "max_model_len": 65536,
    "dtype": "bfloat16",
    "block_size": 128,
    "enable_chunked_prefill": true,
    "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}
  }' \
  --tasks longbench \
  --batch_size 48 \
  --log_samples \
  --output_path lm_eval_results/longbench > /mnt/jfzn/zjh/V2_dev_bench/logs/longbench_sp2_64k_vllm.log 2>&1 &

nohup lm_eval \
  --model vllm \
  --model_args '{
    "pretrained": "/mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling",
    "tensor_parallel_size": 8,
    "gpu_memory_utilization": 0.8,
    "max_model_len": 131072,
    "dtype": "bfloat16",
    "block_size": 128,
    "enable_chunked_prefill": true,
    "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}
  }' \
  --tasks ruler \
  --batch_size 16 \
  --metadata '{"max_seq_lengths":[65526,81920,131072]}' \
  --log_samples \
  --output_path lm_eval_results/ruler > /mnt/jfzn/zjh/V2_dev_bench/logs/ruler_sp2_vllm.log 2>&1 &