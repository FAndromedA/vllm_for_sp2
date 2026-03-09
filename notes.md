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
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "我想要去中国旅游，帮我规划一个7天的旅游计划"}
        ],
        "do_sample": true,
        "temperature": 0.7,
        "max_tokens": 2048,
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


TORCHDYNAMO_VERBOSE=1 nohup vllm serve /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling --max-model-len 65536 --served-model-name SSE_SWA_MOBA --gpu-memory-utilization 0.65 --tensor-parallel-size 4 --block-size 128 --dtype bfloat16 --port 8711 --trust-remote-code --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' > _vllm_serve_sse_swa_moba.log 2>&1


TORCHDYNAMO_VERBOSE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_moba_gdn_u1to3_pureSwa_1.7b_dense_lr3en5_min0p1_bsz64_ep1_aux1en3_pt_data_800k/modeling2/ --max-model-len 65536 --served-model-name SSE_SWA_MOBA --gpu-memory-utilization 0.65 --block-size 128 --dtype bfloat16 --port 8711 --trust-remote-code --enforce-eager > _vllm_serve_sse_swa_moba_pureSWA.log 2>&1



lm-eval:

http://github.com/EleutherAI/lm-evaluation-harness/blob/d800e04dcb1ce96791d8b2926cf0cc7703d58457/lm_eval/_cli/run.py

source /mnt/jfzn/miniconda3/bin/activate
conda activate oylt_sp2
cd /mnt/jfzn/zjh/V2_dev_bench
export HF_ENDPOINT=https://hf-mirror.com 
export http_proxy="http://cloudml:gP1dY0uI0o@10.119.176.202:3128"
export https_proxy="http://cloudml:gP1dY0uI0o@10.119.176.202:3128"

nohup lm_eval \
  --model vllm \
  --model_args '{
    "pretrained": "/mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling",
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.75,
    "max_model_len": 16384,
    "dtype": "bfloat16",
    "block_size": 128,
    "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}
  }' \
  --tasks bbh_cot_fewshot \
  --num_fewshot 3 \
  --batch_size auto \
  --output_path lm_eval_results/bbh > /mnt/jfzn/zjh/V2_dev_bench/logs/bbh_cot3_sp2_vllm.log 2>&1 &

nohup lm_eval \
  --model vllm \
  --model_args '{
    "pretrained": "/mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling",
    "tensor_parallel_size": 8,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 524288,
    "dtype": "bfloat16",
    "block_size": 128,
    "enable_chunked_prefill": false,
    "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}
  }' \
  --tasks bbh_cot_fewshot \
  --num_fewshot 3 \
  --batch_size auto \
  --log_samples \
  --apply_chat_template false \
  --output_path lm_eval_results/bbh_512K_nochat2 > /mnt/jfzn/zjh/V2_dev_bench/logs/bbh_cot3_512K_sp2_vllm2.log 2>&1 &


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