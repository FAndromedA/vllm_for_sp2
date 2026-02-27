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

TORCHDYNAMO_VERBOSE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k_aux1en4/modeling --tensor-parallel-size 4 --max-model-len 65536 --served-model-name SSE_SWA_MOBA --gpu-memory-utilization 0.65 --block-size 128 --dtype bfloat16 --port 8711 --trust-remote-code --enforce-eager --attention-backend CUSTOM > _vllm_serve_sse_swa_moba.log 2>&1

curl http://localhost:8711/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "SSE_SWA_MOBA",
        "messages": [
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

TORCHDYNAMO_VERBOSE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k_aux1en4/modeling --max-model-len 65536 --served-model-name SSE_SWA_MOBA --gpu-memory-utilization 0.65 --block-size 128 --dtype bfloat16 --port 8711 --trust-remote-code --enforce-eager > _vllm_serve_sse_swa_moba.log 2>&1

TORCHDYNAMO_VERBOSE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_moba_gdn_u1to3_pureSwa_1.7b_dense_lr3en5_min0p1_bsz64_ep1_aux1en3_pt_data_800k/modeling2/ --max-model-len 65536 --served-model-name SSE_SWA_MOBA --gpu-memory-utilization 0.65 --block-size 128 --dtype bfloat16 --port 8711 --trust-remote-code --enforce-eager > _vllm_serve_sse_swa_moba_pureSWA.log 2>&1

TORCHDYNAMO_VERBOSE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve /mnt/jfzn/models/Qwen3-4B-Instruct-2507 --max-model-len 32768 --served-model-name SSE_SWA_MOBA --gpu-memory-utilization 0.65 --block-size 128 --dtype bfloat16 --port 8711 --trust-remote-code --enforce-eager > _vllm_serve_qwen.log 2>&1