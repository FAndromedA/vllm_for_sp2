"""
Precision comparison: HF vs vLLM kernel implementations for SSE-GDN.

Key layout difference:
  - FLA (HF):  state shape [N, H, K, V]
  - vLLM chunk/fused_recurrent: state shape [N, H, V, K]
  - vLLM SSE custom kernel: state shape [slot, partition, H, V, K]
"""

import torch
import torch.nn.functional as F
import math
import sys
from einops import rearrange

torch.manual_seed(42)
DEVICE = "cuda"
DTYPE = torch.bfloat16


def compare(name, hf_out, vllm_out, atol=1e-3, rtol=1e-2):
    if hf_out is None or vllm_out is None:
        print(f"  [{name}] SKIP (one is None)")
        return False
    hf_f = hf_out.float().contiguous()
    vl_f = vllm_out.float().contiguous()
    if hf_f.shape != vl_f.shape:
        print(f"  [{name}] SHAPE MISMATCH: hf={hf_f.shape} vs vllm={vl_f.shape}")
        return False
    diff = (hf_f - vl_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (hf_f.abs() + 1e-8)).mean().item()
    cos_sim = F.cosine_similarity(hf_f.reshape(1, -1), vl_f.reshape(1, -1)).item()
    ok = torch.allclose(hf_f, vl_f, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{name}] {status}  max={max_diff:.6e}  mean={mean_diff:.6e}  rel={rel_diff:.6e}  cos={cos_sim:.8f}")
    return ok


def test_gating():
    """Test 1: GDN gating - PyTorch eager vs Triton fused."""
    print("\n===== Test 1: GDN Gating Computation =====")
    num_heads = 24
    seq_len = 128

    A_log = torch.randn(num_heads * 2, dtype=torch.float32, device=DEVICE).abs().log()
    dt_bias = torch.randn(num_heads * 2, dtype=torch.float32, device=DEVICE)
    a_proj = torch.randn(seq_len, num_heads * 2, dtype=DTYPE, device=DEVICE)
    b_proj = torch.randn(seq_len, num_heads * 2, dtype=DTYPE, device=DEVICE)

    b_hf = b_proj.sigmoid()
    g_hf = -A_log.float().exp() * F.softplus(a_proj.float() + dt_bias)

    from vllm.triton_utils import tl, triton

    @triton.jit
    def _fused_gdn_gating_kernel(g, beta_output, A_log, a, b, dt_bias, seq_len,
                                  NUM_HEADS: tl.constexpr, beta_val: tl.constexpr,
                                  threshold: tl.constexpr, BLK_HEADS: tl.constexpr):
        i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
        off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
        mask = head_off < NUM_HEADS
        blk_A = tl.load(A_log + head_off, mask=mask)
        blk_a = tl.load(a + off, mask=mask)
        blk_b = tl.load(b + off, mask=mask)
        blk_bias = tl.load(dt_bias + head_off, mask=mask)
        x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
        sp = tl.where(beta_val * x <= threshold, (1 / beta_val) * tl.log(1 + tl.exp(beta_val * x)), x)
        blk_g = -tl.exp(blk_A.to(tl.float32)) * sp
        tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
        blk_beta = tl.sigmoid(blk_b.to(tl.float32))
        tl.store(beta_output + off, blk_beta.to(beta_output.dtype.element_ty), mask=mask)

    batch, nh = a_proj.shape
    sl = 1
    grid = (batch, sl, triton.cdiv(nh, 8))
    g_vllm = torch.empty(1, batch, nh, dtype=torch.float32, device=DEVICE)
    beta_vllm = torch.empty(1, batch, nh, dtype=b_proj.dtype, device=DEVICE)
    _fused_gdn_gating_kernel[grid](g_vllm, beta_vllm, A_log, a_proj, b_proj, dt_bias, sl, nh, 1.0, 20.0, 8, num_warps=1)
    g_vllm = g_vllm.squeeze(0)
    beta_vllm = beta_vllm.squeeze(0)

    compare("g (decay)", g_hf, g_vllm, atol=1e-4)
    compare("beta (sigmoid)", b_hf, beta_vllm, atol=1e-4)


def test_sort_along_l():
    """Test 2: sort_along_l - HF vs vLLM (same algorithm, just checking equivalence)."""
    print("\n===== Test 2: sort_along_l =====")
    from sse_swa_moba_hf.sse_swa_hf import sort_along_l as hf_sort
    from sse_swa_moba_vllm.layers.sse_swa_h import sort_along_l as vllm_sort

    B, L, H, D = 1, 64, 6, 128
    N = 4
    K = 1

    q = torch.randn(B, L, H, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, L, H, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, L, H, D, dtype=DTYPE, device=DEVICE)
    gk = torch.randn(B, L, H, dtype=torch.float32, device=DEVICE)
    beta = torch.randn(B, L, H, dtype=DTYPE, device=DEVICE).sigmoid()
    e = torch.randn(B, L, N, dtype=DTYPE, device=DEVICE)
    cu_seqlens = torch.tensor([0, L], dtype=torch.int32, device=DEVICE)

    hf_r = hf_sort(q, k, v, gk, beta, e, cu_seqlens, K, True, True)
    vllm_r = vllm_sort(q, k, v, gk, beta, e, cu_seqlens, K, True, True)

    compare("q_sorted", hf_r[0], vllm_r[0])
    compare("k_sorted", hf_r[1], vllm_r[1])
    compare("v_sorted", hf_r[2], vllm_r[2])
    compare("gk_sorted", hf_r[3], vllm_r[3])
    compare("beta_sorted", hf_r[4], vllm_r[4])
    compare("offsets", hf_r[7].float(), vllm_r[7].float())
    compare("global_sorted", hf_r[9].float(), vllm_r[9].float())


def test_chunk_gdn():
    """Test 3: chunk_gated_delta_rule (FLA [N,H,K,V] vs vLLM [N,H,V,K])."""
    print("\n===== Test 3: chunk_gated_delta_rule (FLA vs vLLM) =====")
    print("  NOTE: FLA state=[N,H,K,V], vLLM state=[N,H,V,K]")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_chunk

    for seq_len in [64, 256, 1024]:
        print(f"  --- seq_len={seq_len} ---")
        B, T, H, K, V = 1, seq_len, 6, 128, 128
        q = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE)
        k = F.normalize(torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE), p=2, dim=-1)
        v = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE)
        g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=DEVICE))
        beta = torch.randn(B, T, H, dtype=DTYPE, device=DEVICE).sigmoid()
        cu = torch.tensor([0, T], dtype=torch.long, device=DEVICE)

        # FLA: h0 shape [N, H, K, V]
        h0_fla = torch.zeros(1, H, K, V, dtype=torch.float32, device=DEVICE)
        o_fla, s_fla = fla_chunk(q=q, k=k, v=v, g=g, beta=beta,
                                  initial_state=h0_fla, output_final_state=True,
                                  cu_seqlens=cu, use_qk_l2norm_in_kernel=True)

        # vLLM: h0 shape [N, H, V, K]
        h0_vllm = torch.zeros(1, H, V, K, dtype=torch.float32, device=DEVICE)
        o_vllm, s_vllm = vllm_chunk(q=q, k=k, v=v, g=g, beta=beta,
                                      initial_state=h0_vllm, output_final_state=True,
                                      cu_seqlens=cu, use_qk_l2norm_in_kernel=True)

        compare(f"chunk_o (T={T})", o_fla, o_vllm, atol=5e-3)
        # transpose vLLM state [N,H,V,K] -> [N,H,K,V] for comparison
        s_vllm_t = s_vllm.transpose(-1, -2)
        compare(f"chunk_state (T={T})", s_fla, s_vllm_t, atol=5e-3)


def test_fused_recurrent_gdn():
    """Test 4: fused_recurrent (FLA [N,H,K,V] vs custom Triton [slot,part,H,V,K]).
    NOTE: vLLM kernel is decode-only (T=1 per sequence). For multi-token, use chunk kernel.
    We test with multiple independent sequences via cu_seqlens to verify multi-seq handling."""
    print("\n===== Test 4: fused_recurrent_gated_delta_rule (FLA vs custom Triton) =====")
    print("  NOTE: FLA state=[N,H,K,V], vLLM state=[slot,partition,H,V,K]")
    print("  NOTE: vLLM kernel is decode-only (T=1 per seq), testing single-token only")
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fla_recur
    from sse_swa_moba_vllm.layers.ops.fused_recurrent import sse_fused_recurrent_gated_delta_rule as vllm_recur

    H, K, V = 6, 128, 128
    N_PART = 5

    # Test with different numbers of concurrent sequences (each T=1)
    # and with both zero and non-zero initial states
    for num_seqs in [1, 4]:
        for use_nonzero_h0 in [False, True]:
            tag = f"N={num_seqs}, h0={'nonzero' if use_nonzero_h0 else 'zero'}"
            print(f"  --- {tag} ---")
            B, T = 1, num_seqs

            q = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE)
            k = F.normalize(torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE), p=2, dim=-1)
            v = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE)
            g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=DEVICE))
            beta = torch.randn(B, T, H, dtype=DTYPE, device=DEVICE).sigmoid()
            cu = torch.arange(0, num_seqs + 1, dtype=torch.long, device=DEVICE)

            if use_nonzero_h0:
                h0_fla = torch.randn(num_seqs, H, K, V, dtype=torch.float32, device=DEVICE) * 0.1
            else:
                h0_fla = torch.zeros(num_seqs, H, K, V, dtype=torch.float32, device=DEVICE)

            o_fla, s_fla = fla_recur(q=q, k=k, v=v, g=g, beta=beta,
                                      initial_state=h0_fla.clone(), output_final_state=True,
                                      cu_seqlens=cu, use_qk_l2norm_in_kernel=True)

            # vLLM: h0 [num_slots, N_PART, H, V, K] — transpose from FLA's [N, H, K, V]
            h0_vllm = torch.zeros(num_seqs, N_PART, H, V, K, dtype=torch.float32, device=DEVICE)
            h0_vllm[:, 0] = h0_fla.transpose(-1, -2)  # [N, H, K, V] -> [N, H, V, K]
            ssm_idx = torch.arange(num_seqs, dtype=torch.long, device=DEVICE)
            exp_idx = torch.zeros(num_seqs, dtype=torch.long, device=DEVICE)
            o_vllm, s_vllm = vllm_recur(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=h0_vllm,
                inplace_final_state=True,
                cu_seqlens=cu,
                ssm_state_indices=ssm_idx,
                use_qk_l2norm_in_kernel=True,
                ssm_state_expert_indices=exp_idx,
                num_partitions=N_PART,
            )

            compare(f"fused_o ({tag})", o_fla, o_vllm, atol=5e-3)
            s_vllm_cmp = s_vllm[:, 0].transpose(-1, -2)  # [N, H, K, V]
            compare(f"fused_state ({tag})", s_fla, s_vllm_cmp, atol=5e-3)


def test_fused_recurrent_accumulation():
    """Test 4b: Multi-step decode to detect state drift over time."""
    print("\n===== Test 4b: Multi-step decode state accumulation =====")
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fla_recur
    from sse_swa_moba_vllm.layers.ops.fused_recurrent import sse_fused_recurrent_gated_delta_rule as vllm_recur

    B, H, K, V = 1, 6, 128, 128
    N_PART = 5
    NUM_STEPS = 200

    h_fla = torch.zeros(1, H, K, V, dtype=torch.float32, device=DEVICE)
    h_vllm = torch.zeros(1, N_PART, H, V, K, dtype=torch.float32, device=DEVICE)

    for step in range(NUM_STEPS):
        torch.manual_seed(step + 1000)
        q = torch.randn(B, 1, H, K, dtype=DTYPE, device=DEVICE)
        k = F.normalize(torch.randn(B, 1, H, K, dtype=DTYPE, device=DEVICE), p=2, dim=-1)
        v = torch.randn(B, 1, H, V, dtype=DTYPE, device=DEVICE)
        g = F.logsigmoid(torch.randn(B, 1, H, dtype=torch.float32, device=DEVICE))
        beta = torch.randn(B, 1, H, dtype=DTYPE, device=DEVICE).sigmoid()
        cu = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)

        o_fla, h_fla = fla_recur(q=q, k=k, v=v, g=g, beta=beta,
                                  initial_state=h_fla, output_final_state=True,
                                  cu_seqlens=cu, use_qk_l2norm_in_kernel=True)

        ssm_idx = torch.zeros(1, dtype=torch.long, device=DEVICE)
        exp_idx = torch.zeros(1, dtype=torch.long, device=DEVICE)
        o_vllm, _ = vllm_recur(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=h_vllm,
            inplace_final_state=True,
            cu_seqlens=cu,
            ssm_state_indices=ssm_idx,
            use_qk_l2norm_in_kernel=True,
            ssm_state_expert_indices=exp_idx,
            num_partitions=N_PART,
        )

        if step in [0, 9, 49, 99, 199]:
            h_vllm_cmp = h_vllm[0, 0].transpose(-1, -2).unsqueeze(0)  # [1, H, K, V]
            compare(f"step_{step}_output", o_fla, o_vllm, atol=1e-2)
            compare(f"step_{step}_state", h_fla, h_vllm_cmp, atol=1e-2)


def test_chunk_then_recurrent():
    """Test 5: Prefill with chunk, then decode with fused_recurrent (the real inference path).
    Tests whether state handoff from chunk->recurrent is correct across the V,K layout."""
    print("\n===== Test 5: Chunk (prefill) -> Recurrent (decode) pipeline =====")
    print("  This tests the actual inference path: chunk for prefill, fused_recurrent for decode")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fla_recur
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_chunk
    from sse_swa_moba_vllm.layers.ops.fused_recurrent import sse_fused_recurrent_gated_delta_rule as vllm_recur

    B, H, K, V = 1, 6, 128, 128
    N_PART = 5
    PREFILL_LEN = 256
    DECODE_STEPS = 50

    # --- Prefill phase ---
    torch.manual_seed(42)
    q_pre = torch.randn(B, PREFILL_LEN, H, K, dtype=DTYPE, device=DEVICE)
    k_pre = F.normalize(torch.randn(B, PREFILL_LEN, H, K, dtype=DTYPE, device=DEVICE), p=2, dim=-1)
    v_pre = torch.randn(B, PREFILL_LEN, H, V, dtype=DTYPE, device=DEVICE)
    g_pre = F.logsigmoid(torch.randn(B, PREFILL_LEN, H, dtype=torch.float32, device=DEVICE))
    beta_pre = torch.randn(B, PREFILL_LEN, H, dtype=DTYPE, device=DEVICE).sigmoid()
    cu_pre = torch.tensor([0, PREFILL_LEN], dtype=torch.long, device=DEVICE)

    # FLA prefill: state [N, H, K, V]
    h0_fla = torch.zeros(1, H, K, V, dtype=torch.float32, device=DEVICE)
    o_fla_pre, h_fla = fla_chunk(q=q_pre, k=k_pre, v=v_pre, g=g_pre, beta=beta_pre,
                                  initial_state=h0_fla, output_final_state=True,
                                  cu_seqlens=cu_pre, use_qk_l2norm_in_kernel=True)

    # vLLM prefill: state [N, H, V, K]
    h0_vllm_chunk = torch.zeros(1, H, V, K, dtype=torch.float32, device=DEVICE)
    o_vllm_pre, h_vllm_chunk = vllm_chunk(q=q_pre, k=k_pre, v=v_pre, g=g_pre, beta=beta_pre,
                                            initial_state=h0_vllm_chunk, output_final_state=True,
                                            cu_seqlens=cu_pre, use_qk_l2norm_in_kernel=True)

    compare("prefill_o", o_fla_pre, o_vllm_pre, atol=5e-3)
    compare("prefill_state", h_fla, h_vllm_chunk.transpose(-1, -2), atol=5e-3)

    # --- Prepare vLLM state for decode (copy chunk state into SSE cache layout) ---
    h_vllm_decode = torch.zeros(1, N_PART, H, V, K, dtype=torch.float32, device=DEVICE)
    # partition 0 = base state; copy chunk output state there
    h_vllm_decode[0, 0] = h_vllm_chunk[0]  # [H, V, K]

    # --- Decode phase ---
    for step in range(DECODE_STEPS):
        torch.manual_seed(step + 5000)
        q = torch.randn(B, 1, H, K, dtype=DTYPE, device=DEVICE)
        k = F.normalize(torch.randn(B, 1, H, K, dtype=DTYPE, device=DEVICE), p=2, dim=-1)
        v = torch.randn(B, 1, H, V, dtype=DTYPE, device=DEVICE)
        g = F.logsigmoid(torch.randn(B, 1, H, dtype=torch.float32, device=DEVICE))
        beta = torch.randn(B, 1, H, dtype=DTYPE, device=DEVICE).sigmoid()
        cu = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)

        o_fla_dec, h_fla = fla_recur(q=q, k=k, v=v, g=g, beta=beta,
                                      initial_state=h_fla, output_final_state=True,
                                      cu_seqlens=cu, use_qk_l2norm_in_kernel=True)

        ssm_idx = torch.zeros(1, dtype=torch.long, device=DEVICE)
        exp_idx = torch.zeros(1, dtype=torch.long, device=DEVICE)
        o_vllm_dec, _ = vllm_recur(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=h_vllm_decode,
            inplace_final_state=True,
            cu_seqlens=cu,
            ssm_state_indices=ssm_idx,
            use_qk_l2norm_in_kernel=True,
            ssm_state_expert_indices=exp_idx,
            num_partitions=N_PART,
        )

        if step in [0, 9, 24, 49]:
            h_vllm_cmp = h_vllm_decode[0, 0].transpose(-1, -2).unsqueeze(0)
            compare(f"decode_step_{step}_o", o_fla_dec, o_vllm_dec, atol=1e-2)
            compare(f"decode_step_{step}_state", h_fla, h_vllm_cmp, atol=1e-2)


if __name__ == "__main__":
    import os
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    tests = sys.argv[1:] if len(sys.argv) > 1 else ["1", "3", "4", "4b", "5"]
    if "1" in tests:
        test_gating()
    if "2" in tests:
        test_sort_along_l()
    if "3" in tests:
        test_chunk_gdn()
    if "4" in tests:
        test_fused_recurrent_gdn()
    if "4b" in tests:
        test_fused_recurrent_accumulation()
    if "5" in tests:
        test_chunk_then_recurrent()

    print("\n===== Done =====")
