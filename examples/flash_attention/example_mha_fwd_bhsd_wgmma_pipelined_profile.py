import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import argparse
from pathlib import Path
from collections import Counter

import tilelang.profile as tl_profile

tilelang.disable_cache()


REG_LOAD_Q = 1
REG_INIT = 2
REG_LOAD_K = 3
REG_QK_MMA = 4
REG_SOFTMAX = 5
REG_SCALE_O = 6
REG_LOAD_V = 7
REG_PV_MMA = 8
REG_STORE = 9
REG_WS_PRODUCER = 250

REGION_NAMES = {
    REG_LOAD_Q: "prologue_load_q",
    REG_INIT: "prologue_init",
    REG_LOAD_K: "producer_load_k",
    REG_QK_MMA: "consumer_qk_mma",
    REG_SOFTMAX: "consumer_softmax",
    REG_SCALE_O: "consumer_rescale_o",
    REG_LOAD_V: "producer_load_v",
    REG_PV_MMA: "consumer_pv_mma",
    REG_STORE: "epilogue_store",
    REG_WS_PRODUCER: "ws_producer_copy",
}

DAG = tl_profile.dag("flash_attention_wgmma_pipeline")
for _region_id, _name in REGION_NAMES.items():
    DAG.region(_region_id, _name)
DAG.edge("prologue_load_q", "consumer_qk_mma")
DAG.edge("prologue_init", "consumer_softmax")
DAG.edge("producer_load_k", "consumer_qk_mma")
DAG.edge("consumer_qk_mma", "consumer_softmax")
DAG.edge("consumer_softmax", "consumer_rescale_o")
DAG.edge("consumer_softmax", "consumer_pv_mma")
DAG.edge("producer_load_v", "consumer_pv_mma")
DAG.edge("consumer_pv_mma", "epilogue_store")


@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def profiled_flashattn(batch, heads, seq_q, seq_kv, dim, is_causal, events_per_segment, segments_per_block, record_blocks, block_M=128, block_N=128, num_stages=2, threads=256):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32
    trace_words = tl_profile.segment_buffer_words(record_blocks, segments_per_block, events_per_segment)

    past_len = seq_kv - seq_q
    assert past_len >= 0, "seq_kv must be greater than or equal to seq_q"

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
        trace_buffer: T.Tensor((trace_words,), "int64"),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            tl_profile.import_source()
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            with tl_profile.scope(trace_buffer, events_per_segment, segments_per_block, record_blocks, 0, REG_LOAD_Q, bx, by):
                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)

            with tl_profile.scope(trace_buffer, events_per_segment, segments_per_block, record_blocks, 0, REG_INIT, bx, by):
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M + past_len, block_N))
                if is_causal
                else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(
                loop_range,
                num_stages=num_stages,
                order=[-1, 0, 3, 1, -1, 2],
                stage=[-1, 0, 0, 1, -1, 1],
                group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11], [12], [13], [14]],
            ):
                with tl_profile.scope(trace_buffer, events_per_segment, segments_per_block, record_blocks, 0, REG_LOAD_K, k, bx):
                    T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)

                with tl_profile.scope(trace_buffer, events_per_segment, segments_per_block, record_blocks, 0, REG_QK_MMA, k, bx):
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            q_idx = bx * block_M + i + past_len
                            k_idx = k * block_N + j
                            acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_kv, -T.infinity(acc_s.dtype), 0)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                with tl_profile.scope(trace_buffer, events_per_segment, segments_per_block, record_blocks, 0, REG_SOFTMAX, k, bx):
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                with tl_profile.scope(trace_buffer, events_per_segment, segments_per_block, record_blocks, 0, REG_SCALE_O, k, bx):
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]

                with tl_profile.scope(trace_buffer, events_per_segment, segments_per_block, record_blocks, 0, REG_LOAD_V, k, bx):
                    T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)

                with tl_profile.scope(trace_buffer, events_per_segment, segments_per_block, record_blocks, 0, REG_PV_MMA, k, bx):
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            with tl_profile.scope(trace_buffer, events_per_segment, segments_per_block, record_blocks, 0, REG_STORE, bx, by):
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return main


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype, device=scores.device))
    if is_causal:
        seq_q = Q.size(2)
        seq_kv = K.size(2)
        mask = torch.tril(torch.ones(seq_q, seq_kv, device=scores.device), seq_kv - seq_q)
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--seq-q", type=int, default=128)
    parser.add_argument("--seq-kv", type=int, default=512)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--is-causal", action="store_true")
    parser.add_argument("--events-per-segment", type=int, default=1024)
    parser.add_argument("--out-dir", type=str, default="/tmp/tilescale_flashattn_wgmma_pipeline_profile")
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--producer-threads", type=int, default=128)
    args = parser.parse_args()

    torch.manual_seed(0)
    Q = torch.randn(args.batch, args.heads, args.seq_q, args.dim, dtype=torch.float16, device="cuda")
    K = torch.randn(args.batch, args.heads, args.seq_kv, args.dim, dtype=torch.float16, device="cuda")
    V = torch.randn(args.batch, args.heads, args.seq_kv, args.dim, dtype=torch.float16, device="cuda")

    segments_per_block = (args.threads + args.producer_threads + 31) // 32
    record_blocks = ((args.seq_q + args.block_m - 1) // args.block_m) * args.heads * args.batch
    trace = tl_profile.TraceSession(
        events_per_segment=args.events_per_segment,
        segments_per_block=segments_per_block,
        total_blocks=record_blocks,
        device="cuda",
        region_names=REGION_NAMES,
    )
    (trace_buffer,) = trace.tensors()

    kernel = profiled_flashattn(
        args.batch,
        args.heads,
        args.seq_q,
        args.seq_kv,
        args.dim,
        args.is_causal,
        args.events_per_segment,
        segments_per_block,
        record_blocks,
        block_M=args.block_m,
        block_N=args.block_n,
        num_stages=args.num_stages,
        threads=args.threads,
    )
    O = kernel(Q, K, V, trace_buffer)
    torch.cuda.synchronize()

    ref = ref_program(Q, K, V, args.is_causal)
    max_err = (O - ref).abs().max().item()
    print(f"max_err={max_err:.6f}")
    print(f"allclose={torch.allclose(O, ref, rtol=0.01, atol=0.01)}")

    out_dir = Path(args.out_dir)
    chrome = trace.export_chrome_trace(out_dir / "flashattn_wgmma_trace.json", synchronize=False)
    svg = trace.write_pipeline_svg(
        out_dir / "flashattn_wgmma_pipeline.svg",
        dag=DAG,
        synchronize=False,
        title="FlashAttention WGMMA pipeline profile",
    )
    dag_json = DAG.write_json(out_dir / "flashattn_wgmma_dag.json")
    spans = trace.spans(synchronize=False)
    events = trace.events(synchronize=False)
    print(f"spans={len(spans)}")
    print(f"chrome={chrome}")
    print(f"svg={svg}")
    print(f"dag={dag_json}")

    event_summary = Counter((event.region, event.kind_name) for event in events)
    for (region, kind), count in sorted(event_summary.items()):
        print(f"event {region}/{kind}: count={count}")

    summary = {}
    for span in spans:
        summary.setdefault(span.name, []).append(span.duration_ns / 1000.0)
    for name, values in summary.items():
        print(f"{name}: count={len(values)} avg_us={sum(values) / len(values):.3f} min_us={min(values):.3f} max_us={max(values):.3f}")


if __name__ == "__main__":
    main()
