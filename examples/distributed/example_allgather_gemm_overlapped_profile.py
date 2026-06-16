import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing

import tilelang
import tilelang.language as T
import tilelang.profile as tl_profile
from tilelang.carver.arch import driver
from tilelang.distributed import init_dist, perf_fn

import example_allgather_gemm_overlapped as base

os.environ["NCCL_DEBUG"] = "WARN"

REG_WAIT_SIGNAL = 1
REG_LOAD_AB = 2
REG_GEMM = 3
REG_STORE = 4

PROFILE_REGION_NAMES = {
    REG_WAIT_SIGNAL: "wait_signal",
    REG_LOAD_AB: "load_ab",
    REG_GEMM: "gemm",
    REG_STORE: "store",
}

PROFILE_DAG = tl_profile.dag("allgather_gemm_overlapped")
for _region_id, _name in PROFILE_REGION_NAMES.items():
    PROFILE_DAG.region(_region_id, _name)
PROFILE_DAG.edge("wait_signal", "load_ab").edge("load_ab", "gemm").edge("gemm", "store")


@tilelang.jit
def profiled_gemm_kernel(
    M,
    N,
    K,
    local_rank,
    num_local_rank,
    block_M,
    block_N,
    block_K,
    threads,
    events_per_segment,
    segments_per_block,
    record_blocks,
    persistent=False,
    dtype="float16",
    accum_dtype="float",
):
    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N // num_local_rank, block_N)
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)
    M_per_rank = T.ceildiv(M, num_local_rank)
    GROUP_SIZE_M = 8
    trace_words = tl_profile.segment_buffer_words(record_blocks, segments_per_block, events_per_segment)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N // num_local_rank), dtype),
        signal_buffer: T.Tensor((num_local_rank), "uint32"),
        C: T.Tensor((M, N // num_local_rank), dtype),
        trace_buffer: T.Tensor((trace_words,), "int64"),
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N // num_local_rank, block_N), threads=threads) as (bid):
            tl_profile.import_source()
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            num_pid_m = T.ceildiv(M, block_M)
            num_pid_n = T.ceildiv(N // num_local_rank, block_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = bid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = T.min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m_ = first_pid_m + ((bid % num_pid_in_group) % group_size_m)
            pid_n_ = (bid % num_pid_in_group) // group_size_m

            m_offset = M_per_rank * local_rank
            pid_m_offset = T.ceildiv(m_offset, block_M)
            pid_m = (pid_m_ + pid_m_offset) % num_pid_m
            pid_n = pid_n_

            tid = T.get_thread_binding(0)
            T.clear(C_local)
            if tid == 0:
                tl_profile.begin(
                    trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_WAIT_SIGNAL, pid_m, pid_n
                )
                T.wait_eq(signal_buffer[pid_m * block_M // M_per_rank], 1)
                tl_profile.end(
                    trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_WAIT_SIGNAL, pid_m, pid_n
                )
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                tl_profile.begin(trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_LOAD_AB, k, bid)
                T.copy(A[pid_m * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, pid_n * block_N], B_shared)
                tl_profile.end(trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_LOAD_AB, k, bid)
                tl_profile.begin(trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_GEMM, k, bid)
                T.gemm(A_shared, B_shared, C_local)
                tl_profile.end(trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_GEMM, k, bid)
            tl_profile.begin(trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_STORE, pid_m, pid_n)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[pid_m * block_M, pid_n * block_N])
            tl_profile.end(trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_STORE, pid_m, pid_n)

    @T.prim_func
    def main_persistent(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N // num_local_rank), dtype),
        signal_buffer: T.Tensor((num_local_rank), "uint32"),
        C: T.Tensor((M, N // num_local_rank), dtype),
        trace_buffer: T.Tensor((trace_words,), "int64"),
    ):
        with T.Kernel(sm_num, threads=threads) as (bid):
            tl_profile.import_source()
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for w in T.serial(waves):
                tile_id = bid + w * sm_num
                num_pid_m = T.ceildiv(M, block_M)
                num_pid_n = T.ceildiv(N // num_local_rank, block_N)
                num_pid_in_group = GROUP_SIZE_M * num_pid_n
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = T.min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m_ = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
                pid_n_ = (tile_id % num_pid_in_group) // group_size_m

                m_offset = M_per_rank * local_rank
                pid_m_offset = T.ceildiv(m_offset, block_M)
                pid_m = (pid_m_ + pid_m_offset) % num_pid_m
                pid_n = pid_n_

                if pid_n_ * block_N < (N // num_local_rank) and pid_m_ * block_M < M:
                    tid = T.get_thread_binding(0)
                    T.clear(C_local)
                    if tid == 0:
                        tl_profile.begin(
                            trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_WAIT_SIGNAL, w, tile_id
                        )
                        T.wait_eq(signal_buffer[pid_m * block_M // M_per_rank], 1)
                        tl_profile.end(
                            trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_WAIT_SIGNAL, w, tile_id
                        )
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                        tl_profile.begin(
                            trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_LOAD_AB, k, tile_id
                        )
                        T.copy(A[pid_m * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, pid_n * block_N], B_shared)
                        tl_profile.end(
                            trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_LOAD_AB, k, tile_id
                        )
                        tl_profile.begin(
                            trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_GEMM, k, tile_id
                        )
                        T.gemm(A_shared, B_shared, C_local)
                        tl_profile.end(
                            trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_GEMM, k, tile_id
                        )
                    tl_profile.begin(trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_STORE, w, tile_id)
                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[pid_m * block_M, pid_n * block_N])
                    tl_profile.end(trace_buffer, events_per_segment, segments_per_block, record_blocks, local_rank, REG_STORE, w, tile_id)

    return main if not persistent else main_persistent


def profiled_ag_gemm_op(
    A,
    B,
    C,
    ag_buffer,
    signal_buffer,
    M_per_rank,
    signal_target,
    local_rank,
    local_world_size,
    set_signal_kernel,
    gemm_kernel,
    trace_buffer,
    trace_session,
    gemm_stream,
    ag_stream,
):
    trace_session.reset()
    with torch.cuda.stream(gemm_stream):
        set_signal_kernel(signal_buffer[local_rank])

    ag_stream.wait_stream(gemm_stream)

    base.cp_engine_producer_all_gather_full_mesh_pull(
        ag_buffer, signal_buffer, M_per_rank, signal_target, local_rank, local_world_size, ag_stream
    )

    with torch.cuda.stream(gemm_stream):
        gemm_kernel(ag_buffer[local_rank], B, signal_buffer[local_rank], C, trace_buffer)

    gemm_stream.wait_stream(ag_stream)
    current_stream = torch.cuda.current_stream()
    current_stream.wait_stream(gemm_stream)
    return C


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    dtype = torch.float16
    M = args.M
    N = args.N
    K = args.K
    persistent = args.persistent
    M_per_rank = M // num_local_ranks
    N_per_rank = N // num_local_ranks

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    threads = 256
    segments_per_block = (threads + 31) // 32
    record_blocks = args.trace_blocks

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert rank == local_rank and num_ranks == num_local_ranks, "only support single node for now"
    allocator = tilelang.get_allocator(
        size=2**30, device="cuda", is_distributed=True, local_rank=local_rank, num_local_ranks=num_local_ranks, group=group
    )
    gemm_func = profiled_gemm_kernel(
        M,
        N,
        K,
        local_rank,
        num_local_ranks,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        threads,
        args.trace_events_per_segment,
        segments_per_block,
        record_blocks,
        persistent,
    )
    set_signal_func = base.set_signal_kernel(local_rank=local_rank, num_local_ranks=num_local_ranks, threads=32)
    gemm_func.initialize(allocator=allocator)
    set_signal_func.initialize(allocator=allocator)

    if args.print_kernel_source and local_rank == 0:
        print(gemm_func.get_kernel_source())

    B = tilelang.tensor((K, N_per_rank), dtype, allocator=allocator).normal_()
    C = tilelang.tensor((M, N_per_rank), dtype, allocator=allocator)
    ag_buffer = tilelang.tensor((M, K), dtype, allocator=allocator, return_peers=True)
    A = ag_buffer[local_rank][M_per_rank * local_rank : M_per_rank * (local_rank + 1), :].normal_()
    signal_buffer = tilelang.tensor((num_local_ranks,), torch.uint32, allocator=allocator, return_peers=True)
    trace_session = tl_profile.TraceSession(
        events_per_segment=args.trace_events_per_segment,
        segments_per_block=segments_per_block,
        total_blocks=record_blocks,
        device="cuda",
        region_names=PROFILE_REGION_NAMES,
    )
    (trace_buffer,) = trace_session.tensors()

    gemm_stream = torch.cuda.current_stream()
    ag_stream = torch.cuda.Stream(priority=-1)
    signal_target = 1

    dist.barrier()

    tilelang_C = profiled_ag_gemm_op(
        A,
        B,
        C,
        ag_buffer,
        signal_buffer,
        M_per_rank,
        signal_target,
        local_rank,
        num_local_ranks,
        set_signal_func,
        gemm_func,
        trace_buffer,
        trace_session,
        gemm_stream,
        ag_stream,
    )

    torch_ag_buffer = torch.empty([M, K], dtype=dtype, device="cuda")
    torch_C = base.torch_ag_gemm(group, A, B, torch_ag_buffer)

    if torch.allclose(torch_C, tilelang_C, atol=1e-6, rtol=1e-6):
        print(f"rank {local_rank} check passed.")
    else:
        print(f"rank {local_rank} check failed.")
        print(f"torch_C: {torch_C}, tilelang_C: {tilelang_C}")

    out_dir = Path(args.profile_out_dir)
    chrome_path = out_dir / f"ag_gemm_rank{local_rank}.json"
    svg_path = out_dir / f"ag_gemm_rank{local_rank}.svg"
    dag_path = out_dir / "ag_gemm_dag.json"
    trace_session.export_chrome_trace(chrome_path, rank=local_rank)
    trace_session.write_pipeline_svg(svg_path, dag=PROFILE_DAG, rank=local_rank, title=f"AG GEMM rank {local_rank}")
    if local_rank == 0:
        PROFILE_DAG.write_json(dag_path)
    print(f"rank {local_rank} wrote profile trace: {chrome_path}")
    print(f"rank {local_rank} wrote pipeline svg: {svg_path}")

    tl_t = perf_fn(
        lambda: profiled_ag_gemm_op(
            A,
            B,
            C,
            ag_buffer,
            signal_buffer,
            M_per_rank,
            signal_target,
            local_rank,
            num_local_ranks,
            set_signal_func,
            gemm_func,
            trace_buffer,
            trace_session,
            gemm_stream,
            ag_stream,
        ),
        warmup=5,
        rep=10,
    )

    print(f"rank {local_rank} profiled tilelang ag_gemm time: {tl_t:.2f} ms, TFLOPS: {2 * M * N * K / 1e9 / tl_t / num_local_ranks:.2f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2, help="Number of processes to spawn")
    parser.add_argument("--M", type=int, default=8192, help="M dimension")
    parser.add_argument("--N", type=int, default=28672, help="N dimension")
    parser.add_argument("--K", type=int, default=8192, help="K dimension")
    parser.add_argument("--persistent", action="store_true", help="Use persistent kernel")
    parser.add_argument("--trace-events-per-segment", type=int, default=256, help="Trace event slots per block/warp segment")
    parser.add_argument("--trace-blocks", type=int, default=64, help="Number of linear blocks to record per rank")
    parser.add_argument("--profile-out-dir", type=str, default="/tmp/tilescale_profile", help="Directory for JSON/SVG outputs")
    parser.add_argument("--print-kernel-source", action="store_true", help="Print generated CUDA for rank 0")
    args = parser.parse_args()

    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes)
