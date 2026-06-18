import json

import tilelang
from tilelang import tvm
from tvm import tir

from tilelang.profile import (
    EVENT_BEGIN,
    EVENT_END,
    EVENT_MARK,
    PROFILE_REGION_ATTR,
    PipelineDAG,
    TraceSession,
    decode_records,
    pack_meta,
    pair_spans,
    segment_words,
    write_chrome_trace,
    write_pipeline_svg,
)


def test_decode_pair_and_export(tmp_path):
    region_names = {1: "wait_signal", 2: "gemm", 3: "store"}
    records = [
        [1000, pack_meta(rank=0, kind=EVENT_BEGIN, region_id=1, block=7, warp=0), 10, 20],
        [1400, pack_meta(rank=0, kind=EVENT_END, region_id=1, block=7, warp=0), 10, 20],
        [1500, pack_meta(rank=0, kind=EVENT_BEGIN, region_id=2, block=7, warp=1), 0, 7],
        [2300, pack_meta(rank=0, kind=EVENT_END, region_id=2, block=7, warp=1), 0, 7],
        [2400, pack_meta(rank=0, kind=EVENT_MARK, region_id=3, block=7, warp=0), 0, 0],
    ]

    events = decode_records(records, region_names)
    spans = pair_spans(events)

    assert [span.name for span in spans] == ["wait_signal", "gemm", "store"]
    assert spans[0].duration_ns == 400
    assert spans[1].block == 7
    assert spans[1].warp == 1

    dag = PipelineDAG("ag_gemm")
    dag.region(1, "wait_signal")
    dag.region(2, "gemm")
    dag.edge("wait_signal", "gemm")

    chrome_path = write_chrome_trace(tmp_path / "trace.json", spans)
    svg_path = write_pipeline_svg(tmp_path / "pipeline.svg", spans, dag=dag)

    assert json.loads(chrome_path.read_text())["traceEvents"][0]["name"] == "wait_signal"
    assert "wait_signal" in svg_path.read_text()


def test_trace_session_segment_decode_cpu():
    region_names = {1: "load", 2: "gemm"}
    session = TraceSession(events_per_segment=4, segments_per_block=2, total_blocks=1, device="cpu", region_names=region_names)
    stride = segment_words(session.events_per_segment)

    session.buffer[0] = 2
    session.buffer[1:5] = session.buffer.new_tensor(
        [1000, pack_meta(rank=0, kind=EVENT_BEGIN, region_id=1, block=0, warp=0), 0, 0]
    )
    session.buffer[5:9] = session.buffer.new_tensor(
        [1200, pack_meta(rank=0, kind=EVENT_END, region_id=1, block=0, warp=0), 0, 0]
    )

    session.buffer[stride] = 2
    session.buffer[stride + 1 : stride + 5] = session.buffer.new_tensor(
        [1300, pack_meta(rank=0, kind=EVENT_BEGIN, region_id=2, block=0, warp=1), 3, 4]
    )
    session.buffer[stride + 5 : stride + 9] = session.buffer.new_tensor(
        [1700, pack_meta(rank=0, kind=EVENT_END, region_id=2, block=0, warp=1), 3, 4]
    )

    spans = session.spans(synchronize=False)

    assert [span.name for span in spans] == ["load", "gemm"]
    assert spans[0].duration_ns == 200
    assert spans[1].warp == 1
    assert spans[1].duration_ns == 400


def test_scoped_profile_region_lowers_to_record_pair():
    marker = tir.call_extern(
        "handle",
        "tl_profile_marker",
        tir.IntImm("int64", 0),
        tir.IntImm("int32", 8),
        tir.IntImm("int32", 2),
        tir.IntImm("int32", 1),
        tir.IntImm("int32", 0),
        tir.IntImm("int32", 7),
        tir.IntImm("int32", EVENT_BEGIN),
        tir.IntImm("int32", 11),
        tir.IntImm("int32", 12),
    )
    body = tir.AttrStmt(marker, PROFILE_REGION_ATTR, tir.IntImm("int32", 1), tir.Evaluate(tir.IntImm("int32", 0)))
    mod = tvm.IRModule.from_expr(tir.PrimFunc([], body))

    lowered = tilelang.transform.LowerProfileMarkers()(mod)["main"].body
    text = str(lowered)

    assert text.count("tl_profile_record") == 2
    assert "tl_profile_marker" not in text
    assert "tl.profile_region" not in text


def test_auto_profile_simt_copy_wraps_global_to_shared_loop():
    marker = tir.call_extern(
        "handle",
        "tl_profile_marker",
        tir.IntImm("int64", 0),
        tir.IntImm("int32", 8),
        tir.IntImm("int32", 2),
        tir.IntImm("int32", 1),
        tir.IntImm("int32", 0),
        tir.IntImm("int32", 7),
        tir.IntImm("int32", EVENT_BEGIN),
        tir.IntImm("int32", 0),
        tir.IntImm("int32", 0),
    )
    i = tir.Var("i", "int32")
    A = tir.decl_buffer((16,), "float32", name="A")
    S = tir.decl_buffer((16,), "float32", name="S", scope="shared")
    copy_loop = tir.For(
        i,
        tir.IntImm("int32", 0),
        tir.IntImm("int32", 16),
        tir.ForKind.SERIAL,
        tir.BufferStore(S, tir.BufferLoad(A, [i]), [i]),
    )
    mod = tvm.IRModule.from_expr(tir.PrimFunc([], tir.SeqStmt([tir.Evaluate(marker), copy_loop])))

    marked = tilelang.transform.AutoProfileSimtCopyMarkers()(mod)["main"].body
    lowered = tilelang.transform.LowerProfileMarkers()(tvm.IRModule.from_expr(tir.PrimFunc([], marked)))["main"].body
    text = str(lowered)

    assert text.count("tl_profile_record") == 3
    assert text.count(", 251,") == 2
