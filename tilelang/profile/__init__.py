"""Kernel-internal tracing helpers for TileLang programs.

This first implementation uses a per-block/per-warp segmented trace buffer. It
keeps the Python DSL surface IKET-like while avoiding the single global cursor
hotspot that makes fine-grained in-kernel tracing especially noisy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Iterable

import tilelang.language as T

try:
    import torch
except ImportError:  # pragma: no cover - TileLang normally depends on torch.
    torch = None


WORDS_PER_EVENT = 4
SEGMENT_HEADER_WORDS = 1
EVENT_BEGIN = 0
EVENT_END = 1
EVENT_MARK = 2

_KIND_NAMES = {
    EVENT_BEGIN: "begin",
    EVENT_END: "end",
    EVENT_MARK: "mark",
}

_CUDA_SOURCE = r"""
#ifndef TILELANG_PROFILE_RECORD_DEFINED
#define TILELANG_PROFILE_RECORD_DEFINED

#include <stdint.h>

extern "C" __device__ __forceinline__ unsigned long long tl_profile_now() {
  unsigned long long timestamp;
#if defined(__CUDA_ARCH__)
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp));
#else
  timestamp = 0ULL;
#endif
  return timestamp;
}

extern "C" __device__ __forceinline__ unsigned long long tl_profile_linear_block() {
  return ((unsigned long long)blockIdx.x) +
         ((unsigned long long)gridDim.x) *
             (((unsigned long long)blockIdx.y) +
              ((unsigned long long)gridDim.y) * ((unsigned long long)blockIdx.z));
}

extern "C" __device__ __forceinline__ unsigned long long tl_profile_pack_meta(
    int rank,
    int region_id,
    int kind,
    unsigned long long block,
    unsigned long long warp) {
  return (((unsigned long long)rank & 0xFFULL) << 56) |
         (((unsigned long long)kind & 0x3ULL) << 54) |
         (((unsigned long long)region_id & 0x3FFFULL) << 40) |
         ((block & 0xFFFFFULL) << 16) |
         ((warp & 0xFFULL) << 8);
}

extern "C" __device__ __forceinline__ void tl_profile_record(
    int64_t* trace_buffer,
    int events_per_segment,
    int segments_per_block,
    int record_blocks,
    int rank,
    int region_id,
    int kind,
    int64_t payload0,
    int64_t payload1) {
#if defined(__CUDA_ARCH__)
  int lane = threadIdx.x & 31;
  if (lane != 0) {
    return;
  }

  unsigned long long block = tl_profile_linear_block();
  unsigned long long warp = ((unsigned long long)(threadIdx.x >> 5));
  if (block >= (unsigned long long)record_blocks ||
      warp >= (unsigned long long)segments_per_block ||
      events_per_segment <= 0) {
    return;
  }

  unsigned long long segment_words = 1ULL + ((unsigned long long)events_per_segment) * 4ULL;
  unsigned long long segment = block * ((unsigned long long)segments_per_block) + warp;
  unsigned long long segment_base = segment * segment_words;
  int64_t* cursor = trace_buffer + segment_base;
  unsigned long long idx = atomicAdd((unsigned long long*)cursor, 1ULL);
  unsigned long long slot = idx % ((unsigned long long)events_per_segment);
  unsigned long long base = segment_base + 1ULL + slot * 4ULL;
  unsigned long long meta = tl_profile_pack_meta(rank, region_id, kind, block, warp);

  trace_buffer[base + 0ULL] = (int64_t)tl_profile_now();
  trace_buffer[base + 1ULL] = (int64_t)meta;
  trace_buffer[base + 2ULL] = payload0;
  trace_buffer[base + 3ULL] = payload1;
#endif
}

#endif  // TILELANG_PROFILE_RECORD_DEFINED
"""


def import_source() -> None:
    """Inject the TileLang profile device helper into the current prim_func."""

    T.import_source(_CUDA_SOURCE)


def segment_words(events_per_segment: int) -> int:
    """Number of int64 words in one per-warp trace segment."""

    return SEGMENT_HEADER_WORDS + int(events_per_segment) * WORDS_PER_EVENT


def segment_buffer_words(total_blocks: int, segments_per_block: int, events_per_segment: int) -> int:
    """Number of int64 words needed by a segmented trace buffer."""

    return int(total_blocks) * int(segments_per_block) * segment_words(events_per_segment)


def record(trace_buffer, events_per_segment, segments_per_block, record_blocks, rank, region_id, kind, payload0=0, payload1=0) -> None:
    """Emit a raw profile event from TileLang DSL code.

    ``trace_buffer`` is split into fixed segments, one per recorded warp. Each
    segment has a local cursor followed by ``events_per_segment`` event records.
    """

    T.evaluate(
        T.call_extern(
            "handle",
            "tl_profile_marker",
            T.address_of(trace_buffer[0]),
            events_per_segment,
            segments_per_block,
            record_blocks,
            rank,
            region_id,
            kind,
            payload0,
            payload1,
        )
    )


def begin(trace_buffer, events_per_segment, segments_per_block, record_blocks, rank, region_id, payload0=0, payload1=0) -> None:
    """Start a named region from TileLang DSL code."""

    record(trace_buffer, events_per_segment, segments_per_block, record_blocks, rank, region_id, EVENT_BEGIN, payload0, payload1)


def end(trace_buffer, events_per_segment, segments_per_block, record_blocks, rank, region_id, payload0=0, payload1=0) -> None:
    """End a named region from TileLang DSL code."""

    record(trace_buffer, events_per_segment, segments_per_block, record_blocks, rank, region_id, EVENT_END, payload0, payload1)


def mark(trace_buffer, events_per_segment, segments_per_block, record_blocks, rank, region_id, payload0=0, payload1=0) -> None:
    """Record an instantaneous marker from TileLang DSL code."""

    record(trace_buffer, events_per_segment, segments_per_block, record_blocks, rank, region_id, EVENT_MARK, payload0, payload1)


def pack_meta(rank: int, kind: int, region_id: int, block: int, warp: int) -> int:
    """Pack event metadata. Public mostly to keep tests and tools deterministic."""

    return ((rank & 0xFF) << 56) | ((kind & 0x3) << 54) | ((region_id & 0x3FFF) << 40) | ((block & 0xFFFFF) << 16) | (
        (warp & 0xFF) << 8
    )


def unpack_meta(meta: int) -> dict[str, int]:
    meta &= (1 << 64) - 1
    return {
        "rank": (meta >> 56) & 0xFF,
        "kind": (meta >> 54) & 0x3,
        "region_id": (meta >> 40) & 0x3FFF,
        "block": (meta >> 16) & 0xFFFFF,
        "warp": (meta >> 8) & 0xFF,
    }


@dataclass(frozen=True)
class TraceEvent:
    timestamp_ns: int
    rank: int
    kind: int
    region_id: int
    region: str
    block: int
    warp: int
    payload0: int = 0
    payload1: int = 0

    @property
    def kind_name(self) -> str:
        return _KIND_NAMES.get(self.kind, f"kind{self.kind}")


@dataclass(frozen=True)
class TraceSpan:
    name: str
    region_id: int
    rank: int
    block: int
    warp: int
    start_ns: int
    end_ns: int
    payload0: int = 0
    payload1: int = 0

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)


@dataclass
class PipelineDAG:
    """Small IKET-like metadata object for expected pipeline structure."""

    name: str = "pipeline"
    nodes: dict[int, str] = field(default_factory=dict)
    edges: list[tuple[int, int]] = field(default_factory=list)

    def region(self, region_id: int, name: str) -> int:
        self.nodes[int(region_id)] = name
        return int(region_id)

    def edge(self, src: int | str, dst: int | str) -> "PipelineDAG":
        self.edges.append((self._resolve(src), self._resolve(dst)))
        return self

    def _resolve(self, value: int | str) -> int:
        if isinstance(value, int):
            return value
        for region_id, name in self.nodes.items():
            if name == value:
                return region_id
        raise KeyError(f"unknown DAG node: {value}")

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "nodes": [{"id": region_id, "name": name} for region_id, name in sorted(self.nodes.items())],
            "edges": [{"src": src, "dst": dst} for src, dst in self.edges],
        }

    def write_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return out


def dag(name: str = "pipeline") -> PipelineDAG:
    return PipelineDAG(name=name)


def decode_records(
    records: Iterable[Iterable[int]],
    region_names: dict[int, str] | None = None,
    rank: int | None = None,
) -> list[TraceEvent]:
    names = region_names or {}
    events: list[TraceEvent] = []
    for row in records:
        timestamp_ns, meta, payload0, payload1 = [int(v) for v in row]
        fields = unpack_meta(meta)
        if rank is not None and fields["rank"] != rank:
            continue
        region_id = fields["region_id"]
        events.append(
            TraceEvent(
                timestamp_ns=timestamp_ns,
                rank=fields["rank"],
                kind=fields["kind"],
                region_id=region_id,
                region=names.get(region_id, f"region_{region_id}"),
                block=fields["block"],
                warp=fields["warp"],
                payload0=payload0,
                payload1=payload1,
            )
        )
    events.sort(key=lambda event: (event.timestamp_ns, event.rank, event.block, event.warp, event.kind))
    return events


def pair_spans(events: Iterable[TraceEvent]) -> list[TraceSpan]:
    stacks: dict[tuple[int, int, int, int, int, int], list[TraceEvent]] = {}
    spans: list[TraceSpan] = []
    for event in events:
        key = (event.rank, event.block, event.warp, event.region_id, event.payload0, event.payload1)
        if event.kind == EVENT_BEGIN:
            stacks.setdefault(key, []).append(event)
        elif event.kind == EVENT_END:
            starts = stacks.get(key)
            if starts:
                start = starts.pop()
                spans.append(
                    TraceSpan(
                        name=start.region,
                        region_id=start.region_id,
                        rank=start.rank,
                        block=start.block,
                        warp=start.warp,
                        start_ns=start.timestamp_ns,
                        end_ns=event.timestamp_ns,
                        payload0=start.payload0,
                        payload1=start.payload1,
                    )
                )
        elif event.kind == EVENT_MARK:
            spans.append(
                TraceSpan(
                    name=event.region,
                    region_id=event.region_id,
                    rank=event.rank,
                    block=event.block,
                    warp=event.warp,
                    start_ns=event.timestamp_ns,
                    end_ns=event.timestamp_ns,
                    payload0=event.payload0,
                    payload1=event.payload1,
                )
            )
    spans.sort(key=lambda span: (span.start_ns, span.rank, span.block, span.warp, span.region_id))
    return spans


class TraceSession:
    """Host-side owner/exporter for segmented TileLang profile events."""

    def __init__(
        self,
        events_per_segment: int = 256,
        segments_per_block: int = 8,
        total_blocks: int = 1,
        device: str = "cuda",
        region_names: dict[int, str] | None = None,
    ):
        if torch is None:
            raise RuntimeError("TraceSession requires torch")
        self.events_per_segment = int(events_per_segment)
        self.segments_per_block = int(segments_per_block)
        self.total_blocks = int(total_blocks)
        self.device = device
        self.region_names: dict[int, str] = dict(region_names or {})
        self.segment_words = segment_words(self.events_per_segment)
        self.buffer_words = segment_buffer_words(self.total_blocks, self.segments_per_block, self.events_per_segment)
        self.buffer = torch.empty((self.buffer_words,), dtype=torch.int64, device=device)
        self.reset()

    def reset(self) -> None:
        self.buffer.zero_()

    def tensors(self):
        return (self.buffer,)

    def region(self, name: str, region_id: int | None = None) -> int:
        if region_id is None:
            used = set(self.region_names)
            region_id = 1
            while region_id in used:
                region_id += 1
        self.region_names[int(region_id)] = name
        return int(region_id)

    def raw_records(self, synchronize: bool = True) -> list[list[int]]:
        if synchronize and self.buffer.is_cuda:
            torch.cuda.synchronize(self.buffer.device)
        flat = self.buffer.detach().cpu().reshape(-1)
        records: list[list[int]] = []
        for block in range(self.total_blocks):
            for warp in range(self.segments_per_block):
                segment = block * self.segments_per_block + warp
                start = segment * self.segment_words
                cursor = int(flat[start].item())
                if cursor <= 0:
                    continue
                retained = min(cursor, self.events_per_segment)
                data_start = start + SEGMENT_HEADER_WORDS
                data_stop = data_start + self.events_per_segment * WORDS_PER_EVENT
                rows = flat[data_start:data_stop].reshape(self.events_per_segment, WORDS_PER_EVENT)
                decoded_rows: list[list[int]] = []
                for row in rows.tolist():
                    values = [int(v) for v in row]
                    if values[0] != 0 or values[1] != 0:
                        decoded_rows.append(values)
                decoded_rows.sort(key=lambda row: row[0])
                records.extend(decoded_rows[-retained:])
        records.sort(key=lambda row: (row[0], row[1]))
        return records

    def events(self, rank: int | None = None, synchronize: bool = True) -> list[TraceEvent]:
        return decode_records(self.raw_records(synchronize=synchronize), self.region_names, rank=rank)

    def spans(self, rank: int | None = None, synchronize: bool = True) -> list[TraceSpan]:
        return pair_spans(self.events(rank=rank, synchronize=synchronize))

    def export_chrome_trace(self, path: str | Path, rank: int | None = None, synchronize: bool = True) -> Path:
        spans = self.spans(rank=rank, synchronize=synchronize)
        return write_chrome_trace(path, spans)

    def write_pipeline_svg(
        self,
        path: str | Path,
        dag: PipelineDAG | None = None,
        rank: int | None = None,
        synchronize: bool = True,
        title: str = "TileLang pipeline profile",
    ) -> Path:
        spans = self.spans(rank=rank, synchronize=synchronize)
        return write_pipeline_svg(path, spans, dag=dag, title=title)


def write_chrome_trace(path: str | Path, spans: Iterable[TraceSpan]) -> Path:
    spans = list(spans)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if spans:
        origin = min(span.start_ns for span in spans)
    else:
        origin = 0
    trace_events = []
    for span in spans:
        trace_events.append(
            {
                "name": span.name,
                "cat": "tilelang.profile",
                "ph": "X",
                "ts": (span.start_ns - origin) / 1000.0,
                "dur": max(1, span.duration_ns) / 1000.0,
                "pid": f"rank {span.rank}",
                "tid": f"block {span.block} warp {span.warp}",
                "args": {
                    "region_id": span.region_id,
                    "block": span.block,
                    "warp": span.warp,
                    "payload0": span.payload0,
                    "payload1": span.payload1,
                    "duration_ns": span.duration_ns,
                },
            }
        )
    payload = {"displayTimeUnit": "ns", "traceEvents": trace_events}
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def write_pipeline_svg(
    path: str | Path,
    spans: Iterable[TraceSpan],
    dag: PipelineDAG | None = None,
    title: str = "TileLang pipeline profile",
    max_rows: int = 48,
) -> Path:
    spans = list(spans)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    width = 1280
    left = 168
    right = 32
    row_h = 24
    dag_h = 76 if dag and dag.nodes else 0
    header_h = 52 + dag_h
    palette = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
        "#2F4B7C",
    ]

    if not spans:
        svg = "".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="160">',
                '<rect width="100%" height="100%" fill="#fbfbfc"/>',
                f'<text x="32" y="52" font-family="Inter,Arial" font-size="20" fill="#252a31">{escape(title)}</text>',
                '<text x="32" y="92" font-family="Inter,Arial" font-size="14" fill="#68707d">',
                "No profile spans were recorded.</text></svg>",
            ]
        )
        out.write_text(svg, encoding="utf-8")
        return out

    rows = sorted({(span.rank, span.block, span.warp) for span in spans})[:max_rows]
    row_index = {row: idx for idx, row in enumerate(rows)}
    visible = [span for span in spans if (span.rank, span.block, span.warp) in row_index]
    start = min(span.start_ns for span in visible)
    end = max(max(span.end_ns, span.start_ns + 1) for span in visible)
    duration = max(1, end - start)
    plot_w = width - left - right
    height = header_h + max(1, len(rows)) * row_h + 44

    def x_pos(ns: int) -> float:
        return left + ((ns - start) / duration) * plot_w

    def color(region_id: int) -> str:
        return palette[region_id % len(palette)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfc"/>',
        "<style>"
        "text{font-family:Inter,Arial,sans-serif}"
        ".label{fill:#28313d;font-size:12px}"
        ".muted{fill:#68707d;font-size:11px}"
        ".title{fill:#20242b;font-size:20px;font-weight:650}"
        "</style>",
        f'<text class="title" x="32" y="32">{escape(title)}</text>',
        f'<text class="muted" x="32" y="50">{duration / 1000.0:.3f} us window, {len(visible)} spans, {len(rows)} lanes shown</text>',
    ]

    if dag and dag.nodes:
        y = 76
        node_w = 126
        gap = 34
        ordered_nodes = sorted(dag.nodes.items())
        node_x: dict[int, int] = {}
        parts.append(f'<text class="muted" x="32" y="{y - 12}">expected DAG</text>')
        for idx, (region_id, name) in enumerate(ordered_nodes):
            x = 32 + idx * (node_w + gap)
            node_x[region_id] = x
            parts.append(f'<rect x="{x}" y="{y}" width="{node_w}" height="30" rx="5" fill="{color(region_id)}" opacity="0.90"/>')
            parts.append(f'<text x="{x + 10}" y="{y + 20}" fill="white" font-size="12">{escape(name[:17])}</text>')
        for src, dst in dag.edges:
            if src in node_x and dst in node_x:
                x1 = node_x[src] + node_w
                x2 = node_x[dst]
                yy = y + 15
                parts.append(f'<path d="M{x1} {yy} L{x2 - 8} {yy}" stroke="#3b4450" stroke-width="1.4" fill="none"/>')
                parts.append(f'<path d="M{x2 - 8} {yy - 4} L{x2} {yy} L{x2 - 8} {yy + 4}" fill="#3b4450"/>')

    axis_y = header_h - 14
    parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{axis_y}" y2="{axis_y}" stroke="#c7ccd4"/>')
    for tick in range(6):
        x = left + tick * plot_w / 5
        ns = start + tick * duration / 5
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{axis_y}" y2="{height - 28}" stroke="#e6e8ec"/>')
        parts.append(f'<text class="muted" x="{x - 14:.1f}" y="{axis_y - 6}">{(ns - start) / 1000.0:.1f}us</text>')

    for row, idx in row_index.items():
        y = header_h + idx * row_h
        rank, block, warp = row
        parts.append(f'<text class="label" x="32" y="{y + 16}">r{rank} b{block} w{warp}</text>')
        parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y + 20}" y2="{y + 20}" stroke="#eceff3"/>')

    for span in visible:
        idx = row_index[(span.rank, span.block, span.warp)]
        y = header_h + idx * row_h + 4
        x = x_pos(span.start_ns)
        w = max(1.5, x_pos(max(span.end_ns, span.start_ns + 1)) - x)
        tooltip = escape(
            f"{span.name} block={span.block} warp={span.warp} "
            f"payload=({span.payload0},{span.payload1}) dur={span.duration_ns}ns"
        )
        parts.append(
            f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="15" rx="3" '
            f'fill="{color(span.region_id)}" opacity="0.88"><title>{tooltip}</title></rect>'
        )
        if w > 46:
            parts.append(f'<text x="{x + 4:.2f}" y="{y + 11}" fill="white" font-size="10">{escape(span.name[:18])}</text>')

    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
