from tvm import tir
from tvm.tir import (
    AttrStmt,
    BufferLoad,
    BufferStore,
    Call,
    Evaluate,
    For,
    IfThenElse,
    LetStmt,
    PrimFunc,
    SeqStmt,
    StringImm,
)
from tvm.tir.stmt_functor import ir_transform
from tvm.tir.transform import prim_func_pass


MARKER_EXTERN = "tl_profile_marker"
RECORD_EXTERN = "tl_profile_record"
PROFILE_REGION_ATTR = "tl.profile_region"
AUTO_PRODUCER_REGION_ID = 250
AUTO_SIMT_COPY_REGION_ID = 251
EVENT_BEGIN = 0
EVENT_END = 1

# Lowered async copy operations keep a stable intrinsic shape late in the
# pipeline, so they can be wrapped after warp-specialization rewrites.
COPY_OPS = {
    "tl.tma_load",
    "tl.tma_load_im2col",
    "tir.ptx_cp_async",
    "tir.ptx_cp_async_bulk",
}


def _is_profile_marker_call(call: Call) -> bool:
    if not call.op.same_as(tir.op.Op.get("tir.call_extern")):
        return False
    if len(call.args) == 0:
        return False
    name = call.args[0]
    return isinstance(name, StringImm) and name.value == MARKER_EXTERN


def _call_op_name(call: Call) -> str | None:
    op = call.op
    return getattr(op, "name", None)


def _is_copy_event_call(call: Call) -> bool:
    return _call_op_name(call) in COPY_OPS


def _buffer_scope(buffer) -> str:
    scope = getattr(buffer, "scope", None)
    if callable(scope):
        scope = scope()
    if scope is None:
        return ""
    return str(scope)


def _is_shared_scope(scope: str) -> bool:
    return scope == "shared" or scope.startswith("shared.")


def _is_global_scope(scope: str) -> bool:
    return scope in {"", "global"}


def _expr_has_global_load(expr) -> bool:
    found = False

    def visit(node):
        nonlocal found
        if found:
            return
        if isinstance(node, BufferLoad) and _is_global_scope(_buffer_scope(node.buffer)):
            found = True

    tir.stmt_functor.post_order_visit(expr, visit)
    return found


def _is_simt_g2s_store(stmt) -> bool:
    return (
        isinstance(stmt, BufferStore)
        and _is_shared_scope(_buffer_scope(stmt.buffer))
        and _expr_has_global_load(stmt.value)
    )


def _is_pure_simt_g2s_copy_stmt(stmt) -> bool:
    if _is_simt_g2s_store(stmt):
        return True
    if isinstance(stmt, SeqStmt):
        return len(stmt.seq) > 0 and all(_is_pure_simt_g2s_copy_stmt(child) for child in stmt.seq)
    if isinstance(stmt, IfThenElse):
        if not _is_pure_simt_g2s_copy_stmt(stmt.then_case):
            return False
        return stmt.else_case is None or _is_pure_simt_g2s_copy_stmt(stmt.else_case)
    if isinstance(stmt, LetStmt):
        return _is_pure_simt_g2s_copy_stmt(stmt.body)
    if isinstance(stmt, AttrStmt):
        return _is_pure_simt_g2s_copy_stmt(stmt.body)
    if isinstance(stmt, For):
        return _is_pure_simt_g2s_copy_stmt(stmt.body)
    return False


def _is_profile_region_attr(stmt: AttrStmt) -> bool:
    return stmt.attr_key == PROFILE_REGION_ATTR and isinstance(stmt.node, Call) and _is_profile_marker_call(stmt.node)


def _make_event_from_template(
    template: Call,
    extern_name: str,
    region_id: int | None = None,
    kind: int | None = None,
    payload0=None,
    payload1=None,
):
    args = list(template.args)
    if len(args) < 10:
        return None
    if region_id is not None:
        args[6] = tir.IntImm("int32", region_id)
    if kind is not None:
        args[7] = tir.IntImm("int32", kind)
    if payload0 is not None:
        args[8] = payload0
    if payload1 is not None:
        args[9] = payload1
    return Evaluate(tir.call_extern(template.dtype, extern_name, *args[1:]))


def _make_marker_from_template(template: Call, region_id: int, kind: int):
    return _make_event_from_template(
        template,
        MARKER_EXTERN,
        region_id=region_id,
        kind=kind,
        payload0=tir.IntImm("int32", 0),
        payload1=tir.IntImm("int32", 0),
    )


def _make_record_from_scope_template(template: Call, kind: int):
    return _make_event_from_template(template, RECORD_EXTERN, kind=kind)


def _make_scope_marker(template: Call, kind: int):
    return _make_event_from_template(template, MARKER_EXTERN, kind=kind)


def _find_profile_template(stmt):
    template = None

    def visit(node):
        nonlocal template
        if template is not None:
            return
        if isinstance(node, Evaluate):
            value = node.value
            if isinstance(value, Call) and _is_profile_marker_call(value):
                template = value
        elif isinstance(node, AttrStmt) and _is_profile_region_attr(node):
            template = node.node

    tir.stmt_functor.post_order_visit(stmt, visit)
    return template


def _flatten_seq(stmt):
    if isinstance(stmt, SeqStmt):
        flattened = []
        for child in stmt.seq:
            flat_child = _flatten_seq(child)
            if isinstance(flat_child, SeqStmt):
                flattened.extend(flat_child.seq)
            else:
                flattened.append(flat_child)
        return SeqStmt(flattened)
    return stmt


def ExpandProfileRegions():
    """Expand scoped profile regions into inert begin/body/end markers.

    ``tl_profile.scope`` emits a TIR AttrStmt so the Python DSL has a real scoped
    surface. Pipeline planning still expects the original top-level statement
    structure, especially when users provide explicit ``group`` annotations. This
    pass runs before pipeline and warp-specialization rewrites, expands the scope
    to begin/end marker statements, and flattens the resulting SeqStmt so grouped
    pipeline indices continue to refer to the scoped body statements.
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        def post_visit(stmt):
            if isinstance(stmt, AttrStmt) and _is_profile_region_attr(stmt):
                begin = _make_scope_marker(stmt.node, EVENT_BEGIN)
                end = _make_scope_marker(stmt.node, EVENT_END)
                if begin is None or end is None:
                    return stmt.body
                body = _flatten_seq(stmt.body)
                if isinstance(body, SeqStmt):
                    return SeqStmt([begin, *body.seq, end])
                return SeqStmt([begin, body, end])
            if isinstance(stmt, SeqStmt):
                return _flatten_seq(stmt)
            return stmt

        return func.with_body(ir_transform(func.body, None, post_visit))

    return prim_func_pass(pass_fn, opt_level=0, name="tl.ExpandProfileRegions")


def AutoProfileCopyMarkers(region_id: int = AUTO_PRODUCER_REGION_ID):
    """Add profile markers around lowered async producer-copy intrinsics.

    This pass runs after pipeline/warp-specialization rewriting and before
    ``LowerProfileMarkers``. It uses any surviving frontend profile marker as a
    template for the trace-buffer arguments, then wraps concrete copy intrinsics
    such as TMA loads and cp.async. This makes producer-side events follow the
    final lowered copy path instead of relying on frontend ``T.copy`` statements
    to survive scheduling rewrites.
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        template = _find_profile_template(func.body)
        if template is None:
            return func

        begin = _make_marker_from_template(template, region_id, EVENT_BEGIN)
        end = _make_marker_from_template(template, region_id, EVENT_END)
        if begin is None or end is None:
            return func

        def post_visit(stmt):
            if not isinstance(stmt, Evaluate):
                return stmt
            value = stmt.value
            if not isinstance(value, Call) or not _is_copy_event_call(value):
                return stmt
            return SeqStmt([begin, stmt, end])

        return func.with_body(ir_transform(func.body, None, post_visit))

    return prim_func_pass(pass_fn, opt_level=0, name="tl.AutoProfileCopyMarkers")


def AutoProfileSimtCopyMarkers(region_id: int = AUTO_SIMT_COPY_REGION_ID):
    """Add profile markers around ordinary SIMT global-to-shared copy loops.

    Plain ``T.copy`` may lower to TIR loops containing shared-memory stores whose
    value is loaded from global memory, without producing TMA or cp.async
    intrinsics. This pass runs after buffer/storage rewriting but before loop
    unrolling, so it can wrap the whole lowered copy loop as one span instead of
    recording one event pair per scalar store.
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        template = _find_profile_template(func.body)
        if template is None:
            return func

        begin = _make_marker_from_template(template, region_id, EVENT_BEGIN)
        end = _make_marker_from_template(template, region_id, EVENT_END)
        if begin is None or end is None:
            return func

        def post_visit(stmt):
            if isinstance(stmt, For) and _is_pure_simt_g2s_copy_stmt(stmt.body):
                return SeqStmt([begin, stmt, end])
            return stmt

        return func.with_body(ir_transform(func.body, None, post_visit))

    return prim_func_pass(pass_fn, opt_level=0, name="tl.AutoProfileSimtCopyMarkers")


def LowerProfileMarkers():
    """Lower inert profile markers after pipeline and warp-specialized passes.

    The frontend emits ``tl_profile_marker`` as a placeholder so pipeline passes
    can recognize and group it with neighboring real statements. Late in the
    lowering pipeline this pass renames the marker call to the device helper
    that actually writes the segmented trace buffer.
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        def post_visit(stmt):
            if isinstance(stmt, AttrStmt) and _is_profile_region_attr(stmt):
                begin = _make_record_from_scope_template(stmt.node, EVENT_BEGIN)
                end = _make_record_from_scope_template(stmt.node, EVENT_END)
                if begin is None or end is None:
                    return stmt.body
                return SeqStmt([begin, stmt.body, end])

            if not isinstance(stmt, Evaluate):
                return stmt
            value = stmt.value
            if not isinstance(value, Call) or not _is_profile_marker_call(value):
                return stmt

            return Evaluate(tir.call_extern(value.dtype, RECORD_EXTERN, *value.args[1:]))

        return func.with_body(ir_transform(func.body, None, post_visit))

    return prim_func_pass(pass_fn, opt_level=0, name="tl.LowerProfileMarkers")
