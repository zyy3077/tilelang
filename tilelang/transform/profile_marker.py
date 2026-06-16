from tilelang import tvm as tvm
from tvm import tir
from tvm.tir import Call, Evaluate, PrimFunc, SeqStmt, StringImm
from tvm.tir.stmt_functor import ir_transform
from tvm.tir.transform import prim_func_pass


MARKER_EXTERN = "tl_profile_marker"
RECORD_EXTERN = "tl_profile_record"
AUTO_PRODUCER_REGION_ID = 250
EVENT_BEGIN = 0
EVENT_END = 1
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


def _make_marker_from_template(template: Call, region_id: int, kind: int):
    args = list(template.args)
    if len(args) < 10:
        return None
    args[6] = tir.IntImm("int32", region_id)
    args[7] = tir.IntImm("int32", kind)
    args[8] = tir.IntImm("int32", 0)
    args[9] = tir.IntImm("int32", 0)
    return Evaluate(tir.call_extern(template.dtype, MARKER_EXTERN, *args[1:]))


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

    tir.stmt_functor.post_order_visit(stmt, visit)
    return template


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


def LowerProfileMarkers():
    """Lower inert profile markers after pipeline and warp-specialized passes.

    The frontend emits ``tl_profile_marker`` as a placeholder so pipeline passes
    can recognize and group it with neighboring real statements. Late in the
    lowering pipeline this pass renames the marker call to the device helper
    that actually writes the segmented trace buffer.
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        def post_visit(stmt):
            if not isinstance(stmt, Evaluate):
                return stmt
            value = stmt.value
            if not isinstance(value, Call) or not _is_profile_marker_call(value):
                return stmt

            return Evaluate(tir.call_extern(value.dtype, RECORD_EXTERN, *value.args[1:]))

        return func.with_body(ir_transform(func.body, None, post_visit))

    return prim_func_pass(pass_fn, opt_level=0, name="tl.LowerProfileMarkers")
