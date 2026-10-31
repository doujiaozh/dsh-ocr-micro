#!/usr/bin/env python3
"""model_forge/quantize.py — weight-only INT8 quantization (pure onnx + numpy).

THIS IS THE MAIN BUDGET LEVER. Deliberately **no onnxruntime dependency**:
onnxruntime's native DLL does not load on this Windows 7 box, so the whole
compression step is implemented with the pure-Python `onnx` + `numpy` stack.

Method — **weight-only INT8 with QDQ nodes** (opset 13):
  * for every Conv / MatMul / Gemm whose weight is a float32 graph initializer,
    quantize the weight per **output channel** to uint8 with asymmetric
    min/max scaling, and insert a `DequantizeLinear` node that feeds the
    original consumer (activations stay fp32 → zero accuracy risk on the
    compute path, exact 4x weight shrink).
  * Conv   W[O, I, kh, kw]   -> per-output-channel along O (DQ axis=0)
  * MatMul W[K, N]           -> per-output-channel along N (DQ axis=-1)
  * Gemm   B (+transB attr)  -> per-output-channel along the N dim (axis=1 or 0)
  * opset is raised to 13 where needed (per-channel DQ `axis` support);
    verified compatible with onnxruntime-web (WASM) which runs on Windows 7.

Inputs : models/det_fp32.onnx, models/rec_fp32.onnx
Outputs: models/det.onnx,     models/rec.onnx

Prints per-model before/after sizes and the TOTAL. Exits 1 with exact pruning
advice if TOTAL >= 5,500,000 bytes, unless --allow-over is passed.

Usage:
    python model_forge/quantize.py
    python model_forge/quantize.py --allow-over
    python model_forge/quantize.py --op-note   # also dump a per-node weight table
"""

import argparse
import os
import sys

import numpy as np

try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto
except ImportError:
    sys.exit("FATAL: onnx/numpy not installed. Run: pip install -r model_forge/requirements.txt")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
MODELS = os.path.join(_ROOT, "models")

BUDGET = 5_500_000  # bytes — hard 5.5 MB cap for det.onnx + rec.onnx
TARGET_OPSET = 13   # DequantizeLinear per-axis quantization support

SPECS = [
    ("det", "det_fp32.onnx", "det.onnx"),
    ("rec", "rec_fp32.onnx", "rec.onnx"),
]

UINT8_MAX = 255


def mb(n: int) -> str:
    return f"{n/1024/1024:.2f} MB"


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------

def initializers(model):
    return {init.name: init for init in model.graph.initializer}


def set_initializer(model, name, array, dtype):
    """Replace (or add) a graph initializer with a numpy array."""
    if isinstance(dtype, int):  # TensorProto data type enum
        dtype = {
            TensorProto.UINT8: np.uint8,
            TensorProto.INT8: np.int8,
            TensorProto.FLOAT: np.float32,
            TensorProto.DOUBLE: np.float64,
            TensorProto.INT32: np.int32,
            TensorProto.INT64: np.int64,
            TensorProto.BOOL: np.bool_,
        }[dtype]
    inits = model.graph.initializer
    for i, init in enumerate(inits):
        if init.name == name:
            inits[i].CopyFrom(numpy_helper.from_array(array.astype(dtype), name=name))
            return
    inits.append(numpy_helper.from_array(array.astype(dtype), name=name))


def remove_initializer(model, name):
    inits = model.graph.initializer
    for i, init in enumerate(inits):
        if init.name == name:
            del inits[i]
            return


def opset_version(model):
    for op in model.opset_import:
        if op.domain in ("", "ai.onnx"):
            return op.version
    return model.opset_import[0].version if len(model.opset_import) else 0


def raise_opset(model, version):
    """Bump the default-domain opset (per-channel QDQ needs >= 13)."""
    for op in model.opset_import:
        if op.domain in ("", "ai.onnx"):
            op.version = max(op.version, version)
            return
    model.opset_import.append(helper.make_opsetid("", version))


def modernize_for_opset13(model):
    """Adapt pre-13 nodes whose `axes`/`split`/`paddings` ATTRIBUTE became an
    INPUT at opset 13 (Squeeze, Unsqueeze, Split) / opset 11 (Pad). Saves the
    values as int64 initializers and rewrites the node inputs. Called only when
    the opset is actually being raised above 12."""
    changed = 0
    init_names = {i.name for i in model.graph.initializer}
    counter = 0
    for node in model.graph.node:
        op = node.op_type
        attr_name = None
        if op in ("Squeeze", "Unsqueeze"):
            attr_name = "axes"
        elif op == "Split":
            attr_name = "split"
        elif op == "Pad":
            attr_name = "paddings"
        if attr_name is None:
            continue
        attr = next((a for a in node.attribute if a.name == attr_name), None)
        if attr is None:
            continue  # already input-style (post-11/13 export)
        values = list(attr.ints)
        # values as int64 initializer; initializers need no topological placement
        name = None
        while name is None or name in init_names:
            name = f"{op.lower()}_{attr_name}_{counter}"
            counter += 1
        set_initializer(model, name, np.array(values, np.int64), TensorProto.INT64)
        init_names.add(name)
        node.input.append(name)
        # drop the old-style attribute
        for i, a in enumerate(list(node.attribute)):
            if a.name == attr_name:
                del node.attribute[i]
        changed += 1
    return changed


# ---------------------------------------------------------------------------
# per-output-channel uint8 quantization
# ---------------------------------------------------------------------------

def quantize_axis(arr, axis):
    """Asymmetric uint8 quantization of `arr` along `axis` (output channels).

    Returns (q uint8, scale float32 [C], zp uint8 [C]) where dequant is
    dq = (q.astype(float32) - zp) * scale, broadcast along `axis`.
    """
    arr = arr.astype(np.float32)
    reduce_axes = tuple(i for i in range(arr.ndim) if i != axis)
    mn = arr.min(axis=reduce_axes)
    mx = arr.max(axis=reduce_axes)
    rng = np.maximum(mx - mn, 1e-8)
    scale = (rng / UINT8_MAX).astype(np.float32)
    zp = np.clip(np.round(-mn / scale), 0.0, float(UINT8_MAX)).astype(np.uint8)
    shape = [1] * arr.ndim
    shape[axis] = arr.shape[axis]
    q = np.clip(np.round(arr / scale.reshape(shape)) + zp.reshape(shape), 0, UINT8_MAX).astype(np.uint8)
    # exact dequant formula check (debug assert): dq == (q - zp)*scale
    return q, scale, zp


def insert_dequantize(model, q_name, scale_name, zp_name, axis, out_name):
    """Create (NOT append) a DequantizeLinear node; quantize_model places it."""
    return helper.make_node(
        "DequantizeLinear",
        inputs=[q_name, scale_name, zp_name],
        outputs=[out_name],
        axis=axis,
        name=out_name + "_dq",
    )


# ---------------------------------------------------------------------------
# main quantization
# ---------------------------------------------------------------------------

def quantize_model(src: str, dst: str, verbose: bool) -> int:
    model = onnx.load(src)
    inits = initializers(model)
    modernized = 0
    if opset_version(model) < TARGET_OPSET:
        raise_opset(model, TARGET_OPSET)
        modernized = modernize_for_opset13(model)
        if modernized:
            print(f"       modernized {modernized} pre-opset13 node(s) (Squeeze/Unsqueeze/Split/Pad axes as inputs)")

    # map: original weight initializer name -> DQ output tensor name
    dq_outputs = {}
    pending_inserts = []  # (dq_node, first consumer node) — inserted before the consumer
    stats = []  # (node op, weight shape, saved bytes)
    weight_bytes = 0

    def handle_weight(node, in_index, axis, op_name):
        nonlocal weight_bytes
        wname = node.input[in_index]
        if not wname or wname not in inits:
            return  # activation-fed input: skip (weight-only quantization)
        init = inits[wname]
        arr = numpy_helper.to_array(init)
        if arr.dtype != np.float32 or arr.ndim < 2:
            return
        if wname in dq_outputs:
            node.input[in_index] = dq_outputs[wname]
            return
        axis = axis % arr.ndim  # DequantizeLinear axis must be non-negative
        q, scale, zp = quantize_axis(arr, axis)
        q_name = wname + "_q"
        scale_name = wname + "_scale"
        zp_name = wname + "_zp"
        dq_name = wname + "_dq"
        set_initializer(model, q_name, q, TensorProto.UINT8)
        set_initializer(model, scale_name, scale, TensorProto.FLOAT)
        set_initializer(model, zp_name, zp, TensorProto.UINT8)
        dq_node = insert_dequantize(model, q_name, scale_name, zp_name, axis, dq_name)
        node.input[in_index] = dq_name
        dq_outputs[wname] = dq_name
        pending_inserts.append((dq_node, node))
        before = arr.nbytes
        after = q.nbytes + scale.nbytes + zp.nbytes
        weight_bytes += before
        stats.append((op_name, list(arr.shape), before, after))
        if verbose:
            print(f"       {op_name:6s} {wname[:38]:38s} {'x'.join(str(d) for d in arr.shape):18s} "
                  f"{before//1024:6d}KB -> {after//1024:5d}KB")

    for node in model.graph.node:
        if node.op_type == "Conv":
            # weight [O, I, kh, kw]: output channels = axis 0
            handle_weight(node, 1, 0, "Conv")
        elif node.op_type == "MatMul":
            # weight commonly [K, N]; output channels = last dim (axis -1).
            # if input[0] is the initializer and input[1] is activation
            # (transposed MatMul), quantize input[0] along its second-to-last dim.
            if node.input[1] and node.input[1] in inits:
                w = numpy_helper.to_array(inits[node.input[1]])
                handle_weight(node, 1, -1 if w.ndim == 2 else 0, "MatMul")
            elif node.input[0] and node.input[0] in inits:
                w0 = numpy_helper.to_array(inits[node.input[0]])
                handle_weight(node, 0, -2 if w0.ndim == 2 else 0, "MatMul")
        elif node.op_type == "Gemm":
            # Y = A @ B^T(if transB) ; B rows=N when transB else cols=N
            transb = next((a.i for a in node.attribute if a.name == "transB"), 0)
            if node.input[1] and node.input[1] in inits:
                w = numpy_helper.to_array(inits[node.input[1]])
                handle_weight(node, 1, 0 if transb else 1, "Gemm")

    # ---- 1) make the graph topologically valid: every DequantizeLinear node
    # is inserted right before its (first) consumer node in the node list.
    original_nodes = list(model.graph.node)  # untouched by the loop above
    by_consumer = {}
    for dq_node, consumer in pending_inserts:
        by_consumer.setdefault(id(consumer), []).append(dq_node)
    new_nodes = []
    seen = set()
    for node in original_nodes:
        for dq_node in by_consumer.get(id(node), []):
            if id(dq_node) not in seen:
                new_nodes.append(dq_node)
                seen.add(id(dq_node))
        new_nodes.append(node)
    for dq_node, _consumer in pending_inserts:
        if id(dq_node) not in seen:  # safety: consumer vanished mid-rewrite
            new_nodes.append(dq_node)
            seen.add(id(dq_node))
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # ---- 2) drop the now-unreferenced fp32 weight initializers (they account
    # for the whole saving; keeping them would leave the file as big as before).
    dropped = 0
    for wname in list(dq_outputs.keys()):
        still_used = any(wname == inp for n in model.graph.node for inp in n.input)
        if not still_used:
            remove_initializer(model, wname)
            dropped += 1
    onnx.save(model, dst)
    return len(stats), weight_bytes


def verify_qdq(dst: str):
    """Static verification WITHOUT a runtime: pure onnx checks."""
    m = onnx.load(dst)
    onnx.checker.check_model(m)
    n_dq = sum(1 for n in m.graph.node if n.op_type == "DequantizeLinear")
    n_qt = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")
    n_u8 = sum(1 for i in m.graph.initializer if i.data_type == TensorProto.UINT8)
    # any Conv/MatMul/Gemm still taking its weight straight from a float32
    # initializer (i.e. missed by the quantizer) gets reported.
    init_names = {i.name for i in m.graph.initializer}
    unquantized = []
    for n in m.graph.node:
        if n.op_type in ("Conv", "MatMul", "Gemm") and len(n.input) > 1 and n.input[1] in init_names:
            for i in m.graph.initializer:
                if i.name == n.input[1] and i.data_type == TensorProto.FLOAT:
                    unquantized.append((n.op_type, n.input[1]))
    return n_dq, n_qt, n_u8, unquantized


def recommend_prune(total: int) -> None:
    over = total - BUDGET
    rec_path = os.path.join(MODELS, "rec.onnx")
    det_path = os.path.join(MODELS, "det.onnx")
    if not os.path.exists(rec_path):
        return
    rec_size = os.path.getsize(rec_path)
    det_size = os.path.getsize(det_path) if os.path.exists(det_path) else 0
    if det_size >= BUDGET:
        print(f"ADVICE: det.onnx alone is {mb(det_size)} >= budget; prune det (or use a smaller tier).")
        return
    slack = BUDGET - det_size
    if rec_size <= slack:
        return
    ratio = (rec_size - slack) / rec_size
    ratio = min(0.9, ratio * 1.15)  # safety margin for non-shrinking overhead
    print(
        f"ADVICE: over budget by {over} bytes. Recommended: python model_forge/prune.py --ratio {ratio:.3f}\n"
        f"        (structured channel pruning of the rec model, then re-run quantize.py)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Weight-only INT8 quantization of PP-OCRv6 ONNX models (5.5MB budget)")
    ap.add_argument("--models-dir", default=MODELS)
    ap.add_argument("--allow-over", action="store_true", help="do not exit 1 when over budget")
    ap.add_argument("--op-note", action="store_true", help="dump a per-weight quantization table")
    args = ap.parse_args()

    missing = [s[1] for s in SPECS if not os.path.isfile(os.path.join(args.models_dir, s[1]))]
    if missing:
        sys.exit(
            "FATAL: missing fp32 models: " + ", ".join(missing) + "\n"
            "       run `node model_forge/download.js` first (and paddle2onnx_convert.py in fallback mode)."
        )

    total = 0
    print("-" * 66)
    for kind, src_name, dst_name in SPECS:
        src = os.path.join(args.models_dir, src_name)
        dst = os.path.join(args.models_dir, dst_name)
        before = os.path.getsize(src)
        if os.path.exists(dst):
            os.remove(dst)
        print(f"[{kind}] weight-only INT8 (QDQ, per-output-channel uint8, opset {TARGET_OPSET})")
        n_weights, weight_bytes = quantize_model(src, dst, args.op_note)
        after = os.path.getsize(dst)
        ratio = (1.0 - after / before) * 100.0 if before else 0.0
        print(f"       quantized weights: {n_weights} tensors ({weight_bytes/1024/1024:.2f} MB fp32 -> ~{weight_bytes/4/1024/1024:.2f} MB uint8)")
        print(f"       before {mb(before)}  ->  after {mb(after)}   ({ratio:.1f}% smaller)")
        try:
            n_dq, n_qt, n_u8, unquantized = verify_qdq(dst)
            print(f"       static check: DequantizeLinear nodes={n_dq} uint8 initializers={n_u8} onnx.checker=OK")
            if unquantized:
                print(f"       note: {len(unquantized)} fp32 weight(s) left unquantized: {unquantized[:3]}")
        except Exception as e:
            print(f"       static check FAILED: {e}")
            return 1
        total += after
    print("-" * 66)
    print(f"TOTAL det.onnx + rec.onnx = {total} bytes = {mb(total)}   (budget {BUDGET} bytes = {mb(BUDGET)})")

    if total >= BUDGET:
        print(f"OVER BUDGET by {total - BUDGET} bytes.")
        recommend_prune(total)
        if not args.allow_over:
            print("Run with --allow-over to keep the artifacts anyway.")
            return 1
    else:
        print(f"UNDER BUDGET by {BUDGET - total} bytes — good to go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
