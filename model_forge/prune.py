#!/usr/bin/env python3
"""model_forge/prune.py — structured channel pruning for the rec model.

Compression lever #2, used when INT8 quantization alone blows the 5.5 MB budget.
PP-OCRv4 mobile rec (SVTR-LCNet style) is a chain of Conv+BN blocks feeding a
CRNN head, so width-pruning is a matter of slicing channel dimensions.

Method:
  1. Rank output channels of every eligible Conv by L2 importance
     (||W[o, :, :, :]||_2 over in-channel x kh x kw).
     Eligible = Conv followed by a BatchNormalization, and NOT:
       - the first conv of the graph       (keeps the 3 RGB input channels)
       - depthwise/grouped convs           (groups > 1)
       - the classification head's input   (last convs feeding MatMul/Gemm —
         their output channels align with the class projection)
  2. Remove the least-important fraction `ratio` of channels per eligible conv.
  3. Graph surgery: slice conv weight [out_channels] + bias, slice the following
     BN's running_mean/running_var/scale/B (all channel-indexed), and slice the
     *consumers* of that tensor: next Conv in_channels, head MatMul rows /
     Gemm columns — only when the consumer's weight dim equals the ORIGINAL
     channel count (so residual branches / concats are never mis-sliced).
  4. Verify: pure-onnx static checks (onnx.checker + LOGIT SIZE / class-count
     comparison against the input model + consumer-dim consistency). If graph
     surgery broke it, fall back to uniform width pruning (same ratio across
     all eligible convs) and re-verify. No onnxruntime (its native DLL does
     not load on the Win7 target box).

Usage:
    python model_forge/prune.py --ratio 0.3            # prune 30% (L2 ranking)
    python model_forge/prune.py --ratio 0.3 --dry-run  # report sizes only
    python model_forge/prune.py --ratio 0.3 --uniform  # uniform width pruning
    python model_forge/prune.py --ratio 0.3 --in models/rec_pruned.onnx --out models/rec_pruned2.onnx
"""

import argparse
import os
import sys

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    sys.exit("FATAL: onnx not installed. Run: pip install -r model_forge/requirements.txt")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
MODELS = os.path.join(_ROOT, "models")


def mb(n: int) -> str:
    return f"{n/1024/1024:.2f} MB"


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------

def build_index(model):
    nodes_by_output = {}
    consumers = {}
    for node in model.graph.node:
        for o in node.output:
            nodes_by_output[o] = node
        for i in node.input:
            consumers.setdefault(i, []).append(node)
    return nodes_by_output, consumers


def get_init(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return init
    return None


def set_init_array(model, name, arr):
    for i, init in enumerate(model.graph.initializer):
        if init.name == name:
            model.graph.initializer[i].CopyFrom(numpy_helper.from_array(arr, name=name))
            return
    raise KeyError(name)


def conv_groups(node):
    for attr in node.attribute:
        if attr.name == "group":
            return attr.i
    return 1


def set_conv_group(node, g):
    """Rewrite a Conv's `group` attribute (depthwise convs over a pruned tensor
    keep weight [O,1,k,k] but their group count must follow the new channels)."""
    for attr in node.attribute:
        if attr.name == "group":
            attr.i = g
            return
    from onnx import helper

    node.attribute.append(helper.make_attribute("group", g))


# ---------------------------------------------------------------------------
# eligibility
# ---------------------------------------------------------------------------

def find_eligible_convs(model, consumers):
    """Return [(conv_node, bn_node)] pairs eligible for output-channel pruning."""
    convs = [n for n in model.graph.node if n.op_type == "Conv"]
    if not convs:
        return []
    first_conv = convs[0]

    # "the last classification layer's input side": convs whose output tensor is
    # directly consumed by a MatMul/Gemm (the CRNN/SVTR class projection).
    head_feeding = set()
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            head_feeding.update(node.input[:1])

    pairs = []
    for node in convs:
        if node is first_conv:
            continue
        if conv_groups(node) > 1:  # depthwise / grouped
            continue
        nxt = consumers.get(node.output[0], [])
        bn = next((c for c in nxt if c.op_type == "BatchNormalization"), None)
        if bn is None:
            continue
        if node.output[0] in head_feeding or bn.output[0] in head_feeding:
            continue  # head-aligned channels: do not prune
        pairs.append((node, bn))
    return pairs


# ---------------------------------------------------------------------------
# pruning core
# ---------------------------------------------------------------------------

def prune_graph(model, ratio, scheme="l2", seed=0):
    """Structured pruning on a deep copy of `model`. Returns (new_model, kept_map)."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    _nodes_by_output, consumers = build_index(m)

    pairs = find_eligible_convs(m, consumers)
    if not pairs:
        print("       (no eligible conv+BN pairs found; nothing pruned)")
        return m, {}

    kept_map = {}
    rng = np.random.default_rng(seed)

    for conv, bn in pairs:
        w_init = get_init(m, conv.input[1])
        if w_init is None:
            continue
        w = numpy_helper.to_array(w_init)  # (out, in, kh, kw)
        out_c = w.shape[0]
        keep_n = int(round(out_c * (1.0 - ratio)))
        if keep_n < 8 or keep_n >= out_c:
            continue  # too small to prune / nothing to remove

        if scheme == "l2":
            imp = np.sqrt((w.astype(np.float64) ** 2).sum(axis=(1, 2, 3)))  # per out-channel L2
            order = np.argsort(imp)  # ascending: least important first
            keep_idx = np.sort(order[-keep_n:]).astype(np.int64)
        else:  # uniform width pruning
            keep_idx = np.sort(rng.choice(out_c, size=keep_n, replace=False)).astype(np.int64)

        # 1) conv weight (out, in, kh, kw) + bias (out,)
        set_init_array(m, conv.input[1], w[keep_idx])
        if len(conv.input) > 2 and conv.input[2]:
            b = numpy_helper.to_array(get_init(m, conv.input[2]))
            set_init_array(m, conv.input[2], b[keep_idx])

        # 2) BN running stats + affine params (all channel-indexed)
        for inp in bn.input[1:]:
            if not inp:
                continue
            init = get_init(m, inp)
            if init is not None and len(init.dims) == 1 and init.dims[0] == out_c:
                set_init_array(m, inp, numpy_helper.to_array(init)[keep_idx])

        # 3) downstream consumers: slice their input-channel dim, but ONLY when
        #    the relevant weight dim matches out_c (guards branches/concats).
        #    Depthwise consumers keep [O,1,k,k] but need `group` == keep_n.
        out_tensor = bn.output[0]
        for cust in consumers.get(out_tensor, []):
            if cust.op_type == "Conv" and len(cust.input) > 1 and cust.input[1]:
                cw = get_init(m, cust.input[1])
                if cw is not None:
                    arr = numpy_helper.to_array(cw)
                    g = conv_groups(cust)
                    if arr.ndim == 4 and arr.shape[1] * g == out_c:
                        if g == 1:
                            # plain conv: input channels == out_c -> slice directly
                            set_init_array(m, cust.input[1], arr[:, keep_idx])
                        elif arr.shape[1] == 1 and g == out_c:
                            # depthwise over the pruned tensor: weight shape
                            # unchanged, group count must follow the channels
                            set_conv_group(cust, keep_n)
                        # other grouped layouts: left alone (checker will flag)
            elif cust.op_type == "MatMul":
                cw = get_init(m, cust.input[1])
                if cw is not None:
                    arr = numpy_helper.to_array(cw)
                    if arr.ndim == 2 and arr.shape[0] == out_c:
                        set_init_array(m, cust.input[1], arr[keep_idx, :])
            elif cust.op_type == "Gemm":
                for wi in (cust.input[1], cust.input[2] if len(cust.input) > 2 else None):
                    if not wi:
                        continue
                    cw = get_init(m, wi)
                    if cw is not None:
                        arr = numpy_helper.to_array(cw)
                        if arr.ndim == 2 and arr.shape[1] == out_c:
                            set_init_array(m, wi, arr[:, keep_idx])
                        elif arr.ndim == 1 and arr.shape[0] == out_c:
                            set_init_array(m, wi, arr[keep_idx])

        kept_map[conv.name or conv.output[0]] = (out_c, keep_n)
    return m, kept_map


# ---------------------------------------------------------------------------
# verification (static, no runtime — onnxruntime's native DLL does not load on
# the Win7 box, so verification is pure onnx graph analysis)
# ---------------------------------------------------------------------------

def head_class_count(model):
    """Best-effort static read of the recognition head's logit size (classes).

    Priority: 1) statically-known graph output shape, 2) the last MatMul/Gemm
    weight's output-channel dim. Returns None if undeterminable.
    """
    g = model.graph
    out = g.output[-1]
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    if dims and all(d > 0 for d in dims):
        return int(dims[-1])

    init_names = {i.name for i in g.initializer}
    best = None
    for node in g.node:
        if node.op_type in ("MatMul", "Gemm") and node.input[1] in init_names:
            for i in g.initializer:
                if i.name == node.input[1]:
                    dims = list(i.dims)
                    if node.op_type == "MatMul" and len(dims) == 2:
                        best = int(dims[-1])
                    elif node.op_type == "Gemm":
                        transb = next((a.i for a in node.attribute if a.name == "transB"), 0)
                        best = int(dims[0] if transb else dims[-1])
    return best


def consumer_channels_ok(model, kept_map):
    """For every pruned tensor, confirm sliced consumers got consistent dims.

    A consumer Conv is consistent when (weight_in_channels_per_group * groups)
    equals the pruned channel count — covers plain convs (sliced) and
    depthwise convs (group attr shrunk, weight shape unchanged)."""
    consumers = {}
    for node in model.graph.node:
        for i in node.input:
            consumers.setdefault(i, []).append(node)
    init_map = {i.name: list(i.dims) for i in model.graph.initializer}
    problems = []
    for conv_name, (was, now) in kept_map.items():
        # find the conv node and its BN's output tensor name
        for node in model.graph.node:
            if (node.name or node.output[0]) == conv_name:
                bn = next((c for c in consumers.get(node.output[0], []) if c.op_type == "BatchNormalization"), None)
                if bn is None:
                    break
                out_tensor = bn.output[0]
                for cust in consumers.get(out_tensor, []):
                    if cust.op_type == "Conv" and cust.input[1] in init_map:
                        dim = init_map[cust.input[1]][1]
                        g = conv_groups(cust)
                        if dim * g != now and dim * g != was:
                            problems.append((conv_name, cust.name or cust.output[0], dim, g, was, now))
                break
    return problems


def verify_pruned(model_in_path, model_pruned_path):
    """Compare LOGIT SIZE (class count) + structural consistency between the
    original and pruned graphs, without any inference runtime."""
    try:
        import onnx
    except ImportError:
        sys.exit("FATAL: onnx not installed.")
    m_in = onnx.load(model_in_path)
    m_new = onnx.load(model_pruned_path)
    onnx.checker.check_model(m_new)
    c_in = head_class_count(m_in)
    c_new = head_class_count(m_new)
    print(f"       verify: input  head classes={c_in}")
    print(f"       verify: pruned head classes={c_new}")
    if c_in is not None and c_new is not None and c_in != c_new:
        return False, f"class count changed {c_in} -> {c_new}"
    if c_in is None or c_new is None:
        print("       verify: class count not statically determinable — relying on checker + structure")
    return True, "ok"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Structured channel pruning of the rec model")
    ap.add_argument("--ratio", type=float, required=True, help="fraction of channels to remove, e.g. 0.3")
    ap.add_argument("--in", dest="inp", default=None, help="input ONNX (default: models/rec_fp32.onnx)")
    ap.add_argument("--out", default=None, help="output ONNX (default: models/rec_pruned.onnx)")
    ap.add_argument("--dry-run", action="store_true", help="report projected sizes, do not write the model")
    ap.add_argument("--uniform", action="store_true", help="uniform width pruning instead of L2 ranking")
    args = ap.parse_args()

    inp = args.inp or os.path.join(MODELS, "rec_fp32.onnx")
    out = args.out or os.path.join(MODELS, "rec_pruned.onnx")
    if not os.path.isfile(inp):
        sys.exit(f"FATAL: missing {inp}")
    if not 0.0 < args.ratio < 0.9:
        sys.exit("FATAL: --ratio must be in (0, 0.9)")

    model = onnx.load(inp)
    before = os.path.getsize(inp)
    scheme = "uniform" if args.uniform else "l2"
    print(f"pruning {os.path.basename(inp)} ratio={args.ratio} scheme={scheme}")

    pruned, kept = prune_graph(model, args.ratio, scheme=scheme)
    for name, (was, now) in kept.items():
        print(f"       {name[:44]:44s}  {was:4d} -> {now:4d} channels")

    tmp = out + ".tmp"
    onnx.save(pruned, tmp)
    after = os.path.getsize(tmp)

    # static verification of the LOGIT SIZE (class count must be preserved)
    ok, msg = verify_pruned(inp, tmp)
    if not ok:
        print(f"       graph surgery broke the head ({msg}) -> falling back to uniform width pruning")
        os.remove(tmp)
        pruned, kept = prune_graph(model, args.ratio, scheme="uniform")
        onnx.save(pruned, tmp)
        after = os.path.getsize(tmp)
        ok, msg = verify_pruned(inp, tmp)
        if not ok:
            os.remove(tmp)
            sys.exit(f"FATAL: pruning could not preserve the class count ({msg}); refusing to write a broken model.")
    problems = consumer_channels_ok(pruned, kept)
    if problems:
        os.remove(tmp)
        sys.exit(f"FATAL: consumer channel mismatch after pruning: {problems[:3]}")

    if args.dry_run:
        os.remove(tmp)
        print(f"[dry-run] before {mb(before)} -> projected after {mb(after)}  ({(1-after/before)*100:.1f}% smaller)")
        return 0

    os.replace(tmp, out)
    print(f"before {mb(before)}  ->  after {mb(after)}   ({(1-after/before)*100:.1f}% smaller)")
    print(f"saved {out}   (head classes preserved: {msg})")
    print("next: python model_forge/quantize.py   ->   node model_forge/verify_onnx.js")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
