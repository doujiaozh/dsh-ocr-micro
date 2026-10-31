#!/usr/bin/env python3
"""model_forge/paddle2onnx_convert.py — Paddle inference model -> ONNX converter.

Used only by the FALLBACK path of download.js: when the pre-converted ONNX
mirrors are unreachable, we download the official PaddlePaddle PP-OCRv4 mobile
inference tars (model_forge/cache/{det,rec}/) and convert them here.

Outputs (repo-relative):
    models/det_fp32.onnx   detection model, dynamic H x W (NCHW float32)
    models/rec_fp32.onnx   recognition model, dynamic batch / seq / width

Usage:
    python model_forge/paddle2onnx_convert.py
    python model_forge/paddle2onnx_convert.py --models-dir models --cache-dir model_forge/cache
"""

import argparse
import os
import sys

try:
    import paddle2onnx
except ImportError:
    sys.exit("FATAL: paddle2onnx is not installed. Run: pip install -r model_forge/requirements.txt")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def find_paddle_files(cache_dir: os.PathLike, kind: str):
    base = os.path.join(cache_dir, kind)
    if not os.path.isdir(base):
        sys.exit(f"FATAL: missing extracted paddle model dir: {base} (run node model_forge/download.js first)")
    model = params = None
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f == "inference.pdmodel":
                model = os.path.join(root, f)
            elif f == "inference.pdiparams":
                params = os.path.join(root, f)
    if not model or not params:
        sys.exit(f"FATAL: inference.pdmodel / inference.pdiparams not found under {base}")
    return model, params


def convert(kind: str, model: str, params: str, out_path: str, opset: int) -> None:
    # PP-OCRv4 mobile inference graphs are exported with dynamic input shapes:
    #   det: [?, 3, ?, ?]   rec: [?, 3, 48, ?]
    # paddle2onnx.export keeps the Paddle shape variables as ONNX dynamic dims.
    print(f"[{kind}] paddle -> onnx (opset {opset})")
    print(f"       in : {model}")
    paddle2onnx.export(model_file=model, params_file=params, save_file=out_path, opset_version=opset)
    if not os.path.isfile(out_path):
        sys.exit(f"FATAL: export produced no file: {out_path}")
    size = os.path.getsize(out_path)
    print(f"       out: {out_path}  {size/1024/1024:.2f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert PP-OCRv4 paddle inference models to ONNX")
    ap.add_argument("--models-dir", default=os.path.join(_ROOT, "models"))
    ap.add_argument("--cache-dir", default=os.path.join(_HERE, "cache"))
    ap.add_argument("--opset", type=int, default=11)
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    specs = [
        ("det", "det_fp32.onnx"),
        ("rec", "rec_fp32.onnx"),
    ]
    total = 0
    from onnx import load  # deferred so a missing `onnx` is only fatal here

    for kind, fname in specs:
        target = os.path.join(args.models_dir, fname)
        model, params = find_paddle_files(args.cache_dir, kind)
        convert(kind, model, params, target, args.opset)
        onnx_model = load(target)
        n_nodes = len(onnx_model.graph.node)
        n_init = len(onnx_model.graph.initializer)
        n_inputs = [(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in onnx_model.graph.input]
        size = os.path.getsize(target)
        total += size
        print(f"       nodes={n_nodes}  initializers={n_init}  inputs={n_inputs}")
    print()
    print(f"TOTAL fp32 ONNX: {total} bytes ({total/1024/1024:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
