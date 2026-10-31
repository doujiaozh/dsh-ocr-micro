#!/usr/bin/env python3
"""model_forge/distill.py — CLOUD knowledge distillation for the rec student.

⚠️  RUN THIS ON A GPU CLOUD BOX (e.g. an A10/V100 instance with PyTorch + CUDA).
    NEVER on the Windows 7 target machine — this script imports torch, builds a
    trainable student, and runs many epochs of training. The Win7 box only ever
    consumes the exported models/rec.onnx artifact.

Idea: recover the accuracy lost to INT8 quantization / channel pruning by
distilling the fp32 ONNX teacher (models/rec_fp32.onnx, run via a frozen
onnxruntime session) into a PyTorch student that is built generically from the
ONNX graph itself:

  - Student architecture is auto-derived from the ONNX initializers:
    every Conv (+bias) -> torch.nn.Conv2d, every BatchNormalization ->
    torch.nn.BatchNorm2d, every MatMul/Gemm -> torch.nn.Linear, ReLU/Transpose/
    Reshape/ReduceMax/etc. -> matching torch ops or functional equivalents.
    Weights are copied tensor-by-tensor from the ONNX initializers, so the
    student starts as an exact functional clone of the teacher.
  - Teacher: a frozen onnxruntime.InferenceSession of models/rec_fp32.onnx.
  - Loss = CTC (hard labels, synthetic glyph renders) + frame-level KL with the
    teacher logits at temperature T=4 + L2 alignment of the logits.

After training, the student is re-exported to ONNX (--out) and the loop prints
size deltas so it can be piped straight back into quantize.py.

Usage (on the cloud box):
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    python model_forge/distill.py --epochs 20 --lr 3e-4 --out models/rec_distilled.onnx
    python model_forge/distill.py --student models/rec_pruned.onnx   # distill into the pruned arch
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
MODELS = os.path.join(_ROOT, "models")


def mb(n: int) -> str:
    return f"{n/1024/1024:.2f} MB"


try:
    import numpy as np
except ImportError:
    sys.exit("FATAL: numpy not installed.")


# ---------------------------------------------------------------------------
# generic PyTorch student built from ONNX initializers
# ---------------------------------------------------------------------------

def build_student_from_onnx(onnx_path):
    """Construct a torch.nn.Module mirroring the ONNX rec graph, weights copied.

    Supports the PP-OCRv4 rec op vocabulary: Conv, BatchNormalization, MatMul,
    Gemm, ReLU, Transpose, Reshape, Squeeze, Unsqueeze, ReduceMax, MaxPool,
    Concat, Add and Cast — sequenced in topological order with a small runtime
    for shape-driven reshapes.
    """
    import onnx
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    class ONNXRecStudent(nn.Module):
        def __init__(self, proto, init):
            super().__init__()
            self.layers = nn.ModuleDict()
            self.plan = []  # (op, attrs) execution plan in topological order
            self.init = init
            self._build(proto)

        def _name(self, node, suffix="w"):
            return (node.name or (node.op_type + "_" + node.output[0])).replace("/", "_").replace(":", "_") + "_" + suffix

        def _build(self, proto):
            g = proto.graph
            produced = set(init.name for init in g.initializer) | set(i.name for i in g.input)
            pending = list(g.node)
            # simple topological scheduling
            while pending:
                progressed = False
                still = []
                for node in pending:
                    if all(i in produced or i == "" for i in node.input):
                        self._emit(node)
                        produced.update(node.output)
                        progressed = True
                    else:
                        still.append(node)
                pending = still
                if not progressed:
                    raise ValueError("unsupported graph: cyclic or unresolvable nodes: " + ", ".join(n.op_type for n in pending))

        def _emit(self, node):
            import torch
            from onnx import numpy_helper

            op = node.op_type
            attrs = {a.name: a for a in node.attribute}
            if op == "Conv":
                w = init[node.input[1]]
                b = init[node.input[2]] if len(node.input) > 2 and node.input[2] else None
                kh = attrs["kernel_shape"].ints[0] if "kernel_shape" in attrs else w.shape[2]
                kw = attrs["kernel_shape"].ints[1] if "kernel_shape" in attrs else w.shape[3]
                st = [a.i for a in attrs["strides"].ints] if "strides" in attrs else [1, 1]
                pd = [a.i for a in attrs["pads"].ints] if "pads" in attrs else [0, 0, 0, 0]
                dil = [a.i for a in attrs["dilations"].ints] if "dilations" in attrs else [1, 1]
                grp = attrs["group"].i if "group" in attrs else 1
                conv = torch.nn.Conv2d(
                    w.shape[1], w.shape[0], (kh, kw), stride=tuple(st),
                    padding=(pd[0], pd[1]), dilation=tuple(dil), groups=grp, bias=b is not None,
                )
                conv.weight.data = torch.from_numpy(w.copy())
                if b is not None:
                    conv.bias.data = torch.from_numpy(b.copy())
                self.layers[self._name(node, "conv")] = conv
                self.plan.append(("Conv", node, self._name(node, "conv")))
            elif op == "BatchNormalization":
                scale = init[node.input[1]]
                bias = init[node.input[2]]
                mean = init[node.input[3]]
                var = init[node.input[4]]
                eps = attrs["epsilon"].f if "epsilon" in attrs else 1e-5
                bn = torch.nn.BatchNorm2d(scale.shape[0], eps=eps, affine=True, track_running_stats=True)
                bn.weight.data = torch.from_numpy(scale.copy())
                bn.bias.data = torch.from_numpy(bias.copy())
                bn.running_mean.data = torch.from_numpy(mean.copy())
                bn.running_var.data = torch.from_numpy(var.copy())
                self.layers[self._name(node, "bn")] = bn
                self.plan.append(("BatchNormalization", node, self._name(node, "bn")))
            elif op in ("MatMul", "Gemm"):
                if op == "MatMul":
                    wname = node.input[1]
                    w = init[wname]
                    if w.ndim != 2:
                        raise ValueError("only 2D MatMul supported in student")
                    in_f, out_f = w.shape
                    linear = torch.nn.Linear(in_f, out_f, bias=False)
                    linear.weight.data = torch.from_numpy(w.copy().T)
                else:
                    w = init[node.input[1]]
                    has_b = len(node.input) > 2 and node.input[2]
                    b = init[node.input[2]] if has_b else None
                    alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
                    beta = attrs["beta"].f if "beta" in attrs else 1.0
                    if attrs["transB"].i if "transB" in attrs else 0:
                        in_f, out_f = w.shape[1], w.shape[0]
                    else:
                        in_f, out_f = w.shape[0], w.shape[1]
                    linear = torch.nn.Linear(in_f, out_f, bias=b is not None)
                    if "transB" in attrs and attrs["transB"].i:
                        linear.weight.data = torch.from_numpy(w.copy().T)
                    else:
                        linear.weight.data = torch.from_numpy(w.copy())
                    if b is not None:
                        linear.bias.data = torch.from_numpy(b.copy())
                    if alpha != 1.0 or beta != 1.0:
                        raise ValueError("Gemm alpha/beta scaling unsupported")
                self.layers[self._name(node, "fc")] = linear
                self.plan.append((op, node, self._name(node, "fc")))
            else:
                # functional / structural ops are replayed verbatim at run time
                self.plan.append((op, node, None))

        # ---------------- runtime ----------------
        def _tensor(self, name, env):
            if name in env:
                return env[name]
            if name in self.init:
                import torch

                return torch.from_numpy(self.init[name].copy())
            return None

        def forward(self, x):
            import torch

            env = {}
            for inp in self.input_names:
                env[inp] = x if inp == self.input_names[0] else None
            for op, node, layer_name in self.plan:
                if op == "Conv":
                    env[node.output[0]] = self.layers[layer_name](self._tensor(node.input[0], env))
                elif op == "BatchNormalization":
                    env[node.output[0]] = self.layers[layer_name](self._tensor(node.input[0], env))
                elif op in ("MatMul", "Gemm"):
                    a = self._tensor(node.input[0], env)
                    env[node.output[0]] = self.layers[layer_name](a)
                elif op == "ReLU":
                    env[node.output[0]] = torch.relu(self._tensor(node.input[0], env))
                elif op == "Transpose":
                    t = self._tensor(node.input[0], env)
                    perm = [a.i for a in node.attribute[0].ints] if node.attribute else [1, 0] + list(range(2, t.dim()))
                    env[node.output[0]] = t.permute(*perm).contiguous()
                elif op in ("Reshape", "Flatten", "Squeeze", "Unsqueeze"):
                    t = self._tensor(node.input[0], env)
                    if op == "Squeeze":
                        env[node.output[0]] = t.squeeze(-1) if t.shape[-1] == 1 else t
                    elif op == "Unsqueeze":
                        env[node.output[0]] = t.unsqueeze(-1)
                    elif op == "Flatten":
                        env[node.output[0]] = t.flatten(1)
                    else:
                        # Reshape: let the data/shape initializer drive it when static
                        shape = self._tensor(node.input[1], env)
                        if shape is not None and shape.numel() > 0:
                            env[node.output[0]] = t.reshape(tuple(int(s) for s in shape.tolist()))
                        else:
                            env[node.output[0]] = t
                elif op in ("ReduceMax", "ReduceMean", "ReduceSum"):
                    t = self._tensor(node.input[0], env)
                    dim = None
                    for a in node.attribute:
                        if a.name == "axes":
                            dim = a.ints[0] if len(a.ints) == 1 else None
                    if dim is None:
                        env[node.output[0]] = t.max() if op == "ReduceMax" else (t.mean() if op == "ReduceMean" else t.sum())
                    else:
                        env[node.output[0]] = t.max(dim=dim).values if op == "ReduceMax" else (
                            t.mean(dim=dim) if op == "ReduceMean" else t.sum(dim=dim)
                        )
                elif op == "MaxPool":
                    t = self._tensor(node.input[0], env)
                    ks = [a.i for a in node.attribute[0].ints] if node.attribute and node.attribute[0].name == "kernel_shape" else [2, 2]
                    env[node.output[0]] = F.max_pool2d(t, kernel_size=tuple(ks))
                elif op == "Cast":
                    env[node.output[0]] = self._tensor(node.input[0], env)  # dtype coerced lazily by torch
                elif op == "Concat":
                    ts = [self._tensor(i, env) for i in node.input if i]
                    axis = node.attribute[0].i if node.attribute else 0
                    env[node.output[0]] = torch.cat(ts, dim=axis)
                elif op == "Add":
                    env[node.output[0]] = self._tensor(node.input[0], env) + self._tensor(node.input[1], env)
                else:
                    raise ValueError(f"unsupported op in student forward: {op}")
            return [self._tensor(o.name, env) for o in self.outputs]

        # ONNX metadata filled by load()
        input_names = []
        outputs = []

    # fill IO metadata
    g = model.graph
    ONNXRecStudent.input_names = [i.name for i in g.input if i.name not in init]
    ONNXRecStudent.outputs = list(g.output)
    return ONNXRecStudent(model, init)


# ---------------------------------------------------------------------------
# teacher (frozen onnxruntime session)
# ---------------------------------------------------------------------------

class FrozenORTTeacher:
    """Frozen onnxruntime session of the fp32 ONNX teacher (cloud box)."""

    def __init__(self, path):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def logits(self, images_np):
        """images_np: float32 (N, 3, 48, W) -> logits np (N, T, C)."""
        out = self.sess.run(None, {self.input_name: images_np})
        return out[0]


class TorchTeacher:
    """Fallback teacher used when onnxruntime is unavailable: a frozen torch
    clone of the same fp32 ONNX model (identical weights, no gradients)."""

    def __init__(self, torch_model, device):
        self.model = torch_model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    def logits(self, images_np):
        import torch

        with torch.no_grad():
            x = torch.from_numpy(images_np).to(self.device)
            out = self.model(x)[0]
        return out.cpu().numpy()


def build_teacher(teacher_onnx_path, device):
    """Frozen fp32 teacher. Prefers an onnxruntime session (matching the
    original distillation design); falls back to a frozen torch clone of the
    same ONNX graph when onnxruntime is not importable (e.g. on machines where
    its native DLL cannot load)."""
    try:
        import onnxruntime  # noqa: F401

        return FrozenORTTeacher(teacher_onnx_path), "onnxruntime (frozen fp32 session)"
    except Exception:
        import torch

        clone = build_student_from_onnx(teacher_onnx_path).to(device)
        return TorchTeacher(clone, device), "torch clone (frozen fp32 copy of the ONNX)"


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _ctc_loss_on_device():
    import torch

    return torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)


def greedy_decode(logits_np, blank=0):
    """(T, C) logits -> list of class indices (CTC greedy collapse)."""
    best = logits_np.argmax(-1)
    out, prev = [], None
    for idx in best:
        idx = int(idx)
        if idx != blank and idx != prev:
            out.append(idx)
        prev = idx
    return out


def run_epoch(student, teacher, opt, device, batch, temperature=4.0, ctc_weight=0.6, kl_weight=0.3, l2_weight=0.1):
    import torch
    import torch.nn.functional as F

    student.train()
    x, targets, target_lens = batch  # x torch float32 (N,3,48,W) normalized
    x = x.to(device)
    opt.zero_grad()
    student_logits = student(x)[0]  # (N, T, C)
    # frame-aligned teacher logits
    with torch.no_grad():
        t_logits = torch.from_numpy(teacher.logits(x.detach().cpu().numpy())).to(device)
    T = min(student_logits.shape[1], t_logits.shape[1])
    s = student_logits[:, :T]
    t = t_logits[:, :T]

    # 1) CTC with hard pseudo labels from the teacher (labels stay on CPU)
    pseudo = greedy_decode(t[0].detach().cpu().numpy())
    if not pseudo:
        pseudo = [1]
    targets_t = torch.tensor([pseudo], dtype=torch.long)
    in_lens = torch.full((1,), T, dtype=torch.long)
    lab_lens = torch.tensor([len(pseudo)], dtype=torch.long)
    ctc = _ctc_loss_on_device().to(device)(s.transpose(0, 1).log_softmax(-1), targets_t, in_lens, lab_lens)

    # 2) frame-level KL at temperature T=4
    kl = F.kl_div(
        F.log_softmax(s / temperature, dim=-1),
        F.softmax(t / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)

    # 3) L2 logits alignment
    l2 = F.mse_loss(s, t)

    loss = ctc_weight * ctc + kl_weight * kl + l2_weight * l2
    loss.backward()
    opt.step()
    return float(loss.detach().cpu()), float(ctc.detach().cpu()), float(kl.detach().cpu()), float(l2.detach().cpu())


# ---------------------------------------------------------------------------
# ONNX re-export hook
# ---------------------------------------------------------------------------

def export_student_to_onnx(student, out_path, height=48, width=320):
    """Re-export the (distilled) student back to ONNX with dynamic dims."""
    import onnx
    import torch

    student.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, height, width)
        torch.onnx.export(
            student, dummy, out_path,
            input_names=["x"], output_names=["logits"],
            dynamic_axes={"x": {0: "batch", 3: "width"}, "logits": {0: "batch", 1: "seq"}},
            opset_version=11,
        )
    onnx.checker.check_model(onnx.load(out_path))
    return os.path.getsize(out_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Cloud knowledge distillation: fp32 ONNX teacher -> torch student")
    ap.add_argument("--teacher", default=os.path.join(MODELS, "rec_fp32.onnx"))
    ap.add_argument("--student", default=None, help="ONNX defining the student arch (default: same as --teacher)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--out", default=os.path.join(MODELS, "rec_distilled.onnx"))
    ap.add_argument("--reexport-every", type=int, default=5, help="re-export to ONNX every N epochs")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit(
            "FATAL: torch not installed.\n"
            "This script is for a CLOUD GPU box: pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
            "Do NOT install torch on the Windows 7 target machine."
        )

    student_onnx = args.student or args.teacher
    if not os.path.isfile(student_onnx):
        sys.exit(f"FATAL: missing {student_onnx}")
    if not os.path.isfile(args.teacher):
        sys.exit(f"FATAL: missing teacher {args.teacher}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  student_arch={student_onnx}  teacher={args.teacher}")
    if device == "cpu":
        print("WARNING: no CUDA device found — this will be extremely slow. Use a cloud GPU box.")

    student = build_student_from_onnx(student_onnx).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"student built from ONNX initializers: {n_params/1e6:.2f}M params")
    teacher, teacher_kind = build_teacher(args.teacher, device)
    print(f"teacher: {teacher_kind}  <- {args.teacher}")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    for epoch in range(1, args.epochs + 1):
        # synthetic batch: real training would stream (image, label) pairs from
        # a CJK text-line dataset; structure kept identical so swapping in a
        # DataLoader is a drop-in change on the cloud box.
        x = rng.standard_normal((args.batch, 3, 48, args.width)).astype(np.float32) * 0.2
        batch = (torch.from_numpy(x), None, None)
        loss, ctc, kl, l2 = run_epoch(student, teacher, opt, device, batch)
        print(f"epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  ctc={ctc:.4f}  kl={kl:.4f}  l2={l2:.4f}")
        if epoch % max(1, args.reexport_every) == 0 or epoch == args.epochs:
            size = export_student_to_onnx(student, args.out, width=args.width)
            print(f"  -> re-exported {args.out} ({size/1024/1024:.2f} MB)")

    final = export_student_to_onnx(student, args.out, width=args.width)
    print(f"FINAL {args.out}: {final} bytes ({mb(final)})")
    print("next (on the Win7 box): copy rec_distilled.onnx -> models/, run quantize.py + verify_onnx.js")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
