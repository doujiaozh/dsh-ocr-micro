# model_forge · 模型锻造 (Model Forge)

世界级中英文 OCR（**PaddleOCR PP-OCRv6 tiny**，2025 年最新代，1.5M 参数量级：
检测 Hmean 80.6 / 识别 73.5）的模型获取与压缩流水线，为 `ocr_infer` 提供
**总占用 < 5.5 MB** 的 ONNX 模型组。

平台约束（本仓库核心原则）：纯 Node.js/TypeScript + ONNX Runtime，不编译任何
C++ 原生模块，**必须跑在 Windows 7 / 8GB RAM** 上。实测结论（PE 导入表扫描 +
真机验证）：**onnxruntime-node 全部版本（1.1~1.31）都无法在 Win7 上加载**
（需要 Win8+ 的 `GetSystemTimePreciseAsFileTime`/`CreateFile2`/
`api-ms-win-core-libraryloader-l1-2-0` 等）；运行期引擎因此锁定
**onnxruntime-web（WASM）**，锻造链 Python 侧只用 **onnx + numpy**（纯 Python）。

## 产物契约（交付给 ocr_infer）

| 文件 | 角色 | 说明 |
| --- | --- | --- |
| `models/det.onnx` | 文字检测 | 权重量化 INT8（uint8 权重 + QDQ），动态 HxW |
| `models/rec.onnx` | 文字识别 | 权重量化 INT8，动态 batch/seq/width；名字对齐 |
| `models/ppocrv6_tiny_dict.txt` | 字典 | 6904 字符，与 rec 输出类别一一对应 |

**硬性预算：`det.onnx + rec.onnx` < 5,500,000 字节**（字典不计入预算）。
`verify_onnx.js` 是这道闸门的最终裁决（onnxruntime-web WASM 实跑）。

## 流水线

```
node model_forge/download.js                 # ① 取 PP-OCRv6 tiny（oar-ocr v0.7.0 多镜像优先，bcebos 兜底）+ 字典
python model_forge/paddle2onnx_convert.py    # ①' 仅 Paddle tar 兜底路径需要：Paddle → ONNX（动态形状）
python model_forge/quantize.py               # ② 纯 onnx+numpy 权重量化 INT8（per-output-channel uint8 + DequantizeLinear）
python model_forge/prune.py --ratio R        # ③ 超预算时：rec 结构化通道剪枝（可 --dry-run）
node model_forge/verify_onnx.js              # ④ 最终校验：体积 + WASM dummy forward + 输出形状
# 可选（云端 GPU）:
python model_forge/distill.py --epochs 20 --lr 3e-4 --out models/rec_distilled.onnx
```

对应根目录 npm scripts：`npm run models` / `compress` / `prune` / `distill` / `verify`。

## ① download.js — 模型获取（多镜像，任一成功即够）

按顺序尝试，逐个兜底：

1. **oar-ocr v0.7.0 GitHub release**（GreatV/oar-ocr，release-assets 重定向自动跟随）
2. **ghproxy.net / gh-proxy.com** 镜像同一 release
3. **ModelScope**（`modelscope.cn/models/greatv/oar-ocr`，国内最快）
4. **bcebos 官方 ONNX 包**（`paddle-model-ecology .../PP-OCRv6_tiny_*_onnx_infer.tar`，纯 JS `tar` 解包取 `.onnx`）
5. （最后手段）**bcebos Paddle 推理 tar** → 交给 `paddle2onnx_convert.py` 转换

字典 `ppocrv6_tiny_dict.txt` 走同一套镜像。校验：HTTP 200、Content-Length 一致、
最小体积地板线、HTML 页面拦截、SHA-256 全程记录，结尾打印汇总表。
标称大小：det 1,780,590 B / rec 4,462,639 B / dict 27,156 B。
Flags：`--force`（强制重下）、`--paddle`（强制走 Paddle tar 兜底）。

## ② quantize.py — 权重量化 INT8（纯 onnx + numpy，预算主力）

- **刻意不依赖 onnxruntime**（native DLL 在 Win7 上加载不了）。
- 对每个 Conv / MatMul / Gemm 的 **float32 权重 initializer** 做
  **per-output-channel 非对称 uint8 量化**（min/max，scale + zero-point），
  插入 `DequantizeLinear`（opset 13，支持 per-axis）把量化权重还原后喂给原算子——
  **激活值保持 fp32**，计算路径语义不变，权重精确 4x 瘦身。
- opset 不足 13 时自动提升到 13（onnxruntime-web WASM 支持）。
- 逐模型打印 before/after + 量化张量数，最后打印 TOTAL；`>= 5,500,000` 时
  **exit 1**（除非 `--allow-over`）并给出精确的 `prune.py --ratio` 建议值。
- 纯静态校验（`onnx.checker` + 残留 fp32 权重检查），无运行时依赖。

## ③ prune.py — 结构化通道剪枝（rec 专用，纯 onnx/numpy）

- 对「Conv + 紧随 BatchNormalization」配对里的 Conv 做输出通道剪枝：
  - 重要性 = 每输出通道 **L2 范数**排序，淘汰最低的 `ratio`；
  - **跳过**首个 conv（保住 3 通道输入）、depthwise/分组 conv、分类头（MatMul/Gemm）
    的输入侧 conv（通道与类别投影对齐）；
  - 图手术：切 conv 权重/偏置 → 切 BN 的 mean/var/scale/B → 切下游消费者
    （下一个 Conv 输入通道、MatMul 行 / Gemm 列），**仅当消费者相应维度等于原通道数
    时才切**（保护残差/concat 分支）。
- **校验为纯 onnx 静态分析**（无运行时）：`onnx.checker` + LOGIT SIZE（类别数前后一致）+
  消费者通道一致性；失败时自动退回**均匀宽度剪枝**并复核。
- `--dry-run` 只报体积不落盘；`--in/--out` 可链式多次剪枝。

## ④ verify_onnx.js — 预算闸门（onnxruntime-web 主路径）

- **为什么是 web 不是 node**：onnxruntime-node 任何版本在 Win7 都加载失败（见上）。
  onnxruntime-web（WASM）在 Node 26 + Win7 实测可用，是 Win7 唯一能跑的 ORT。
- 加载方式（已验证）：`require('onnxruntime-web')`（node build；失败则动态 import）+
  `ort.env.wasm.wasmPaths` 指向包内 `dist/` 的 **file:// URL** + `numThreads=1`。
- 打印两个模型体积、输入/输出名与形状（dummy forward：det 1x3x640x640、rec 1x3x48x320，
  动态维度自动适配会话声明），断言 det + rec < 5.5MB，exit 0/1。

## ⑤ distill.py — 云端知识蒸馏（⚠️ 只在 GPU 云主机上跑）

> **绝不要在 Windows 7 机器上运行**：本脚本 import torch、构建可训练 student、
> 跑很多个 epoch。

- **Student 架构由 ONNX 图自动派生**：遍历 initializers，Conv→`nn.Conv2d`、
  BatchNormalization→`nn.BatchNorm2d`、MatMul/Gemm→`nn.Linear`，权重逐张量从 ONNX
  拷贝——student 起点即 teacher 的功能克隆（`--student rec_pruned.onnx` 可蒸馏到
  剪枝后架构）。
- **Teacher**：优先冻结的 onnxruntime fp32 会话（原始蒸馏设计）；
  onnxruntime 不可导入时自动退回**冻结 torch 克隆**（同权重、无梯度）。
- **损失 = CTC（teacher 伪标签）+ 逐帧 KL（T=4）+ logits L2**。
- `--reexport-every N` 周期性把 student 重新导出为 ONNX（动态 batch/seq/width），
  导出物可直接接回 `quantize.py`。

## 环境要求

```bash
npm install                                        # 仓库根目录（workspaces：tar + onnxruntime-web 等纯 JS 依赖）
pip install -r model_forge/requirements.txt        # onnx / numpy / paddle2onnx —— 不要装 onnxruntime、不要装 torch
```

- Node >= 14；`download.js` / `verify_onnx.js` 只用内建模块 + `tar` + `onnxruntime-web`（纯 JS + WASM，无 C++ 编译）。
- Python 侧仅用于离线锻造，且**全程不依赖 onnxruntime**。

## 原理说明（为什么这套能做到 <5.5MB 且保持世界级水准）

1. **起点就是 2025 年最新轻量代**：PP-OCRv6 tiny（det ~0.5M / rec ~1M 参数，
   fp32 合计 ~6MB）——检测 Hmean 80.6、识别 73.5，mobile 级 SOTA 之一。
2. **权重量化是体积主力**：per-output-channel uint8 把 4 字节权重压到 1 字节，
   精确 4x 收益；QDQ 节点只在权重侧，激活值 fp32 → 计算路径零精度损失风险，
   且 onnxruntime-web WASM 原生支持 QDQ。预测 det+rec ≈ 1.6~2.2MB，预算极其充裕。
3. **剪枝兜底**：若量化后仍超 5.5MB，rec 的 Conv+BN 链式结构天然适合结构化宽度
   剪枝；按通道 L2 重要性保留大通道、类别数静态校验，保证图的语义不崩。
4. **蒸馏回血**：量化和剪枝的精度损失可在云端用 fp32 teacher 蒸馏回来——
   student 架构直接从 ONNX 生成、teacher 在 ORT 不可用时退化为冻结 torch 克隆，
   锻造链始终闭环；蒸馏完再量化，体积仍在预算内。
5. **字典外挂**：6904 类字符表以纯文本下发（不计入模型预算），rec 的输出类别数
   在校验/剪枝/量化三个环节都被显式追踪，避免类别错位这类隐蔽错误。
6. **Win7 现实主义**：不与 onnxruntime-node 的 Win7 不兼容硬碰硬，引擎层直接
   用 WASM（跨平台、无原生构建），把“MUST run on Windows 7”落到实处。
