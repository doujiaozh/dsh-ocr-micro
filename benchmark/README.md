# @dsh-ocr/benchmark — 「世界级吊打测试」

DSH-OCR-Micro 的基准评测套件：公开数据集获取、检测评测（IoU/F1）、识别评测
（CER/NED/Accuracy）、延迟与体积评测（p50/p95 vs 5.5MB 预算）、已发表基线对比。

**设计原则（硬约束）**

- 纯 CommonJS `.js`，**零 npm 依赖**：仅用 Node 内置模块 + 可选 `python3` 子进程；
- 数据集无法下载时自动回退合成数据，`run_all.js` **始终以 exit 0 结束**并产出报告；
- 所有脚本可从仓库根目录或 `benchmark/` 目录直接运行；
- 对比表里**已发表数值**与**本机实测数值**严格区分标注，不编造任何未发表指标。

## 文件清单

| 文件 | 作用 |
|---|---|
| `package.json` | `@dsh-ocr/benchmark`（private，无依赖） |
| `lib/util.js` | 共享工具：argv、AABB/IoU、Levenshtein、分位数、HTTP HEAD/下载、python 查找、zip/tar 解包（python stdlib） |
| `datasets.js` | 数据集注册表 / 下载器 / 加载器（`listDatasets()` / `loadDataset(name)` / `--download` / `--synth`） |
| `gen_synth.py` | PIL 合成数据生成器（中英文混排、±8° 倾斜、高斯噪声、渐变背景、运动模糊） |
| `run_det.js` | 逐图子进程调用 CLI：`node ../dist/ocr_infer/src/cli.js <img> --json`，输出 `out/<dataset>.pred.json` |
| `eval_det.js` | 检测评测：AABB IoU 贪心匹配（IoU=0.5 可配置）、`###` don't-care 规则、P/R/F1 |
| `eval_rec.js` | 识别评测：CER% / 精确匹配 Accuracy% / NED（检测匹配行上计算） |
| `bench_speed.js` | 延迟：N=30 warm + 5 cold（fresh subprocess），阶段耗时 p50/p95，模型字节数 vs 5.5MB 徽章 |
| `compare.js` | 硬编码已发表基线行（明确标注）+ 本机实测行；`--tesseract` / `--paddle` 本地引擎钩子 |
| `run_all.js` | 总编排：verify_onnx → 数据集 → run_det → eval_det/eval_rec → bench_speed → compare → `out/REPORT.md` |
| `test/stub_cli.js` | **契约桩**（仅自测用）：模拟 CLI 契约输出，用于在 `dist/` 构建前验证整套评测管线 |
| `.gitignore` | 忽略 `data/`、`out/` |

## 快速开始

```bash
# 仓库根目录（推荐入口，根 package.json 已配置）
npm run bench                                   # = node benchmark/run_all.js
npm run bench:speed                             # = node benchmark/bench_speed.js

# 或在 benchmark/ 目录下直接运行
node run_all.js                                  # 完整流水线
node run_all.js --dataset synth --limit 50       # 仅合成集、前 50 张
node run_all.js --no-net                         # 跳过网络探测
node run_all.js --tesseract --paddle             # 启用本地引擎对比行（未安装则自动跳过）
node datasets.js --list                          # 查看数据集在盘状态
node datasets.js --download icdar2015            # 探测镜像并下载
node datasets.js --synth --n 500                 # 生成更多合成数据
```

常用参数：`--dataset auto|icdar2015|icdar2013|synth`、`--limit N`、`--n 200`、
`--warm 30 --cold 5`、`--mirror URL`（可逗号分隔）、`--no-net`、`--iou 0.5`、`--case-sensitive true`。

环境变量：`DSH_BENCH_MIRRORS="url1|url2"`（追加镜像，优先探测）、
`DSH_BENCH_CLI=<path>`（覆盖 CLI 路径，自测专用）。

## 消费的 CLI 契约

`node dist/ocr_infer/src/cli.js <image> --json` → stdout 单行 JSON：

```json
{"ok":true,"version":"1.0.0","text":"...",
 "results":[{"text":"...","score":0.97,"box":[[x,y],[x,y],[x,y],[x,y]]}],
 "timing":{"preprocessMs":1,"detMs":2,"recMs":3,"postMs":0.5,"totalMs":7},
 "models":{"detBytes":N,"recBytes":N}}
```

错误：非零退出码 + stderr 信息（`run_det.js` 逐图记录，不中断流水线）。

## 数据集

| 名称 | 说明 |
|---|---|
| `icdar2015` | ICDAR 2015 Incidental Scene Text（Challenge 4 训练集），GT 格式 `x1,y1,...,y4,word`，`###` = don't-care |
| `icdar2013` | ICDAR 2013 Focused Scene Text（Challenge 2），GT 支持 `x1 y1 x2 y2 word` 与**纯 word 每行**两种格式 |
| `synth` | `gen_synth.py` 生成的合成本地集（回退 + 回归用，始终可用） |

- `rr.cvc.uab.es` 需注册，故默认探测：官方 URL（标记为 gated）+ 公共 Hugging Face
  社区镜像（`hf-api` 类型，自动列出仓库内压缩包）+ `--mirror` / `DSH_BENCH_MIRRORS`
  提供的任意镜像。全部 HEAD 探测，失败即跳过并输出原因；支持 zip / tar / tar.gz，
  用 python stdlib 解包。
- 镜像仅返回**字节数与扩展名达标**的压缩包，清单页（如 HCIILAB 的 README）会被
  探测到但跳过。

### 可比性声明（重要）

合成数据（`synth`）用于**管线自测与回归**，其数值**不代表**公开数据集精度，也
不与已发表 ICDAR2015 数值处于同一评测协议。`REPORT.md` 会自动标注每个数据集的
来源（real / synthetic），只有真实数据集下载成功时才输出可比数值。

## 评测协议

- **检测**：GT 四边形与预测 4 点框各自取 AABB，贪心按 IoU 降序匹配（阈值默认 0.5）。
  GT 为 `###` 的 don't-care 区域：未匹配不计 FN；匹配到的预测为 neutral（不计 TP/FP）。
  输出 Precision / Recall / F1（=Hmean）。
- **识别**：仅统计检测匹配行。CER = Σ编辑距离 / ΣGT 长度；Accuracy = 整行完全相等
  比例（默认大小写不敏感、空白归一）；NED = 编辑距离 / max(1, 双方最长长度)。
- **延迟**：warm = 同一图片连续 N 次 CLI 子进程调用（OS 缓存已热）；cold = 每次用
  **未见过的不同图片** + 全新进程，报告 `wall − inProc` 进程启动开销。
- **体积**：`models.detBytes + recBytes` 与 5.5MB 预算对比，输出徽章
  （OK under 5.5MB / WARN / OVER）。

## 对比表（已发表 vs 实测）

已发表基线行（来源已在表中逐行列出）：

| 引擎 | 体积 | 备注 |
|---|---|---|
| Tesseract 5.3 LSTM (chi_sim+eng) | 64.69 MB | tessdata：eng 22.38MB + chi_sim 42.31MB；未发表 ICDAR2015 指标 |
| PaddleOCR PP-OCRv4 server | 198 MB | models_list.md；PaddleX 3.2 基准 det Hmean 69.2% / rec acc 85.19%（**非 ICDAR2015**） |
| PaddleOCR PP-OCRv4 mobile | 14.7 MB | models_list.md；PaddleX 3.2 基准 det Hmean 63.8% / rec acc 78.74%（**非 ICDAR2015**） |
| RapidOCR (rapidocr-onnxruntime 1.4.4) | 14.9 MB | PyPI wheel 体积（内置 PP-OCRv4 mobile ONNX） |
| EasyOCR (CRAFT+CRNN, ch_sim+en) | 108.6 MB | HF qualcomm/EasyOCR：det 79.2MB + rec 14.7MB/语言 |

本地钩子：`--tesseract`（调本地 `tesseract ... tsv`，词级框）、`--paddle`（调本地
`paddleocr` pip 包，兼容 `ocr()` 与 `predict()` 两代 API）。两者在**同一批图片、
同一套 IoU 匹配**下评测为 `measured-here (local hook)` 行；引擎未安装则自动跳过并
在表下注明。

## 在 dist/ 构建前自测整套管线

```bash
# Windows
set DSH_BENCH_CLI=%CD%\benchmark\test\stub_cli.js
set DSH_STUB_GT=%CD%\benchmark\data\synth\gt.json
set DSH_STUB_MODE=degrade     # 或 perfect
node benchmark\run_all.js --limit 8
```

`perfect` 模式应得到 F1=100% / CER=0%（验证评测数学）；`degrade` 模式得到非平凡
数值（验证降级路径）。真实 CLI 就绪后无需任何代码改动，去掉环境变量即可。

## 输出（`benchmark/out/`）

| 文件 | 内容 |
|---|---|
| `<dataset>.pred.json` | 逐图 CLI 原始预测 + 耗时 + models 字节 |
| `<dataset>.det.json` | 检测报告（P/R/F1、逐图、匹配对、ignore 统计） |
| `<dataset>.rec.json` | 识别报告（CER/Accuracy/NED、逐对结果） |
| `speed_report.json` | warm/cold 延迟分位数、阶段耗时、模型体积徽章 |
| `compare.json` | 完整对比表（含 skip 原因） |
| `REPORT.md` | 「5.5MB 下的世界级精度」汇总报告 |

## 兼容性

Windows 7 / 8GB RAM 目标：无 C++ 原生模块、无 npm 依赖；Node ≥14（用 ES2019 语法）；
python ≥3.6 + Pillow（缺失时 `pip install pillow` 即可；python 完全不可用时仅跳过
下载与合成，其余管线仍可用已下载数据运行）。
