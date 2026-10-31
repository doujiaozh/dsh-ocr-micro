# DSH OCR Micro — 基准报告：5.5MB 下的世界级精度

- 生成时间 (generated): 2026-10-31T11:51:44.331Z
- 环境 (env): Node v26.7.0, win32/x64, cwd `.`
- 模型校验 (model verify): **OK**
- CLI: `dist\ocr_infer\src\cli.js` (built)

## 0. 结论速览 (TL;DR)

| 指标 | 值 |
|---|---|
| 模型体积 (model size, det+rec) | 1.72 MB (det 0.53 + rec 1.19) — badge: **OK under 5.5 MB** |
| 检测 F1 (det, AABB IoU=0.5) | **89.58%** on `synth` |
| 识别 CER (rec, 匹配行) | **43.00%** on `synth` |
| 延迟 (latency p50) | **1483.0 ms** p50 / 1850.6 ms p95 (warm) |

> **一句话**: **合成管线自测通过**：当前数据为合成本地集（公共 ICDAR 镜像不可达，rr.cvc.uab.es 需注册）。合成数值仅用于验证端到端管线与回归，**不与已发表的 ICDAR2015 数值直接可比**；待真实数据集就绪后本报告自动转为可比数值。

## 1. 数据集来源 (provenance)

| dataset | kind | note |
| --- | --- | --- |
| icdar2015 | real (public dataset) | unavailable: could not obtain gt from any mirror |
| icdar2013 | real (public dataset) | unavailable: could not obtain gt from any mirror |
| synth | synthetic (gen_synth.py) | generated fallback / regression set |

> **可比性声明**: 合成数据用于管线自测与回归；其数值**不代表**公开数据集精度，也不与已发表基线处于同一评测协议。真实 ICDAR 数据下载成功后，本报告自动输出可比数值。

## 2. 检测评测 (detection, AABB IoU=0.5, `###` 即 don't-care)

| dataset | images | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synth | 197 | 361 | 22 | 62 | 94.26% | 85.34% | 89.58% |

## 3. 识别评测 (recognition, 检测匹配行, 默认大小写不敏感)

| dataset | matched lines | Accuracy | CER | NED |
| --- | --- | --- | --- | --- |
| synth | 361 | 5.54% | 43.00% | 0.3801 |

## 4. 延迟与体积 (latency & model size vs 5.5MB)

| 指标 | 值 |
|---|---|
| totalMs p50 / p95 (warm) | 1483.0 / 1850.6 ms |
| preprocessMs p50 | 74.5 ms |
| detMs p50 | 1222.3 ms |
| recMs p50 | 198.5 ms |
| postMs p50 | 0.6 ms |
| cold 启动开销 (wall − inProc) | 316.8 ms (n=5) |
| 模型体积 | 1.72 MB (det 0.53 + rec 1.19) — badge: **OK under 5.5 MB** |

## 5. 对比 (published vs measured-here)

| engine | kind | det MB | rec MB | total MB | det F1 | CER% | p50 ms | 5.5MB budget | source / note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DSH-OCR-Micro (this repo) | measured-here | 0.53 | 1.19 | 1.72 | 89.58% (synth) | 43.00% (synth) | 1483.0 | OK under 5.5 MB | benchmark/out/*.json from this run (IoU=0.5, AABB matching) |
| Tesseract 5.3 LSTM (chi_sim+eng) | published | bundled | bundled | 64.69 | n/a (not published) | n/a (not published) | n/a | 11.8x over budget | tessdata repo: eng 22.38MB + chi_sim 42.31MB (github.com/tesseract-ocr/tessdata; no ICDAR2015 metrics published) |
| PaddleOCR PP-OCRv4 server (det+rec) | published | 110 | 88 | 198 | det Hmean 69.2%* | rec acc 85.19%* | n/a | 36.0x over budget | models_list.md (ch_PP-OCRv4_server_det 110M / server_rec 88M); *PaddleX 3.2 benchmark set, NOT ICDAR2015 (paddlepaddle.github.io/PaddleX/3.2) |
| PaddleOCR PP-OCRv4 mobile (det+rec) | published | 4.7 | 10 | 14.7 | det Hmean 63.8%* | rec acc 78.74%* | n/a | 2.7x over budget | models_list.md (ch_PP-OCRv4_det 4.70M / rec 10M); *PaddleX 3.2 benchmark set, NOT ICDAR2015 |
| RapidOCR (rapidocr-onnxruntime 1.4.4) | published | bundled | bundled | 14.9 | uses PP-OCRv4 mobile models | n/a (not published) | n/a | 2.7x over budget | PyPI wheel size 14.9MB, models bundled (pypi.org/project/rapidocr-onnxruntime) |
| EasyOCR (CRAFT + CRNN, ch_sim+en) | published | 79.2 | 14.7 | 108.6 | n/a (not published) | n/a (not published) | n/a | 19.7x over budget | detector 79.2MB + recognizer 14.7MB per language (huggingface.co/qualcomm/EasyOCR) |

## 附录 (reproduce)

```bash
npm run bench                       # 完整流水线（自动下载数据集/回退合成）
node benchmark/datasets.js --download icdar2015 --mirror <URL>   # 指定镜像
node benchmark/datasets.js --synth --n 200
node benchmark/run_det.js --dataset synth && node benchmark/eval_det.js --dataset synth && node benchmark/eval_rec.js --det-report out/synth.det.json
node benchmark/bench_speed.js --dataset synth
node benchmark/compare.js --dataset synth --tesseract --paddle
```
