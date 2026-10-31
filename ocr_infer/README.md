# ocr_infer — 极速推理中枢

端到端链路：**图片 → 传统 CV 增强 → DB 文本检测 → CRNN/SVTR 识别 → CTC 解码 → 后处理纠错 → 文本**。

## 架构

| 文件 | 职责 |
|---|---|
| `session.ts` | onnxruntime-node 优先、onnxruntime-web 回退的会话缓存（预编译、零 C++ 构建） |
| `geometry.ts` | 纯 JS 几何内核：双线性重采样、单应性求解、透视裁剪、凸包、旋转卡爪最小外接矩形、unclip |
| `det.ts` | DB 后处理：阈值图二值化 → 八连通域 → 最小外接矩形 → 计分过滤 → 扩边 → 阅读顺序排序 |
| `rec.ts` | CRNN/SVTR 识别：48×W 归一化裁剪、CTC 贪心解码（重复折叠 + 去空白）、字典解码、行置信度 |
| `postprocess.ts` | 纠错：数字串内 O/0、l/1、S/5 修复；CJK 半角标点全角化；可选形近字修复 |
| `pipeline.ts` | 总装：decode → enhance(auto) → detect → recognize → post-correct |
| `cli.ts` | 命令行，`--json` 输出单行 JSON（benchmark 消费的契约） |
| `smoke.ts` | 纯 JS 单元测试 + 端到端冒烟（缺模型时自动跳过 e2e） |

## 使用

```bash
npm run build
node dist/ocr_infer/src/cli.js receipt.jpg --json
```

```ts
import { runOcr } from 'dsh-ocr-micro/ocr_infer';
const res = await runOcr('receipt.jpg', { enhance: 'auto', cnConfusions: false });
console.log(res.text);          // 全文（行连接）
console.log(res.results);       // 每行：{ text, score, box: [[x,y]×4] }
```

## 关键参数

- `detThresh=0.3`：DB 概率图二值化阈值；`boxScoreThresh=0.6`：框置信度过滤。
- `unclipRatio=1.5`：DB 框扩边系数（笔画外扩，提高裁剪完整性）。
- `recMaxWidth=320`：识别图宽度上限；`dropScore=0.5`：低置信行丢弃。
- `recNormalize='half255'`：Paddle PP-OCR 系列 rec 预处理约定 `(x/255-0.5)/0.5`。

## Win7 友好

全部为 CommonJS + ES2019，仅依赖 `onnxruntime-node` 预编译二进制，线程池上限 4，单张 720p 图片常驻内存 < 60MB。
