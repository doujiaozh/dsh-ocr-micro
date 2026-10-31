# cv_preprocess — 传统 CV 暴力增强

纯 JS（pngjs + jpeg-js）实现的 OCR 前处理库，零原生编译，Win7 可跑。体积受限的模型识别烂图，靠这里把烂图洗成好图。

## 能力

| 类别 | 算法 |
|---|---|
| 解码 | PNG / JPEG 解码编码（pngjs / jpeg-js） |
| 灰度 | BT.601 luma、积分图（summed-area table）、盒式滤波 |
| 二值化 | Otsu 全局、自适应均值、**Sauvola**（默认窗口 25 / k=0.2 / R=128，抗阴影渐变） |
| 去噪 | 中值滤波、CLAHE（8x8 tile / clip 2.0，直方图剪切重分配 + 双线性 LUT 混合） |
| 边缘 | Sobel（gx/gy/模长）、Canny（非极大抑制 + 双阈值滞后连通） |
| 几何 | 双线性旋转（RGBA / 灰度）、投影方差法倾斜检测（[-10°,10°] 步长 0.5°，搜索图降采样到 ≤400px）、deskew（\|angle\|<0.3° 跳过，角度取整 0.1°） |
| 超分 | Catmull-Rom 双三次整数倍放大 + USM 锐化（superResolve） |
| 总装 | `enhance()`：短边 <320 时按 2x/4x 幂次自动放大 → deskew → 可选 CLAHE → USM |

## 设计决策

- **默认不强制二值化**：DB 检测器在自然图上训练，二值化会破坏其概率分布；扫描件可显式 `binarize: 'sauvola'`。
- **倾斜搜索在降采样图上做**：全尺寸 41 次旋转约慢 40 倍且无精度收益。
- **双三次而非双线性放大**：文本笔画的高频边缘对插值质量敏感。

## 冒烟测试

```bash
npm run build && node dist/cv_preprocess/src/smoke.js
```

覆盖：PNG 往返像素一致且编码确定性、6° 倾斜页 detectSkew 误差 <0.6°、双三次输出 = scale×输入、Sauvola 输出域 {0,255}、deskew 反向旋转、enhance 自动缩放元信息。
