#!/usr/bin/env python3
"""benchmark/gen_synth.py — synthetic CJK+EN scene-text generator (PIL).

Fallback + always-available dataset for the benchmark suite. Renders mixed
Chinese/English text lines with gradient backgrounds, random skew (+/-8 deg),
gaussian noise and motion blur (PIL ImageFilter), then writes:

    <out>/img/NNNN.jpg     JPEG images
    <out>/gt.json          [{img, w, h, words:[{text, box:[[x,y]x4]}]}]

Ground-truth boxes are the axis-aligned bounding boxes of the *rotated* text
(alpha-channel bbox), matching the AABB convention used by eval_det.js.

Usage: python gen_synth.py [--n 200] [--out DIR] [--seed 1234]
"""

import argparse
import json
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ------------------------------------------------------------------ fonts

FONT_DIRS = [
    r"C:\Windows\Fonts",
    "/usr/share/fonts",
    "/usr/share/fonts/truetype",
    os.path.expanduser("~/.fonts"),
]

# Preferred CJK fonts (Windows 7/10/11 ship these; first loadable wins).
CJK_CANDIDATES = [
    "msyh.ttc", "msyh.ttf",
    "msyhbd.ttc",
    "simhei.ttf",
    "simsun.ttc", "simsun.ttf",
    "simkai.ttf",
    "Deng.ttf", "Dengb.ttf", "Dengl.ttf",
    "msyhl.ttc",
    "NotoSansCJK-Regular.ttc", "NotoSansCJK-Regular.otf",
    "NotoSansSC-Regular.otf", "NotoSansSC-Regular.ttf",
    "wqy-microhei.ttc", "wqy-zenhei.ttc",
    "SourceHanSansSC-Regular.otf",
]

LATIN_FALLBACK = ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]


def list_font_files():
    seen = []
    for d in FONT_DIRS:
        if not os.path.isdir(d):
            continue
        try:
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                if os.path.isfile(p) and name.lower().endswith((".ttf", ".ttc", ".otf")):
                    seen.append(p)
        except OSError:
            pass
    return seen


def pick_font(size):
    """Pick a working CJK font at `size`; fall back to any loadable font."""
    files = list_font_files()
    # 1) preferred CJK candidates by basename
    for cand in CJK_CANDIDATES:
        for f in files:
            if os.path.basename(f).lower() == cand.lower():
                try:
                    return ImageFont.truetype(f, size), os.path.basename(f)
                except Exception:
                    continue
    # 2) any font whose name hints at CJK
    for f in files:
        b = os.path.basename(f).lower()
        if any(k in b for k in ("cjk", "hei", "song", "kai", "yahei", "ming", "gothic", "noto", "wqy")):
            try:
                return ImageFont.truetype(f, size), os.path.basename(f)
            except Exception:
                continue
    # 3) latin fallback
    for cand in LATIN_FALLBACK:
        for f in files:
            if os.path.basename(f).lower() == cand.lower():
                try:
                    return ImageFont.truetype(f, size), os.path.basename(f)
                except Exception:
                    continue
    # 4) first loadable file at all
    for f in files:
        try:
            return ImageFont.truetype(f, size), os.path.basename(f)
        except Exception:
            continue
    return ImageFont.load_default(), "builtin"


# ------------------------------------------------------------------ content

CJK_CHARS = (
    "中文混合文字识别模型训练数据测试图像场景文本检测识别准确率"
    "深度学习神经网络轻量级推理引擎量化压缩速度精度对比基准评测"
    "人工智能机器学习计算机视觉自然语言处理大数据云计算边缘设备"
    "北京上海深圳杭州广州成都西安武汉南京苏州重庆天津香港台湾"
    "汉字母数字符号标点符号空格换行段落文章新闻广告招牌路牌菜单"
)

EN_WORDS = [
    "DeepSeek", "OCR", "Vision", "Model", "Cloud", "Server", "Edge", "Neural",
    "Network", "Quantize", "Tensor", "Inference", "Micro", "Tiny", "Fast",
    "Hello", "World", "OpenSource", "Engine", "Scene", "Image", "Data",
    "Attention", "Transformer", "ONNX", "Runtime", " WASM", "Benchmark",
    "Precision", "Recall", "Latency", "P50", "P95", "Mobile", "Desktop",
    "Windows", "Linux", "Alpha", "Beta", "Gamma", "Delta", "Sample", "Test",
]


def rand_line(rng):
    """A mixed CJK/EN line: e.g. '识别引擎 DeepSeek' + 'OCR Speed 2026'."""
    parts = []
    n_cjk = rng.randint(1, 3)
    n_en = rng.randint(1, 3)
    for _ in range(n_cjk):
        k = rng.randint(2, 6)
        parts.append("".join(rng.choice(CJK_CHARS) for _ in range(k)))
    for _ in range(n_en):
        parts.append(rng.choice(EN_WORDS).strip())
    if rng.random() < 0.35:
        parts.append(str(rng.randint(1990, 2026)))
    rng.shuffle(parts)
    return " ".join(parts)


# ------------------------------------------------------------------ rendering

def gradient_bg(w, h, rng):
    """Low-saturation diagonal gradient background."""
    c1 = tuple(rng.randint(30, 150) for _ in range(3))
    c2 = tuple(rng.randint(120, 235) for _ in range(3))
    img = Image.new("RGB", (w, h), c1)
    # cheap vertical/diagonal gradient: resize a 2x2 image up
    corner = Image.new("RGB", (2, 2))
    corner.putpixel((0, 0), c1)
    corner.putpixel((1, 0), tuple((a + b) // 2 for a, b in zip(c1, c2)))
    corner.putpixel((0, 1), tuple((a + b) // 2 for a, b in zip(c1, c2)))
    corner.putpixel((1, 1), c2)
    img = corner.resize((w, h), Image.BILINEAR)
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(4, 14)))
    return img


def add_gaussian_noise(img, rng, sigma):
    """In-place additive gaussian noise via a noise layer."""
    if sigma <= 0:
        return img
    noise = Image.effect_noise(img.size, sigma * 6).convert("L")
    import PIL.ImageChops as IC
    img.paste(IC.add(img, noise.convert("RGB"), scale=1.0, offset=-64))
    return img


def render_text_layer(text, font, rng):
    """Draw text on a transparent layer, skew it +/-8 deg, return (layer, world_box)."""
    tmp = Image.new("L", (10, 10))
    d = ImageDraw.Draw(tmp)
    try:
        bbox = d.textbbox((0, 0), text, font=font)
    except AttributeError:  # very old PIL
        bbox = (0, 0, int(d.textlength(text, font=font)), font.size)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(8, int(th * 0.6))
    layer = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    color = rng.choice([
        (255, 255, 255), (245, 245, 240), (20, 20, 20), (10, 10, 30),
        (200, 30, 30), (30, 60, 200), (220, 180, 40), (240, 240, 240),
    ])
    draw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=color)

    angle = rng.uniform(-8.0, 8.0)
    layer = layer.rotate(angle, resample=Image.BICUBIC, expand=False, center=(layer.width // 2, layer.height // 2))

    if rng.random() < 0.30:  # motion blur: horizontal smear kernel (middle row only, k*k odd)
        k = rng.choice([3, 5])
        kernel = [0.0] * (k * k)
        for c in range(k):
            kernel[(k // 2) * k + c] = 1.0 / k
        layer = layer.filter(ImageFilter.Kernel(size=(k, k), kernel=kernel, scale=1.0, offset=0))
    if rng.random() < 0.20:
        layer = layer.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.1)))
    return layer


def place(bg, layer, rng, used):
    """Paste layer at a random non-overlapping spot; return world AABB or None."""
    lw, lh = layer.size
    if lw >= bg.width or lh >= bg.height:
        return None
    for _ in range(20):
        x = rng.randint(0, bg.width - lw)
        y = rng.randint(0, bg.height - lh)
        box = (x, y, x + lw, y + lh)
        if any(_overlap(box, u, 4) for u in used):
            continue
        bg.paste(layer, (x, y), layer)  # layer alpha = mask
        used.append(box)
        alpha = layer.split()[-1]
        ab = alpha.getbbox()
        if not ab:
            return None
        # map alpha-bbox (layer-local) to world coords
        return [[x + ab[0], y + ab[1]], [x + ab[2] - 1, y + ab[1]],
                [x + ab[2] - 1, y + ab[3] - 1], [x + ab[0], y + ab[3] - 1]]
    return None


def _overlap(a, b, pad):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(here, "data", "synth")
    img_dir = os.path.join(out_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    rng = random.Random(args.seed)
    fonts_used = set()
    manifest = []
    made = 0
    for i in range(1, args.n + 1):
        w = rng.randint(640, 1024)
        h = rng.randint(420, 720)
        bg = gradient_bg(w, h, rng)
        used = []
        words = []
        n_lines = rng.randint(1, 4)
        font_size = rng.randint(18, 40)
        font, fname = pick_font(font_size)
        fonts_used.add(fname)
        for _ in range(n_lines):
            text = rand_line(rng)
            fs = max(14, font_size + rng.randint(-6, 6))
            if fs != font_size:
                font, fname = pick_font(fs)
                fonts_used.add(fname)
            layer = render_text_layer(text, font, rng)
            box = place(bg, layer, rng, used)
            if box is not None and text.strip():
                words.append({"text": text, "box": box})
        if not words:
            continue
        img = add_gaussian_noise(bg, rng, rng.uniform(0.8, 4.5))
        name = "img_%04d.jpg" % i
        img.save(os.path.join(img_dir, name), "JPEG", quality=rng.randint(70, 92))
        manifest.append({"img": "img/" + name, "w": img.width, "h": img.height, "words": words})
        made += 1

    with open(os.path.join(out_dir, "gt.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    print(json.dumps({
        "ok": True, "n": made, "dir": out_dir.replace("\\", "/"),
        "fonts": sorted(fonts_used), "gt": (out_dir + "/gt.json").replace("\\", "/"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
