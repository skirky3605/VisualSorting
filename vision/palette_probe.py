"""调色板探针：用 K-means 统计图像主色，并支持在指定像素点采样。

用途：换灯光/换背景后，快速重新标定 sorter_vision.py 里的颜色阈值。

用法::

    python palette_probe.py 图片1.jpg 图片2.jpg --k 8 --resize 640
    python palette_probe.py 图片.jpg --points 560,330 835,610
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

import sorter_vision as sv

imread_unicode = sv.imread_unicode


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    px = np.uint8([[bgr]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(int)


def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    px = np.uint8([[bgr]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].astype(int)


def describe(bgr: np.ndarray) -> str:
    bgr = np.rint(np.asarray(bgr, dtype=float)).astype(int)
    lab = bgr_to_lab(bgr)
    hsv = bgr_to_hsv(bgr)
    hue_deg = (np.degrees(np.arctan2(lab[2] - 128.0, lab[1] - 128.0)) + 360) % 360
    chroma = float(np.hypot(lab[1] - 128.0, lab[2] - 128.0))
    return (
        f"BGR=({bgr[0]:3d},{bgr[1]:3d},{bgr[2]:3d}) "
        f"HSV=({hsv[0]:3d},{hsv[1]:3d},{hsv[2]:3d}) "
        f"Lab=({lab[0]:3d},{lab[1]:4d},{lab[2]:4d}) "
        f"Lab色相={hue_deg:6.1f}° 色度={chroma:5.1f}"
    )


def kmeans_palette(img: np.ndarray, k: int, resize: int, min_chroma: float) -> None:
    if resize:
        scale = resize / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab.reshape(-1, 3)
    if min_chroma > 0:
        chroma = np.hypot(samples[:, 1] - 128.0, samples[:, 2] - 128.0)
        samples = samples[chroma >= min_chroma]
        if len(samples) < k * 10:
            print("  高饱和像素过少，跳过聚类")
            return
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(
        samples, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    tag = f"chroma>={min_chroma:g}" if min_chroma > 0 else "全图"
    print(f"  主色 (k={k}, {img.shape[1]}x{img.shape[0]}, {tag}):")
    for idx in order:
        center_lab = np.clip(centers[idx], 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(np.uint8([[center_lab]]), cv2.COLOR_LAB2BGR)[0, 0]
        pct = 100.0 * counts[idx] / counts.sum()
        print(f"    {pct:5.1f}%  {describe(bgr)}")


def sample_points(img: np.ndarray, points: list[tuple[int, int]], radius: int) -> None:
    print("  采样点:")
    for x, y in points:
        x0, x1 = max(0, x - radius), x + radius + 1
        y0, y1 = max(0, y - radius), y + radius + 1
        patch = img[y0:y1, x0:x1].reshape(-1, 3)
        mean_bgr = patch.mean(axis=0)
        print(f"    ({x:4d},{y:4d}) 均值 {describe(mean_bgr)}")


def chroma_histogram(img: np.ndarray, resize: int) -> None:
    """打印 Lab 色度直方图，并建议 chroma_min 该切在哪里。

    新换摄像头后跑一次：背景（纸板/桌面）会挤在低色度区，
    物料集中在高色度区，两者之间那条"空缝"就是最佳阈值。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.hypot(a, b).ravel()
    hist, edges = np.histogram(chroma, bins=np.arange(0, 85, 5))
    total = hist.sum() or 1
    print("  色度直方图 (0-80, 每 5 一档):")
    for i, cnt in enumerate(hist):
        pct = 100.0 * cnt / total
        if pct < 0.05:
            continue
        print(f"    {edges[i]:3.0f}-{edges[i + 1]:3.0f}: {pct:5.1f}%  {'#' * int(pct / 2)}")
    # 在 10~40 之间找最空的一段，作为建议阈值
    lo, hi = 2, 8            # 对应色度 10~40
    best = None
    for i in range(lo, hi):
        if hist[i] == 0:
            continue
        # 连续为空的区间
        j = i
        while j + 1 < hi and hist[j + 1] == 0:
            j += 1
        if j > i or best is None:
            best = (i, j)
        i = j
    # 直接给出"两侧都有量、中间最少"的位置
    window = [int(hist[i]) for i in range(lo, hi + 1)]
    if window:
        k = int(np.argmin(window)) + lo
        print(
            f"  建议 chroma_min ≈ {edges[k]:.0f}~{edges[k + 1]:.0f} "
            f"（该档像素最少，正好是背景与物料之间的空缝）"
        )


# 色相角 → 颜色名（角度用 atan2 结果，范围 -180~180）
NAME_BINS: list[tuple[str, float, float]] = [
    ("red", -20.0, 60.0),
    ("orange", 60.0, 82.0),
    ("yellow", 82.0, 118.0),
    ("green", 118.0, 168.0),
    ("cyan", -180.0, -100.0),
    ("pink", -100.0, -20.0),
]


def suggest_config(img: np.ndarray, resize: int, margin: float = 4.0) -> None:
    """扫一张有物料的图，直接输出可粘贴的 LAB_HUE_RANGES / chroma_min。"""
    if resize:
        s = resize / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    # 跳过低饱和像素（背景），只看可能是物料的像素
    hist, edges = np.histogram(chroma, bins=np.arange(0, 65, 5))
    lo, hi = 2, 6                      # 找到 10~30 之间最空的色度档
    k = int(np.argmin([hist[i] for i in range(lo, hi + 1)])) + lo
    chroma_min = float(edges[k])
    # 背景尾部：色度低于候选阈值的像素里最高的那 0.5%，宽松阈值必须高于它，
    # 否则纸板/白纸的阴影会被当成物体回收进来
    bg = chroma[chroma < chroma_min + 2]
    bg_tail = float(np.percentile(bg, 99.5)) if bg.size > 100 else 0.0
    strict = max(chroma_min, bg_tail + 6.0, 12.0)
    loose = max(chroma_min - 6.0, bg_tail + 2.0, 8.0)
    print(f"\n  ▶ 背景色度上界(99.5%) = {bg_tail:.1f}")
    print(f"  ▶ 建议 chroma_min = {strict:.0f}（严格）/ {loose:.0f}（宽松）")

    sel = chroma >= chroma_min + 2
    h = hue[sel]
    c = chroma[sel]
    if h.size < 200:
        print("  物料像素太少，换一张物料更大/更多的图")
        return
    print(f"  物料像素占比 {100.0 * h.size / hue.size:.1f}%（色度 ≥ {chroma_min + 2:.0f}）")
    print("  ── 各颜色实测区间 ──")
    lines = []
    for name, h0, h1 in NAME_BINS:
        m = (h >= h0) & (h <= h1)
        if int(m.sum()) < 200 or m.sum() < 0.002 * h.size:
            continue
        lo_v = float(np.percentile(h[m], 1)) - margin
        hi_v = float(np.percentile(h[m], 99)) + margin
        if name == "red":
            hi_v = min(hi_v, 70.0)
        lines.append((name, max(lo_v, h0 - 20), min(hi_v, h1 + 20),
                      float(np.percentile(c[m], 50)), int(m.sum())))
    if not lines:
        print("  没识别出成规模的颜色簇")
        return
    for name, lo_v, hi_v, c_mid, n in sorted(lines, key=lambda t: -t[4]):
        print(f"    {name:7s} 色相 {lo_v:7.1f} ~ {hi_v:6.1f}   中位色度 {c_mid:5.1f}   像素 {n}")
    print("\n  ── 粘贴到 sorter_vision.py ──")
    print("LAB_HUE_RANGES: dict[str, tuple[float, float]] = {")
    for name, lo_v, hi_v, _, _ in sorted(lines, key=lambda t: -t[4]):
        print(f'    "{name}": ({lo_v:.1f}, {hi_v:.1f}),')
    print("}")
    print(f"\n# Config 里改成:\n#   chroma_min: float = {strict:.0f}.0")
    print(f"#   chroma_loose: float = {loose:.0f}.0")


def main() -> None:
    parser = argparse.ArgumentParser(description="图像主色 / 像素采样探针")
    parser.add_argument("images", nargs="+", help="图片路径")
    parser.add_argument("--k", type=int, default=8, help="K-means 聚类数")
    parser.add_argument("--resize", type=int, default=640, help="聚类前缩放长边")
    parser.add_argument(
        "--points",
        nargs="*",
        default=[],
        help="采样点，格式 x,y（可多个）",
    )
    parser.add_argument("--radius", type=int, default=4, help="采样半径")
    parser.add_argument(
        "--min-chroma",
        type=float,
        default=0.0,
        help="只看 Lab 色度高于该值的像素（背景滤除，建议 20~30）",
    )
    parser.add_argument(
        "--hist",
        action="store_true",
        help="打印色度直方图并建议 chroma_min（换摄像头/换灯后跑这个）",
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="直接输出可粘贴的颜色配置块（LAB_HUE_RANGES + chroma_min）",
    )
    args = parser.parse_args()

    points = []
    for item in args.points:
        x_str, y_str = item.split(",")
        points.append((int(x_str), int(y_str)))

    for path in args.images:
        img = imread_unicode(path)
        print(f"{path}  ({img.shape[1]}x{img.shape[0]})")
        kmeans_palette(img, args.k, args.resize, args.min_chroma)
        if args.hist:
            chroma_histogram(img, args.resize)
        if args.suggest:
            suggest_config(img, args.resize)
        if points:
            sample_points(img, points, args.radius)
        print()


if __name__ == "__main__":
    main()
