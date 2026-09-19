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


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    px = np.uint8([[bgr]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(int)


def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    px = np.uint8([[bgr]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].astype(int)


def describe(bgr: np.ndarray) -> str:
    lab = bgr_to_lab(bgr)
    hsv = bgr_to_hsv(bgr)
    return (
        f"BGR=({bgr[0]:3d},{bgr[1]:3d},{bgr[2]:3d}) "
        f"HSV=({hsv[0]:3d},{hsv[1]:3d},{hsv[2]:3d}) "
        f"Lab=({lab[0]:3d},{lab[1]:4d},{lab[2]:4d})"
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
    args = parser.parse_args()

    points = []
    for item in args.points:
        x_str, y_str = item.split(",")
        points.append((int(x_str), int(y_str)))

    for path in args.images:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"无法读取: {path}")
        print(f"{path}  ({img.shape[1]}x{img.shape[0]})")
        kmeans_palette(img, args.k, args.resize, args.min_chroma)
        if points:
            sample_points(img, points, args.radius)
        print()


if __name__ == "__main__":
    main()
