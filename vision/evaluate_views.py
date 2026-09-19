#!/usr/bin/env python3
"""四个候选机位（视角）的定量比较：检出率、分类正确率、误检、分辨率、透视压缩。

用法::

    python evaluate_views.py test1.jpg test2.jpg test3.jpg test4.jpg \
        --out-dir ../out --csv ../out/views.csv

说明：脚本内置了这四张样张的人工标注真值（见 VIEWS），每张图对应一个候选机位。
换新样张时，把新图片名加到对应视角的 files 里，或按同样格式新增一个视角。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import sorter_vision as sv

# --------------------------------------------------------------------------
# 四个候选机位与人工真值
#   gt 项 = (颜色, 形状, 中心像素)，坐标在 1280x960 的处理尺寸下标注
#   形状族：cuboid_long 长条 / cube 立方体 / prism 三棱柱
#          arch 拱桥(U 形) / cylinder 圆柱
#   files 里任一文件名（不含扩展名、不区分大小写）都能被识别
# --------------------------------------------------------------------------
VIEWS: dict[str, dict] = {
    "near_nadir": {
        "cn": "近正俯视",
        "files": ["test1", "9adc9f0d4a9c41f972a589886298b68a"],
        "gt": [
            ("pink", "cuboid_long", (692, 307)),
            ("green", "cube", (666, 417)),
            ("yellow", "arch", (954, 384)),
            ("green", "prism", (941, 592)),
            ("green", "cube", (1088, 534)),
            ("green", "cuboid_long", (658, 709)),
            ("yellow", "cylinder", (913, 783)),
        ],
    },
    "wide_low": {
        "cn": "低角度大视野",
        "files": ["test2", "20ac9c3c498b71e8a6063287f3451bb9"],
        "gt": [
            ("pink", "cuboid_long", (381, 428)),
            ("green", "cube", (420, 507)),
            ("yellow", "arch", (645, 388)),
            ("green", "cube", (830, 430)),
            ("green", "prism", (770, 500)),
            ("green", "cuboid_long", (632, 714)),
            ("yellow", "cylinder", (914, 632)),
        ],
    },
    "mid_tilt": {
        "cn": "中俯视≈40°",
        "files": ["test3", "280fd27c422e02c7a0dccc8e4382bd38"],
        "gt": [
            ("pink", "cuboid_long", (458, 309)),
            ("green", "cube", (429, 374)),
            ("yellow", "arch", (678, 341)),
            ("green", "prism", (731, 464)),
            ("green", "cube", (840, 410)),
            ("green", "cuboid_long", (393, 564)),
            ("yellow", "cylinder", (720, 608)),
        ],
    },
    "close_low": {
        "cn": "低角度近拍",
        "files": ["test4", "314f7b916ae4fce15c2b0939aed2debf"],
        "gt": [
            ("green", "cuboid_long", (178, 408)),
            ("green", "cube", (403, 288)),
            ("pink", "cuboid_long", (490, 254)),
            ("yellow", "arch", (656, 333)),
            ("green", "cube", (698, 465)),
            ("green", "prism", (513, 460)),
            ("yellow", "cylinder", (302, 573)),
        ],
    },
}

MATCH_TOL_PX = 80.0        # 检测中心与真值中心的最大匹配距离
ROI_MARGIN_PX = 130.0      # 工作区 = 真值凸包外扩，用于统计「工作区内误检」


def lookup_view(stem: str) -> tuple[str, dict] | None:
    """按文件名匹配候选视角，返回 (视角 id, 视角定义)。"""
    key = stem.lower()
    for vid, spec in VIEWS.items():
        if key in [f.lower() for f in spec["files"]]:
            return vid, spec
    return None


def build_roi(points: list[tuple[int, int]]) -> np.ndarray:
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    center = hull.reshape(-1, 2).mean(axis=0)
    out = []
    for p in hull.reshape(-1, 2):
        v = p - center
        n = float(np.hypot(*v)) or 1.0
        out.append(p + v / n * ROI_MARGIN_PX)
    return np.asarray(out, dtype=np.float32)


def pairwise_min_gap(dets: list[sv.Detection]) -> float:
    """任意两个物体轮廓间的最小距离（像素），用于判断粘连风险。"""
    best = float("inf")
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            a = dets[i].contour.reshape(-1, 2).astype(np.float32)
            b = dets[j].contour.reshape(-1, 2).astype(np.float32)
            # 逐点最近距离（点数不多，够用）
            d = np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2))
            best = min(best, float(d))
    return best


def evaluate_one(
    path: str,
    cfg: sv.Config,
    view_id: str,
    spec: dict,
) -> tuple[dict, np.ndarray, list[sv.Detection], list[sv.Detection]]:
    gt = spec["gt"]
    roi = build_roi([p for _, _, p in gt])

    # 全图检测
    cfg_full = sv.Config(**{**cfg.__dict__, "roi": None, "roi_poly": None})
    bgr, dets_all = sv.process(path, cfg_full)
    # ROI 内检测
    cfg_roi = sv.Config(**{**cfg.__dict__, "roi_poly": [tuple(p) for p in roi]})
    _, dets_roi = sv.process(path, cfg_roi)

    matched_gt: set[int] = set()
    matched_det: set[int] = set()
    correct = 0
    errors: list[str] = []
    for gi, (gcolor, gshape, gcenter) in enumerate(gt):
        best_i, best_d = -1, MATCH_TOL_PX
        for di, det in enumerate(dets_roi):
            if di in matched_det:
                continue
            dist = float(np.hypot(det.cx - gcenter[0], det.cy - gcenter[1]))
            if dist < best_d:
                best_i, best_d = di, dist
        if best_i < 0:
            errors.append(f"漏检 {gcolor}_{gshape}@{gcenter}")
            continue
        matched_gt.add(gi)
        matched_det.add(best_i)
        det = dets_roi[best_i]
        if det.color == gcolor and det.shape == gshape:
            correct += 1
        else:
            errors.append(
                f"误判 {gcolor}_{gshape} -> {det.label} @{gcenter}"
            )

    fp_in = len(dets_roi) - len(matched_det)
    fp_out = len(dets_all) - len(dets_roi)
    areas = [d.area_px for d in dets_roi]
    box_like = [
        d for d in dets_roi if d.shape in ("cube", "cuboid_long", "prism")
    ]
    h_ext = float(np.median([d.hull_extent for d in box_like])) if box_like else 0.0
    box_aspect = (
        float(np.median([d.aspect for d in box_like])) if box_like else 0.0
    )
    edge_conv = (
        float(np.median([d.edge_conv for d in box_like])) if box_like else 0.0
    )

    stats = {
        "视角": f"{view_id}({spec['cn']})",
        "图片": Path(path).name,
        "检出数(全图)": len(dets_all),
        "检出数(工作区内)": len(dets_roi),
        "真值数": len(gt),
        "匹配数": len(matched_gt),
        "分类正确数": correct,
        "分类正确率": round(correct / len(gt), 3),
        "工作区内误检": fp_in,
        "工作区外误检": fp_out,
        "中位物体面积px": int(np.median(areas)) if areas else 0,
        "最小物体面积px": int(np.min(areas)) if areas else 0,
        "物体间最小间距px": round(pairwise_min_gap(dets_roi), 1)
        if len(dets_roi) > 1
        else 0.0,
        "box类中位填充率hExt": round(h_ext, 3),
        "box类中位长短轴比": round(box_aspect, 3),
        "透视收敛比(近边/远边)": round(edge_conv, 3),
        "错误明细": "; ".join(errors),
    }
    return stats, bgr, dets_all, dets_roi


def main() -> None:
    parser = argparse.ArgumentParser(description="候选机位定量评估")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--out-dir", help="标注图输出目录")
    parser.add_argument("--csv", help="结果表输出 CSV")
    parser.add_argument("--json", help="结果表输出 JSON")
    args = parser.parse_args()

    cfg = sv.Config()
    rows: list[dict] = []
    for path in args.images:
        hit = lookup_view(Path(path).stem)
        if hit is None:
            print(f"跳过（没有对应真值）: {path}")
            continue
        view_id, spec = hit
        stats, bgr, dets_all, dets_roi = evaluate_one(path, cfg, view_id, spec)
        rows.append(stats)

        roi = build_roi([p for _, _, p in spec["gt"]])
        vis = bgr.copy()
        cv2.polylines(vis, [roi.astype(np.int32)], True, (255, 128, 0), 2)
        vis = sv.annotate(vis, dets_all, cfg)
        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{view_id}_{Path(path).stem}.png"), vis)
        print(f"[{view_id} {spec['cn']}] {path}")
        for k, v in stats.items():
            print(f"    {k}: {v}")
        print()

    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV 已保存: {args.csv}")
    if args.json and rows:
        Path(args.json).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON 已保存: {args.json}")


if __name__ == "__main__":
    main()
