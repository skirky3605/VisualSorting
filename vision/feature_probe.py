"""几何特征诊断：打印每个连通域的候选判别特征，用于标定形状规则阈值。

用法::

    python feature_probe.py 图片.jpg [--chroma-min 26] [--min-area 2000]
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np

import sorter_vision as sv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--chroma-min", type=float, default=26.0)
    parser.add_argument("--min-area", type=float, default=1500.0)
    parser.add_argument("--max-width", type=int, default=1280)
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if args.max_width and img.shape[1] > args.max_width:
        s = args.max_width / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cfg = sv.Config(chroma_min=args.chroma_min, min_area_frac=0.0,
                    min_area_abs=args.min_area)
    masks = sv.color_masks(img, cfg)
    l_chan, _, _ = sv.lab_hue_chroma(img)

    header = (
        f"{'color':>7} {'center':>12} {'area':>7} {'hullA':>7} {'hExt':>5} "
        f"{'hAsp':>5} {'vc':>3} {'vf':>3} {'hCirc':>6} {'defPx':>6} {'d/min':>6} "
        f"{'A/hull':>6} {'circ':>6} {'turn':>5} {'splMin':>6} {'splCon':>6} "
        f"{'->命中的规则':<16}"
    )
    print(header)
    for color, mask in masks.items():
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < args.min_area:
                continue
            comp = sv.fill_holes(np.where(labels == i, 255, 0).astype(np.uint8))
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            hull_peri = cv2.arcLength(hull, True)
            (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
            hull_rect = cv2.minAreaRect(hull)
            hw, hh = hull_rect[1]
            vc = len(cv2.approxPolyDP(hull, 0.02 * hull_peri, True))
            vf = len(cv2.approxPolyDP(hull, 0.01 * hull_peri, True))
            turn = sv._max_turn(cv2.approxPolyDP(hull, 0.02 * hull_peri, True))
            split_min, split_contrast = sv.brightness_split(l_chan, comp)
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            mn = min(rw, rh)
            # 用主程序里同一套规则表判断，方便核对「这行会被判成什么」
            feats = sv.measure(c, l_chan, comp)
            if (feats["solidity"] < cfg.min_solidity
                    or feats["hull_extent"] < cfg.min_hull_extent):
                rule_name, rule_idx = "unknown(太零碎被滤除)", -1
            else:
                rule_name, rule_idx = sv.match_rule(feats)
            defect = 0.0
            try:
                hi = cv2.convexHull(c, returnPoints=False)
                dpts = np.asarray(
                    cv2.convexityDefects(c, np.sort(hi[:, 0]).reshape(-1, 1))
                ).reshape(-1, 4)
                if len(dpts):
                    defect = float(dpts[:, 3].max()) / 256.0
            except cv2.error:
                pass
            print(
                f"{color:>7} {f'({cx:.0f},{cy:.0f})':>12} {area:7.0f} "
                f"{hull_area:7.0f} {hull_area / max(hw * hh, 1e-6):5.2f} "
                f"{max(hw, hh) / max(min(hw, hh), 1e-6):5.2f} {vc:3d} {vf:3d} "
                f"{4 * math.pi * hull_area / hull_peri ** 2:6.3f} {defect:6.0f} "
                f"{defect / max(mn, 1.0):6.2f} {area / max(hull_area, 1e-6):6.3f} "
                f"{4 * math.pi * area / peri ** 2:6.3f} {turn:5.1f} "
                f"{split_min:6.2f} {split_contrast:6.1f} "
                f"-> {rule_name} (规则#{rule_idx})"
            )


if __name__ == "__main__":
    main()
