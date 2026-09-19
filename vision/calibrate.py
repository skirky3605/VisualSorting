#!/usr/bin/env python3
"""单应标定：把图像像素映射到工作台毫米坐标，供 sorter_vision.py 输出抓取姿态。

相机固定后只需做一次。做法：在工作台上放一块尺寸已知的矩形标定板
（或在纸板上贴 4 个角标），量出它的实际长宽，再在图像里读出这 4 个角点像素坐标。

角点顺序必须一致（对应工作台坐标系原点、+X、+X+Y、+Y）：

    表坐标                图像像素
    (0,0)   ──→  p1 = x1,y1
    (W,0)   ──→  p2 = x2,y2
    (W,H)   ──→  p3 = x3,y3
    (0,H)   ──→  p4 = x4,y4

工作台坐标系约定：X 向右、Y 向下（与图像一致），角度逆时针为正，
即 yaw=0° 表示物体长轴平行于工作台 X 轴指向右。

用法::

    python calibrate.py --points "300,700;1180,690;1210,120;320,110" \
        --size-mm "520,430" --out calib.json
    # 结果自检：把工作台网格投影回图像，人工看一次
    python calibrate.py --points ... --size-mm ... --out calib.json \
        --check 图片.jpg --check-out grid.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_points(text: str) -> np.ndarray:
    pts = [tuple(float(v) for v in p.split(",")) for p in text.split(";") if p.strip()]
    if len(pts) != 4:
        raise SystemExit("--points 需要 4 个点，格式 x,y;x,y;x,y;x,y")
    return np.asarray(pts, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="工作台单应标定（图像像素 → 毫米）")
    parser.add_argument("--points", required=True, help="四个角点像素坐标")
    parser.add_argument("--size-mm", required=True, help="标定板实际尺寸 W,H（毫米）")
    parser.add_argument("--out", required=True, help="标定文件输出路径")
    parser.add_argument("--check", help="自检用图片")
    parser.add_argument("--check-out", help="自检图输出路径")
    parser.add_argument("--grid-mm", type=float, default=50.0, help="自检网格间距")
    args = parser.parse_args()

    img_pts = parse_points(args.points)
    w_mm, h_mm = (float(v) for v in args.size_mm.split(","))
    table_pts = np.asarray(
        [[0.0, 0.0], [w_mm, 0.0], [w_mm, h_mm], [0.0, h_mm]], dtype=np.float32
    )
    # 图像 → 工作台
    h_img2table = cv2.getPerspectiveTransform(img_pts, table_pts)
    data = {
        "homography": h_img2table.tolist(),
        "note": "homography 把图像像素 (x,y) 映射到工作台毫米 (X,Y)，X 向右、Y 向下",
        "table_size_mm": [w_mm, h_mm],
        "image_points": img_pts.tolist(),
    }
    Path(args.out).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"标定已保存: {args.out}")
    print(f"  工作台尺寸: {w_mm:.1f} x {h_mm:.1f} mm")

    if args.check:
        img = cv2.imread(args.check, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"无法读取自检图片: {args.check}")
        h_table2img = np.linalg.inv(h_img2table)
        lines = []
        xs = np.arange(0.0, w_mm + 1e-6, args.grid_mm)
        ys = np.arange(0.0, h_mm + 1e-6, args.grid_mm)
        for x in xs:
            lines.append([[x, 0.0], [x, h_mm]])
        for y in ys:
            lines.append([[0.0, y], [w_mm, y]])
        pts = np.asarray(lines, dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, h_table2img).reshape(-1, 2, 2)
        for (a, b) in proj:
            cv2.line(
                img,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                (255, 0, 0),
                2,
            )
        for p, name in zip(img_pts, ["O", "X", "XY", "Y"]):
            cv2.circle(img, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
            cv2.putText(img, name, (int(p[0]) + 8, int(p[1])), 0, 0.8,
                        (0, 0, 255), 2)
        out = args.check_out or "calib_check.png"
        cv2.imwrite(out, img)
        print(f"自检图已保存: {out}（网格应为正方形，间隔 {args.grid_mm:g}mm）")


if __name__ == "__main__":
    main()
