#!/usr/bin/env python3
"""单应标定：把图像像素映射到工作台毫米坐标，供 sorter_vision.py 输出抓取姿态。

相机固定后只需做一次。核心是拿到工作台上**一个已知尺寸矩形的 4 个角点在图像中的像素坐标**，
也就是"取样点"。提供三种取点方式：

1. 鼠标点选（推荐）：`--interactive --camera 0` 或 `--interactive --image 帧图.jpg`
   空格冻结画面 → 按提示顺序点 4 个角 → 回车确认，自动算标定并出网格自检图。
2. 网格图手抄：`--grid-image 帧图.jpg --grid-out 网格图.png`
   在图上叠加带刻度的坐标网格，肉眼读出角点像素后，用 `--points` 传入。
3. 直接给坐标：`--points "x1,y1;x2,y2;x3,y3;x4,y4" --size-mm "W,H"`

角点顺序必须与工作台坐标一一对应（工具里也会提示）::

    工作台坐标          图像像素
    (0,0)    原点   →  第 1 点
    (W,0)    +X 方向 →  第 2 点
    (W,H)    对角   →  第 3 点
    (0,H)    +Y 方向 →  第 4 点

工作台坐标系约定：X 向右、Y 向下（与图像一致），角度逆时针为正，
即 yaw=0° 表示物体长轴平行于工作台 X 轴指向右。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import sorter_vision as sv

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

WINDOW = "calibrate: SPACE freeze | click 4 corners | ENTER ok | R reset | ESC quit"
CORNER_NAMES = ["1/4 原点(0,0)", "2/4 +X(W,0)", "3/4 对角(W,H)", "4/4 +Y(0,H)"]


def parse_points(text: str) -> np.ndarray:
    pts = [tuple(float(v) for v in p.split(",")) for p in text.split(";") if p.strip()]
    if len(pts) != 4:
        raise SystemExit("--points 需要 4 个点，格式 x,y;x,y;x,y;x,y")
    return np.asarray(pts, dtype=np.float32)


def parse_size(text: str) -> tuple[float, float]:
    t = text.lower().replace("*", "x").replace("，", ",")
    parts = t.split("x") if "x" in t else t.split(",")
    if len(parts) != 2:
        raise SystemExit("--size-mm 格式应为 宽,高，例如 520,430")
    return float(parts[0]), float(parts[1])


def build_homography(img_pts: np.ndarray, size_mm: tuple[float, float]) -> np.ndarray:
    w_mm, h_mm = size_mm
    table_pts = np.asarray(
        [[0.0, 0.0], [w_mm, 0.0], [w_mm, h_mm], [0.0, h_mm]], dtype=np.float32
    )
    return cv2.getPerspectiveTransform(img_pts, table_pts)


def draw_grid_overlay(
    img: np.ndarray,
    img_pts: np.ndarray,
    size_mm: tuple[float, float],
    grid_mm: float,
) -> np.ndarray:
    """把工作台网格投影回图像，用来肉眼验证标定是否正确（格子应是正方形）。"""
    out = img.copy()
    w_mm, h_mm = size_mm
    h_table2img = np.linalg.inv(build_homography(img_pts, size_mm))

    segs = []
    for x in np.arange(0.0, w_mm + 1e-6, grid_mm):
        segs.append([[x, 0.0], [x, h_mm]])
    for y in np.arange(0.0, h_mm + 1e-6, grid_mm):
        segs.append([[0.0, y], [w_mm, y]])
    pts = np.asarray(segs, dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, h_table2img).reshape(-1, 2, 2)
    for a, b in proj:
        cv2.line(
            out,
            (int(round(a[0])), int(round(a[1]))),
            (int(round(b[0])), int(round(b[1]))),
            (255, 0, 0),
            2,
        )
    for p, name in zip(img_pts, ["O", "X", "XY", "Y"]):
        cv2.circle(out, (int(p[0]), int(p[1])), 7, (0, 0, 255), -1)
        cv2.putText(out, name, (int(p[0]) + 10, int(p[1])), 0, 0.9, (0, 0, 255), 2)
    return out


def draw_coord_grid(img: np.ndarray, step: int = 100) -> np.ndarray:
    """叠加带刻度的坐标网格，便于肉眼读出像素坐标（无 GUI 时的取点办法）。"""
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), (0, 255, 255), 1)
        for y in range(0, h, step):
            cv2.putText(
                out, f"{x},{y}", (x + 4, y + 16), 0, 0.45, (0, 0, 255), 1
            )
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)
    return out


def pick_points_interactive(
    frame: np.ndarray, size_mm: tuple[float, float] | None, grid_mm: float
) -> np.ndarray | None:
    """冻结画面后鼠标点 4 个角点，返回原图坐标；取消返回 None。"""
    max_w, max_h = 1280, 800
    scale = min(1.0, max_w / frame.shape[1], max_h / frame.shape[0])
    disp = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()

    clicks: list[tuple[int, int]] = []
    frozen = False

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and frozen and len(clicks) < 4:
            clicks.append((int(round(x / scale)), int(round(y / scale))))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print("  空格 = 冻结画面；然后按提示顺序点 4 个角点；回车 = 确认；R = 重来；ESC = 退出")

    while True:
        if frozen:
            view = frame.copy()
            for i, (cx, cy) in enumerate(clicks):
                cv2.circle(view, (cx, cy), 7, (0, 0, 255), -1)
                cv2.putText(view, str(i + 1), (cx + 10, cy - 8), 0, 0.9, (0, 0, 255), 2)
            if len(clicks) > 1:
                for i in range(len(clicks) - 1):
                    cv2.line(view, clicks[i], clicks[i + 1], (0, 255, 0), 2)
            if len(clicks) == 4:
                cv2.line(view, clicks[3], clicks[0], (0, 255, 0), 2)
                if size_mm:
                    view = draw_grid_overlay(view, np.asarray(clicks, dtype=np.float32),
                                             size_mm, grid_mm)
                else:
                    cv2.polylines(view, [np.asarray(clicks, np.int32)], True, (0, 255, 0), 2)
            hint = (
                CORNER_NAMES[len(clicks)] if len(clicks) < 4
                else "回车确认 / R 重来"
            )
            cv2.putText(view, f"next: {hint}", (10, 28), 0, 0.8, (0, 0, 255), 2)
        else:
            view = frame.copy()
            cv2.putText(view, "SPACE to freeze", (10, 28), 0, 0.8, (0, 0, 255), 2)

        if scale < 1:
            view = cv2.resize(view, None, fx=scale, fy=scale)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:                     # ESC
            clicks = []
            break
        if key == ord(" "):
            frozen = not frozen
            if not frozen:
                clicks = []
        elif key in (ord("r"), ord("R")):
            clicks = []
        elif key in (13, 10) and len(clicks) == 4:   # ENTER
            quad = np.asarray(clicks, dtype=np.float32)
            if not cv2.isContourConvex(quad.astype(np.int32)):
                print("  ✗ 四个点顺序不对（四边形自交），按 R 重来")
                continue
            break
    cv2.destroyWindow(WINDOW)
    return np.asarray(clicks, dtype=np.float32) if len(clicks) == 4 else None


def pick_one_point(frame: np.ndarray, hint: str) -> tuple[float, float] | None:
    """冻结画面后点一个点（用于指定参考点）。"""
    max_w, max_h = 1280, 800
    scale = min(1.0, max_w / frame.shape[1], max_h / frame.shape[0])
    clicks: list[tuple[float, float]] = []
    frozen = False

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and frozen and not clicks:
            clicks.append((x / scale, y / scale))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(f"  空格冻结 → 点击{hint} → 回车确认（R 重来，ESC 跳过）")
    while True:
        view = frame.copy()
        cv2.putText(view, "SPACE freeze" if not frozen else hint,
                    (10, 28), 0, 0.8, (0, 0, 255), 2)
        for (cx, cy) in clicks:
            cv2.drawMarker(view, (int(cx), int(cy)), (0, 0, 255),
                           cv2.MARKER_CROSS, 26, 3)
        if scale < 1:
            view = cv2.resize(view, None, fx=scale, fy=scale)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            clicks = []
            break
        if key == ord(" "):
            frozen = not frozen
        elif key in (ord("r"), ord("R")):
            clicks = []
        elif key in (13, 10) and clicks:
            break
    cv2.destroyWindow(WINDOW)
    return clicks[0] if clicks else None


def read_frame(args) -> np.ndarray:
    """交互模式取一帧：来自图片或 USB 摄像头。"""
    if args.image:
        return sv.imread_unicode(args.image)
    size = sv.parse_size(args.camera_size) if args.camera_size else None
    cap = sv.open_source(args.camera, size, args.warmup)
    frame = None
    for _ in range(30):          # 多读几帧，等自动曝光稳定
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.03)
    cap.release()
    if frame is None:
        raise SystemExit("摄像头没有取到图像")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="工作台单应标定（图像像素 → 毫米）")
    parser.add_argument("--interactive", action="store_true", help="鼠标点选 4 个角点")
    parser.add_argument("--image", help="交互模式输入图片")
    parser.add_argument("--camera", type=int, help="交互模式使用 USB 摄像头序号")
    parser.add_argument("--camera-size", default="1280x720", help="摄像头请求分辨率")
    parser.add_argument("--warmup", type=int, default=10, help="丢弃前多少帧")
    parser.add_argument("--points", help="四角点像素 x,y;x,y;x,y;x,y（非交互模式）")
    parser.add_argument("--size-mm", help="标定矩形实际尺寸 W,H（毫米）")
    parser.add_argument("--out", help="标定文件输出路径 calib.json")
    parser.add_argument("--grid-mm", type=float, default=50.0, help="自检网格间距")
    parser.add_argument("--check", help="自检用图片（把网格投回该图）")
    parser.add_argument("--check-out", help="自检图输出路径")
    parser.add_argument("--grid-image", help="给图片叠加坐标网格，便于手抄角点像素")
    parser.add_argument("--grid-out", help="坐标网格图输出路径")
    parser.add_argument("--grid-step", type=int, default=100, help="坐标网格间距（像素）")
    parser.add_argument(
        "--set-origin", action="store_true",
        help="交互模式：点完 4 个角点后，再点一下工作台上的参考点",
    )
    parser.add_argument(
        "--origin-px", help="参考点的像素坐标 x,y（二选一，与 --origin-mm 等效）",
    )
    parser.add_argument(
        "--origin-mm",
        help="参考点在工作台坐标下的位置 X,Y；不填默认用角点1(0,0) 作为参考点",
    )
    args = parser.parse_args()

    # 取点方式 2：坐标网格图
    if args.grid_image:
        img = sv.imread_unicode(args.grid_image)
        out = args.grid_out or "coord_grid.png"
        sv.imwrite_unicode(out, draw_coord_grid(img, args.grid_step))
        print(f"坐标网格图已保存: {out}")
        print("  图片上黄线为每 100 像素一格，左上角标注即该点像素坐标 (x,y)")
        return

    # 取点方式 1：鼠标点选
    if args.interactive:
        if not (args.image or args.camera is not None):
            raise SystemExit("交互模式需要 --image 或 --camera")
        size_mm = parse_size(args.size_mm) if args.size_mm else None
        frame = read_frame(args)
        print(f"  已取到画面: {frame.shape[1]}x{frame.shape[0]}")
        img_pts = pick_points_interactive(frame, size_mm, args.grid_mm)
        if img_pts is None:
            raise SystemExit("已取消，未生成标定文件")
        print("  角点像素: " + ";".join(f"{int(x)},{int(y)}" for x, y in img_pts))
        if not args.size_mm:
            args.size_mm = input("  请输入标定矩形的实际尺寸 W,H（毫米，如 520,430）: ")
        size_mm = parse_size(args.size_mm)
        if args.set_origin:
            # 复用刚才那帧画面，再点一次参考点
            pt = pick_one_point(frame, "工作台参考点（以后所有坐标都相对它）")
            if pt is not None:
                h_tmp = build_homography(img_pts, size_mm)
                mm = cv2.perspectiveTransform(
                    np.asarray([[pt]], dtype=np.float64), h_tmp
                ).reshape(-1, 2)[0]
                args.origin_mm = f"{mm[0]:.1f},{mm[1]:.1f}"
                print(f"  参考点像素 ({pt[0]:.0f},{pt[1]:.0f}) → 工作台坐标 "
                      f"({mm[0]:.1f}, {mm[1]:.1f}) mm")
        args.check = args.check or args.image
        if args.check is None:
            frame_path = "calib_check_source.jpg"
            sv.imwrite_unicode(frame_path, frame)
            args.check = frame_path
        args.check_out = args.check_out or "calib_check.png"
    else:
        if not (args.points and args.size_mm):
            raise SystemExit("非交互模式需要 --points 与 --size-mm（或用 --interactive / --grid-image）")
        img_pts = parse_points(args.points)
        size_mm = parse_size(args.size_mm)

    h_img2table = build_homography(img_pts, size_mm)
    # 参考点：优先用给定的毫米坐标；否则用 --origin-px 换算；都没给就是 (0,0)
    origin_mm: list[float] = [0.0, 0.0]
    if args.origin_mm:
        ox, oy = (float(v) for v in args.origin_mm.replace("，", ",").split(","))
        origin_mm = [ox, oy]
    elif args.origin_px:
        px, py = (float(v) for v in args.origin_px.replace("，", ",").split(","))
        mm = cv2.perspectiveTransform(
            np.asarray([[[px, py]]], dtype=np.float64), h_img2table
        ).reshape(-1, 2)[0]
        origin_mm = [float(mm[0]), float(mm[1])]
        print(f"  参考点像素 ({px:.0f},{py:.0f}) → 工作台坐标 "
              f"({origin_mm[0]:.1f}, {origin_mm[1]:.1f}) mm")

    if args.out:
        data = {
            "homography": h_img2table.tolist(),
            "note": "homography 把图像像素 (x,y) 映射到工作台毫米 (X,Y)，X 向右、Y 向下",
            "table_size_mm": list(size_mm),
            "image_points": img_pts.tolist(),
            "origin_mm": origin_mm,
            "origin_note": "识别结果输出的 table_xy_mm 是工作台绝对坐标；"
                           "rel_xy_mm / dist_mm 是相对 origin_mm 的偏移与距离",
        }
        Path(args.out).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"标定已保存: {args.out}   工作台 {size_mm[0]:.0f} x {size_mm[1]:.0f} mm")
        print(f"  参考点 (0,0) 位于工作台坐标 ({origin_mm[0]:.1f}, {origin_mm[1]:.1f}) mm")

    # 现场体检：四条边的"像素/毫米"应当接近，差太多说明角点顺序或实际尺寸填错了
    px_mm = [
        float(np.linalg.norm(img_pts[i] - img_pts[(i + 1) % 4])) / d
        for i, d in enumerate([size_mm[0], size_mm[1], size_mm[0], size_mm[1]])
    ]
    print("  四边像素/毫米: " + ", ".join(f"{v:.2f}" for v in px_mm)
          + "  （四条应接近，差太多说明角点或实际尺寸填错了）")

    if args.check:
        img = sv.imread_unicode(args.check)
        vis = draw_grid_overlay(img, img_pts, size_mm, args.grid_mm)
        out = args.check_out or "calib_check.png"
        sv.imwrite_unicode(out, vis)
        print(f"自检图已保存: {out}（蓝线网格应为正方形，间隔 {args.grid_mm:g}mm）")
    print("  复用本次取点: --points \"" +
          ";".join(f"{int(x)},{int(y)}" for x, y in img_pts) +
          f"\" --size-mm \"{size_mm[0]:g},{size_mm[1]:g}\"")


if __name__ == "__main__":
    main()
