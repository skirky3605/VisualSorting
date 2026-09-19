"""【已归档】分拣视觉 v0：HSV 阈值 + 简单几何规则的单文件版本。

这是"Result"窗口截图中那版脚本，已被 vision/sorter_vision.py 取代
（新版改为 Lab 色度/色相分割 + 迟滞阈值 + 凸包几何判据 + 姿态与标定）。
保留仅供对照，运行方式（在项目根目录执行，图片路径自行修改）：

    python legacy\main_v0_hsv.py
"""

import cv2
import numpy as np

# 重新调准针对该图的 HSV 阈值
COLOR_CONFIG = {
    "green":  {"lower": np.array([35, 50, 30]),  "upper": np.array([85, 255, 255]), "bgr": (0, 255, 0)},
    "yellow": {"lower": np.array([18, 60, 60]),  "upper": np.array([34, 255, 255]), "bgr": (0, 255, 255)},
    # 修复粉紫色：图中粉色长方体偏冷品红
    "pink":   {"lower": np.array([130, 40, 50]), "upper": np.array([175, 255, 255]), "bgr": (255, 0, 255)}
}

def classify_shape(cnt):
    """通过几何特征严格区分形状"""
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return "unknown", None

    # 1. 圆形度 (圆盘/圆柱的值接近 1.0)
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if circularity > 0.78:
        return "cylinder", None

    # 2. 最小外接旋转矩形分析 (Rotated Rect)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90.0
    aspect_ratio = w / max(h, 1e-5)

    # 3. 多边形拟合顶点数
    epsilon = 0.04 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    vertices = len(approx)

    if vertices == 3:
        return "prism_triangle", rect

    # 根据长宽比和面积区分大小立方体与长方体
    if aspect_ratio > 1.6:
        return "cuboid_long", rect
    elif area > 12000:
        # 特殊异形（如拱门桥）
        return "arch_bridge", rect
    else:
        return "cube_small", rect

def main(image_path="test.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        print("未找到图片")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    vis = img.copy()

    for color_name, cfg in COLOR_CONFIG.items():
        mask = cv2.inRange(hsv, cfg["lower"], cfg["upper"])
        
        # 开闭运算清理瓦楞纸缝隙的噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 过滤面积太小或太长条的反光噪点 (如左下角的压痕)
            if area < 2500:
                continue

            shape_name, rect = classify_shape(cnt)

            if shape_name == "cylinder":
                # 绘制圆形/圆柱
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                cv2.circle(vis, center, int(radius), cfg["bgr"], 2)
                cv2.putText(vis, f"{color_name} {shape_name}", (center[0]-40, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            elif rect is not None:
                # 绘制带抓取旋转角的有向矩形框
                box = cv2.boxPoints(rect)
                box = np.int64(box)
                cv2.drawContours(vis, [box], 0, cfg["bgr"], 2)

                cx, cy = int(rect[0][0]), int(rect[0][1])
                angle = rect[2]
                
                # 绘制中心点与抓取偏航角朝向线
                cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
                rad = np.deg2rad(angle)
                end_x = int(cx + 40 * np.cos(rad))
                end_y = int(cy + 40 * np.sin(rad))
                cv2.line(vis, (cx, cy), (end_x, end_y), (0, 0, 255), 2)

                label = f"{color_name} {shape_name} ({int(angle)}deg)"
                cv2.putText(vis, label, (cx - 50, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    scale = 800 / max(vis.shape[:2])
    disp = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
    cv2.imshow("Result", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("test3.jpg")
