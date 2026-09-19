#!/usr/bin/env python3
"""智能分拣视觉：颜色 + 形状识别与姿态估计（初版）。

方案要点
--------
1. 颜色：在 CIE-Lab 空间用「色度 chroma + 色相角 hue」分割。实测本场景中
   物块的 chroma 为 30~55，纸板/桌面背景 chroma < 12，因此仅靠色度即可把
   背景干净滤除，再用色相角区分 品红 / 黄 / 绿 三类。
2. 形状：在连通域轮廓上取几何特征（外接旋转矩形填充率 extent、圆度、
   凸包顶点数、凸度 solidity、长短轴比），用规则判别
   圆柱 / 拱桥(U 形) / 三棱柱 / 立方体 / 长条。无需训练数据。
3. 姿态：外接旋转矩形的中心 = 物体位置，长轴方向 = 偏航角；若提供单应标定，
   再换算成工作台坐标系下的毫米坐标与角度。拱桥块额外给出「缺口朝向」，
   因为抓取拱桥必须知道缺口的方位。

用法::

    # 单张图片
    python sorter_vision.py 图片.jpg --out 结果.jpg --json 结果.json
    # 带标定（工作台坐标系 mm / 度）
    python sorter_vision.py 图片.jpg --calib calib.json
    # 打印全部几何特征，便于重新调阈值
    python sorter_vision.py 图片.jpg --debug
"""

from __future__ import annotations

import argparse
import json
import math
import operator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# 颜色定义：Lab 色相角 = atan2(B-128, A-128)，单位度
#   pink(品红) ≈ -60 ~ -30      yellow(黄) ≈ 100 ~ 118      green(绿) ≈ 130 ~ 145
# 实测三类物体与背景可分性很好，这里留了较宽余量。
# --------------------------------------------------------------------------
LAB_HUE_RANGES: dict[str, tuple[float, float]] = {
    "pink": (-115.0, -8.0),
    "yellow": (96.0, 124.0),
    "green": (124.0, 200.0),
}

# 标注用色（BGR）：与物体同色系但更亮/更饱和，并配黑色描边保证可读性
ANNOT_BGR: dict[str, tuple[int, int, int]] = {
    "pink": (255, 0, 255),
    "yellow": (0, 220, 255),
    "green": (0, 255, 0),
}

SHAPE_CN = {
    "cylinder": "圆柱",
    "arch": "拱桥",
    "prism": "三棱柱",
    "cube": "立方体",
    "cuboid_long": "长条",
    "unknown": "未知",
}

# --------------------------------------------------------------------------
# 形状判别规则表（新增形状只需在这里加行，详见 README「如何添加新形状」）
#
#   * 从上往下逐条匹配，命中即返回，所以**兜底规则放在最后**；
#   * 每条 = (形状名, [(特征名, 比较符, 阈值), ...])，条件之间是「与」；
#   * 同一形状写多条规则 = 条件之间「或」（斜视/俯视需要不同判据时就靠这个）；
#   * 特征名对应 measure() 返回的字典键，新增特征只要在 measure() 里补一行。
# --------------------------------------------------------------------------
SHAPE_RULES: list[tuple[str, list[tuple[str, str, float]]]] = [
    # 拱桥：U 形缺口使轮廓面积明显小于凸包面积
    ("arch", [("solidity", "<=", 0.80), ("hull_extent", ">=", 0.65)]),
    # 拱桥（斜视时缺口被压扁，改用凸包最深缺陷判据）
    ("arch", [("defect_ratio", ">=", 0.45), ("solidity", "<=", 0.92)]),
    # 圆柱：边界平滑（圆/椭圆无尖角），立方体的直角棱边最大转角 > 68°
    ("cylinder", [("hull_circ", ">=", 0.90), ("max_turn", "<=", 62.0)]),
    ("cylinder", [("hull_circ", ">=", 0.90), ("hull_verts_fine", ">=", 9)]),
    # 长条：长短轴比很大（先于三棱柱判断，避免被压扁的长条落到棱柱分支）
    ("cuboid_long", [("hull_aspect", ">=", 1.5)]),
    # 三棱柱：斜视时露出三角端面，凸包填充率明显低于立方体
    ("prism", [("hull_extent", "<=", 0.78)]),
    # 兜底：剩下的都是立方体（条件为空 = 恒真，必须放最后）
    ("cube", []),
]

_CMP = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
}


@dataclass
class Config:
    chroma_min: float = 26.0          # Lab 色度下限，低于此值视为背景
    l_min: float = 35.0               # 亮度下限，滤掉阴影
    chroma_loose: float = 13.0        # 迟滞阈值：暗面/高光用更宽的下限回收
    l_min_loose: float = 20.0
    strict_evidence: float = 0.12     # 连通域内至少这么多比例的像素要满足严格阈值
    min_area_frac: float = 0.0006     # 最小连通域面积（占整图比例）
    min_area_abs: float = 400.0       # 最小连通域面积（绝对像素）
    morph_open: int = 3
    morph_close: int = 7
    max_width: int = 1280             # 处理前缩放的长边上限，0 表示保持原尺寸
    cube_small_ratio: float = 0.80    # 立方体尺寸小于该比例时判为小立方体
    min_solidity: float = 0.55        # 轮廓/凸包面积比下限，滤掉零碎背景
    min_hull_extent: float = 0.50     # 凸包/外接矩形填充率下限
    # 工作区 ROI（相机固定后设置一次）：矩形 (x, y, w, h) 或任意多边形，
    # 中心不在 ROI 内的检测结果直接丢弃，可滤掉画面里的背景杂物
    roi: tuple[float, float, float, float] | None = None
    roi_poly: list[tuple[float, float]] | None = None


@dataclass
class Detection:
    color: str
    shape: str
    cx: float
    cy: float
    angle_deg: float          # 长轴方向：图像 x 轴正向为 0°，逆时针（向上）为正
    length_px: float          # 外接旋转矩形长边
    width_px: float           # 外接旋转矩形短边
    area_px: float
    extent: float             # 轮廓面积 / 外接旋转矩形面积（=顶面可见度代理）
    circularity: float
    solidity: float
    hull_verts: int
    aspect: float
    hull_extent: float = 0.0
    hull_circ: float = 0.0
    defect_ratio: float = 0.0
    edge_conv: float = 1.0
    feats: dict = field(default_factory=dict, repr=False)   # measure() 的原始特征
    box_poly: np.ndarray = field(repr=False, default=None)
    contour: np.ndarray = field(repr=False, default=None)
    yaw_deg: float | None = None       # 标定后：工作台坐标系下的偏航角
    table_xy_mm: tuple[float, float] | None = None
    size_mm: tuple[float, float] | None = None
    notch_dir_deg: float | None = None  # 拱桥缺口朝向（图像坐标系）
    size_class: str = ""

    @property
    def label(self) -> str:
        name = f"{self.color}_{self.shape}"
        if self.size_class:
            name += f"_{self.size_class}"
        return name

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "color": self.color,
            "shape": self.shape,
            "size_class": self.size_class,
            "center_px": [round(self.cx, 1), round(self.cy, 1)],
            "table_xy_mm": None
            if self.table_xy_mm is None
            else [round(v, 1) for v in self.table_xy_mm],
            "angle_img_deg": round(self.angle_deg, 1),
            "yaw_deg": None if self.yaw_deg is None else round(self.yaw_deg, 1),
            "notch_dir_deg": None
            if self.notch_dir_deg is None
            else round(self.notch_dir_deg, 1),
            "size_px": [round(self.length_px, 1), round(self.width_px, 1)],
            "size_mm": None
            if self.size_mm is None
            else [round(self.size_mm[0], 1), round(self.size_mm[1], 1)],
            "area_px": int(self.area_px),
            # 直接输出 measure() 里的全部标量特征，新增特征无需改这里
            "features": {
                k: (round(float(v), 4) if isinstance(v, float) else v)
                for k, v in self.feats.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            },
        }


# --------------------------------------------------------------------------
# 分割
# --------------------------------------------------------------------------
def lab_hue_chroma(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan = lab[..., 0]
    a_chan = lab[..., 1] - 128.0
    b_chan = lab[..., 2] - 128.0
    chroma = np.hypot(a_chan, b_chan)
    hue = np.degrees(np.arctan2(b_chan, a_chan))
    return l_chan, hue, chroma


def color_masks(bgr: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    """按颜色分割物体。

    采用「迟滞阈值」：严格阈值（chroma_min）用于确认物体，宽松阈值
    （chroma_loose）用于回收同一物体上被阴影压暗或高光冲淡的面，
    但一个宽松连通域内必须有足够比例的严格像素才被接受，避免把
    纸板阴影误当成物体。
    """
    l_chan, hue, chroma = lab_hue_chroma(bgr)
    strict_all = (chroma >= cfg.chroma_min) & (l_chan >= cfg.l_min)
    loose_all = (chroma >= cfg.chroma_loose) & (l_chan >= cfg.l_min_loose)
    h, w = bgr.shape[:2]
    min_area = max(cfg.min_area_abs, cfg.min_area_frac * h * w)
    masks: dict[str, np.ndarray] = {}
    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morph_open, cfg.morph_open)
    )
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.morph_close, cfg.morph_close)
    )
    for name, (h0, h1) in LAB_HUE_RANGES.items():
        in_hue = (hue >= h0) & (hue <= h1)
        strict = (strict_all & in_hue).astype(np.uint8)
        loose = (loose_all & in_hue).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(loose, 8)
        keep = np.zeros_like(loose)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            strict_px = int(np.count_nonzero(strict[labels == i]))
            if strict_px < max(120, cfg.strict_evidence * area):
                continue
            keep[labels == i] = 255
        mask = keep
        if cfg.morph_open > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        if cfg.morph_close > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        masks[name] = mask
    return masks


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """用外轮廓填充孤岛空洞，同时保留与边缘相连的缺口（拱桥开口不会被填掉）。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, -1)
    return filled


# --------------------------------------------------------------------------
# 几何特征与形状判别
# --------------------------------------------------------------------------
def _angle_of(dx: float, dy: float) -> float:
    """图像坐标系角度：向右为 0°，向上为正，范围 (-90, 90]。"""
    ang = math.degrees(math.atan2(-dy, dx))
    while ang > 90.0:
        ang -= 180.0
    while ang <= -90.0:
        ang += 180.0
    return ang


def _max_turn(poly: np.ndarray) -> float:
    """多边形顶点的最大转折角（度）。椭圆/圆 ≈ 小，多边形 ≈ 大。"""
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return 0.0
    best = 0.0
    for i in range(n):
        a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        best = max(best, 180.0 - math.degrees(math.acos(cosv)))
    return best


def _edge_convergence(hull: np.ndarray, max_angle_diff: float = 20.0) -> float:
    """剪影中最长的两条近平行、不相邻边的长度比（>=1）。

    正交投影（正上方）时立方体的对边等长 → ≈1.0；
    斜视时远边被压缩 → 比值变大，等价于「透视畸变强度」。
    """
    pts = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 4:
        return 1.0
    edges = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        v = b - a
        length = float(np.hypot(*v))
        if length < 1.0:
            continue
        edges.append((i, length, math.degrees(math.atan2(v[1], v[0]))))
    if len(edges) < 2:
        return 1.0
    edges.sort(key=lambda e: -e[1])
    best = 1.0
    for a in range(min(6, len(edges))):
        for b in range(a + 1, min(6, len(edges))):
            i, li, ai = edges[a]
            j, lj, aj = edges[b]
            # 不相邻（不能共享顶点，考虑闭合首尾）
            if (i - j) % n in (0, 1, n - 1):
                continue
            diff = abs(ai - aj) % 180.0
            diff = min(diff, 180.0 - diff)
            if diff > max_angle_diff:
                continue
            ratio = max(li, lj) / min(li, lj)
            if ratio > best:
                best = ratio
    return float(best)


def brightness_split(
    l_chan: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    """在目标区域内做 Otsu 明暗二分，返回（较小区域占比, 两区域灰度差）。"""
    vals = l_chan[mask > 0]
    if vals.size < 50:
        return 0.5, 0.0
    u8 = np.clip(vals, 0, 255).astype(np.uint8).reshape(-1, 1)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = vals[vals > thr]
    dark = vals[vals <= thr]
    if bright.size == 0 or dark.size == 0:
        return 0.5, 0.0
    frac = bright.size / vals.size
    return float(min(frac, 1.0 - frac)), float(abs(bright.mean() - dark.mean()))


def measure(contour: np.ndarray, l_chan: np.ndarray | None = None,
            mask: np.ndarray | None = None) -> dict:
    """在轮廓上提取形状判别所需的全部几何特征。

    关键点是同时使用「轮廓」和「凸包」两套量：

    * 凸包填充率 ``hull_extent`` = 凸包面积 / 凸包外接矩形面积。
      立方体/长条/圆柱 ≈ 0.8~1.0，三棱柱 ≈ 0.5~0.7（俯视时三角足迹）。
    * 轮廓/凸包面积比 ``fill``：U 形拱桥因缺口而明显小于 1。
    * 凸包缺陷深度 ``defect_ratio`` = 最深缺陷 / 短边：拱桥的缺口很深
      （≈0.5~0.8），而高光/阴影在轮廓边缘啃出的小缺口较浅（<0.45）。
    """
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    rect = cv2.minAreaRect(contour)
    (rcx, rcy), (rw, rh), _ = rect
    box = cv2.boxPoints(rect)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    hull_peri = float(cv2.arcLength(hull, True))
    hull_rect = cv2.minAreaRect(hull)
    hw, hh = hull_rect[1]
    hull_extent = float(hull_area / max(hw * hh, 1e-6))
    hull_circ = float(4 * math.pi * hull_area / max(hull_peri ** 2, 1e-6))
    hull_verts = len(cv2.approxPolyDP(hull, 0.02 * hull_peri, True))
    hull_verts_fine = len(cv2.approxPolyDP(hull, 0.01 * hull_peri, True))
    max_turn = _max_turn(cv2.approxPolyDP(hull, 0.02 * hull_peri, True))
    # 透视收敛度：剪影中两条最长的近平行边的长度比。
    # 正俯视 ≈ 1.0（正交投影），斜视时近边明显长于远边，该比值直接
    # 反映「透视压缩」强度，也就是姿态（位置/角度）误差的量级。
    # 注意：凸包原始点很密（掩膜边缘呈锯齿），必须先做多边形简化，
    # 否则「最长的边」会被锯齿切碎，得到无意义的结果。
    edge_conv = _edge_convergence(cv2.approxPolyDP(hull, 0.02 * hull_peri, True))

    split_min, split_contrast = 0.5, 0.0
    if l_chan is not None and mask is not None:
        split_min, split_contrast = brightness_split(l_chan, mask)

    defect_depth = 0.0
    try:
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        dpts = np.asarray(
            cv2.convexityDefects(contour, np.sort(hull_idx[:, 0]).reshape(-1, 1))
        ).reshape(-1, 4)
        if len(dpts):
            defect_depth = float(dpts[:, 3].max()) / 256.0
    except (cv2.error, ValueError):
        pass

    # 长轴方向：取旋转矩形中较长的一条边
    edges = [(i, (i + 1) % 4) for i in range(4)]
    lengths = [
        float(np.hypot(*(box[j] - box[i]))) for i, j in edges
    ]
    k = int(np.argmax(lengths))
    i, j = edges[k]
    v = box[j] - box[i]
    angle = _angle_of(float(v[0]), float(v[1]))
    length = max(rw, rh)
    width = min(rw, rh)

    return {
        "area": area,
        "peri": peri,
        "rect": rect,
        "box": box,
        "cx": float(rcx),
        "cy": float(rcy),
        "length": float(length),
        "width": float(width),
        "aspect": float(length / max(width, 1e-6)),
        "extent": float(area / max(length * width, 1e-6)),
        "circularity": float(4 * math.pi * area / max(peri * peri, 1e-6)),
        "solidity": float(area / max(hull_area, 1e-6)),
        "hull_extent": hull_extent,
        "hull_aspect": float(max(hw, hh) / max(min(hw, hh), 1e-6)),
        "hull_circ": hull_circ,
        "hull_verts_fine": int(hull_verts_fine),
        "hull_verts": int(hull_verts),
        "max_turn": float(max_turn),
        "edge_conv": float(edge_conv),
        "split_min": split_min,
        "split_contrast": split_contrast,
        "defect_depth": defect_depth,
        "defect_ratio": float(defect_depth / max(min(rw, rh), 1.0)),
        "angle": angle,
        "hull": hull,
    }


def match_rule(f: dict) -> tuple[str, int]:
    """在 SHAPE_RULES 里逐条匹配，返回 (形状名, 规则序号)，未命中返回 ("", -1)。"""
    for idx, (shape, conds) in enumerate(SHAPE_RULES):
        if all(_CMP[op](f[feat], thr) for feat, op, thr in conds):
            return shape, idx
    return "", -1


def classify_shape(f: dict, cfg: Config) -> str:
    """基于几何特征的形状判别。

    规则本身在 SHAPE_RULES 里（数据驱动，新增形状 = 加一行），
    这里只做「零碎背景剔除」+ 规则匹配。阈值由 feature_probe.py 在样张上标定。
    """
    # 拒绝零碎背景：轮廓太不饱满、或凸包填充率过低
    if f["solidity"] < cfg.min_solidity or f["hull_extent"] < cfg.min_hull_extent:
        return "unknown"
    shape, _ = match_rule(f)
    return shape or "unknown"


def notch_direction(contour: np.ndarray, cx: float, cy: float) -> float | None:
    """拱桥缺口朝向：用凸包缺陷最深点相对中心的方向表示。"""
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 4:
        return None
    try:
        defects = cv2.convexityDefects(contour, np.sort(hull_idx[:, 0]))
    except cv2.error:
        return None
    if defects is None:
        return None
    d = np.asarray(defects).reshape(-1, 4)
    depths = d[:, 3] / 256.0
    order = np.argsort(-depths)
    for idx in order[:3]:
        if depths[idx] < 5.0:
            break
        px, py = np.asarray(contour).reshape(-1, 2)[d[idx, 2]]
        return _angle_of(float(px - cx), float(py - cy))
    return None


def in_roi(x: float, y: float, cfg: Config) -> bool:
    """判断点是否落在工作区 ROI 内（未设置 ROI 时恒为 True）。"""
    if cfg.roi_poly:
        poly = np.asarray(cfg.roi_poly, dtype=np.float32)
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
    if cfg.roi:
        rx, ry, rw, rh = cfg.roi
        return rx <= x <= rx + rw and ry <= y <= ry + rh
    return True


def detect(bgr: np.ndarray, cfg: Config) -> list[Detection]:
    masks = color_masks(bgr, cfg)
    l_chan, _, _ = lab_hue_chroma(bgr)
    h, w = bgr.shape[:2]
    min_area = max(cfg.min_area_abs, cfg.min_area_frac * h * w)
    dets: list[Detection] = []

    for color, mask in masks.items():
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for idx in range(1, num):
            if stats[idx, cv2.CC_STAT_AREA] < min_area:
                continue
            comp = np.where(labels == idx, 255, 0).astype(np.uint8)
            comp = fill_holes(comp)
            contours, _ = cv2.findContours(
                comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) < min_area:
                continue
            f = measure(contour, l_chan, comp)
            shape = classify_shape(f, cfg)
            if shape == "unknown":
                continue
            det = Detection(
                color=color,
                shape=shape,
                cx=f["cx"],
                cy=f["cy"],
                angle_deg=f["angle"],
                length_px=f["length"],
                width_px=f["width"],
                area_px=f["area"],
                extent=f["extent"],
                circularity=f["circularity"],
                solidity=f["solidity"],
                hull_verts=f["hull_verts"],
                aspect=f["aspect"],
                hull_extent=f["hull_extent"],
                hull_circ=f["hull_circ"],
                defect_ratio=f["defect_ratio"],
                edge_conv=f["edge_conv"],
                feats=f,
                box_poly=f["box"],
                contour=contour,
            )
            if shape == "arch":
                det.notch_dir_deg = notch_direction(contour, f["cx"], f["cy"])
            if not in_roi(det.cx, det.cy, cfg):
                continue
            dets.append(det)

    _assign_size_class(dets, cfg)
    dets.sort(key=lambda d: -d.area_px)
    return dets


def _assign_size_class(dets: list[Detection], cfg: Config) -> None:
    """立方体按尺寸分为 large / small（对应"立方体"与"小立方体"两种物料）。"""
    cubes = [d for d in dets if d.shape == "cube"]
    if not cubes:
        return
    ref = float(np.median([d.length_px for d in cubes]))
    for d in cubes:
        d.size_class = "small" if d.length_px < cfg.cube_small_ratio * ref else "large"
    # 只有一种立方体时不再区分大小，避免噪声
    if len({d.size_class for d in cubes}) == 1:
        for d in cubes:
            d.size_class = ""


# --------------------------------------------------------------------------
# 标定：图像 <-> 工作台
# --------------------------------------------------------------------------
def load_calibration(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["H"] = np.array(data["homography"], dtype=np.float64)
    return data


def apply_calibration(dets: list[Detection], calib: dict) -> None:
    h_mat = calib["H"]

    def to_table(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, h_mat).reshape(-1, 2)

    for det in dets:
        # 位置
        (mx, my) = to_table([[det.cx, det.cy]])[0]
        det.table_xy_mm = (float(mx), float(my))
        # 姿态：把长轴两端点投到工作台坐标系，重算角度
        rad = math.radians(det.angle_deg)
        half = det.length_px / 2.0
        p0 = (det.cx - half * math.cos(rad), det.cy + half * math.sin(rad))
        p1 = (det.cx + half * math.cos(rad), det.cy - half * math.sin(rad))
        q0, q1 = to_table([p0, p1])
        det.yaw_deg = _angle_of(float(q1[0] - q0[0]), float(q1[1] - q0[1]))
        if det.notch_dir_deg is not None:
            rad = math.radians(det.notch_dir_deg)
            r = det.width_px / 2.0
            n0 = (det.cx, det.cy)
            n1 = (det.cx + r * math.cos(rad), det.cy - r * math.sin(rad))
            a0, a1 = to_table([n0, n1])
            det.notch_dir_deg = _angle_of(float(a1[0] - a0[0]), float(a1[1] - a0[1]))
        # 尺寸：用轴端点在台面上的距离近似
        rad = math.radians(det.angle_deg + 90.0)
        hw = det.width_px / 2.0
        s0 = (det.cx - hw * math.cos(rad), det.cy + hw * math.sin(rad))
        s1 = (det.cx + hw * math.cos(rad), det.cy - hw * math.sin(rad))
        b0, b1 = to_table([s0, s1])
        det.size_mm = (
            float(np.hypot(*(q1 - q0))),
            float(np.hypot(*(b1 - b0))),
        )


# --------------------------------------------------------------------------
# 可视化
# --------------------------------------------------------------------------
def annotate(bgr: np.ndarray, dets: list[Detection], cfg: Config) -> np.ndarray:
    img = bgr.copy()
    for det in dets:
        color = ANNOT_BGR.get(det.color, (255, 255, 255))
        box = det.box_poly.astype(np.int32)
        cv2.polylines(img, [box], True, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.polylines(img, [box], True, color, 2, cv2.LINE_AA)

        # 长轴方向：白底黑边，避免与物体同色看不清
        rad = math.radians(det.angle_deg)
        half = det.length_px * 0.35
        p0 = (int(det.cx - half * math.cos(rad)), int(det.cy + half * math.sin(rad)))
        p1 = (int(det.cx + half * math.cos(rad)), int(det.cy - half * math.sin(rad)))
        cv2.arrowedLine(img, p0, p1, (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(img, p0, p1, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(img, (int(det.cx), int(det.cy)), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (int(det.cx), int(det.cy)), 3, (255, 255, 255), -1, cv2.LINE_AA)

        if det.notch_dir_deg is not None:
            rad = math.radians(det.notch_dir_deg)
            r = det.width_px * 0.5
            n1 = (
                int(det.cx + r * math.cos(rad)),
                int(det.cy - r * math.sin(rad)),
            )
            cv2.arrowedLine(
                img,
                (int(det.cx), int(det.cy)),
                n1,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
                tipLength=0.3,
            )
            cv2.arrowedLine(
                img,
                (int(det.cx), int(det.cy)),
                n1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.3,
            )

        text = f"{det.label} ({det.angle_deg:.0f}deg)"
        if det.size_mm:
            text += f" {det.size_mm[0]:.0f}x{det.size_mm[1]:.0f}mm"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx = int(np.clip(box[:, 0].min(), 0, img.shape[1] - tw - 8))
        ty = int(np.clip(box[:, 1].min() - 8, th + 6, img.shape[0] - 4))
        cv2.rectangle(
            img, (tx - 4, ty - th - 6), (tx + tw + 4, ty + 4), (0, 0, 0), -1
        )
        cv2.putText(
            img,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return img


def print_table(dets: list[Detection]) -> None:
    if not dets:
        print("  未检测到任何物体")
        return
    head = (
        f"  {'#':>2}  {'label':<22}{'center_px':>16}  {'angle':>7}  "
        f"{'size_px':>14}  {'hAsp':>6}  {'hExt':>6}  {'fill':>6}  "
        f"{'hCirc':>6}  {'defR':>6}  {'vf':>4}"
    )
    print(head)
    for i, d in enumerate(dets, 1):
        print(
            f"  {i:>2}  {d.label:<22}"
            f"{f'({d.cx:.0f},{d.cy:.0f})':>16}  {d.angle_deg:>6.1f}°  "
            f"{f'{d.length_px:.0f}x{d.width_px:.0f}':>14}  {d.aspect:>6.2f}  "
            f"{d.hull_extent:>6.3f}  {d.solidity:>6.3f}  {d.hull_circ:>6.3f}  "
            f"{d.defect_ratio:>6.2f}  {d.hull_verts:>4}"
        )


# --------------------------------------------------------------------------
def process(path: str, cfg: Config, calib: dict | None = None) -> tuple[np.ndarray, list[Detection]]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"无法读取图片: {path}")
    if cfg.max_width and bgr.shape[1] > cfg.max_width:
        scale = cfg.max_width / bgr.shape[1]
        bgr = cv2.resize(
            bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    dets = detect(bgr, cfg)
    if calib:
        apply_calibration(dets, calib)
    return bgr, dets


def run_stream(
    url: str, cfg: Config, calib: dict | None, args: argparse.Namespace
) -> None:
    """实时模式：从 MJPEG 视频流（ESP32-S3 /stream）逐帧识别。

    这是分拣系统的实际工作形态：ESP32 只负责出图，识别与姿态解算在 PC 端。
    """
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频流: {url}")
    print(f"已连接视频流: {url}   (Ctrl+C 退出)")
    frame_id = 0
    last_vis = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取失败，重试…")
            continue
        frame_id += 1
        if cfg.max_width and frame.shape[1] > cfg.max_width:
            scale = cfg.max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        dets = detect(frame, cfg)
        if calib:
            apply_calibration(dets, calib)
        last_vis = annotate(frame, dets, cfg)
        if args.frames == 0 or frame_id % 10 == 1:
            summary = ", ".join(
                f"{d.label}({d.angle_deg:.0f}°)" for d in dets
            )
            print(f"  #{frame_id:5d} {len(dets)} 个: {summary}")
        if args.snapshot and (args.frames == 0 or frame_id % 30 == 0):
            cv2.imwrite(args.snapshot, last_vis)
        if args.frames and frame_id >= args.frames:
            break
    cap.release()
    if args.snapshot and last_vis is not None:
        cv2.imwrite(args.snapshot, last_vis)
        print(f"最后一帧已保存: {args.snapshot}")


def main() -> None:
    parser = argparse.ArgumentParser(description="分拣视觉：颜色+形状+姿态")
    parser.add_argument("images", nargs="+", help="输入图片")
    parser.add_argument("--out", help="标注结果输出路径（仅单张输入时可用）")
    parser.add_argument("--json", help="检测结果 JSON 输出路径")
    parser.add_argument("--calib", help="标定文件（calibrate.py 生成）")
    parser.add_argument("--chroma-min", type=float, default=None)
    parser.add_argument("--max-width", type=int, default=None)
    parser.add_argument(
        "--roi",
        help="工作区矩形 x,y,w,h（像素，按处理后的尺寸）",
    )
    parser.add_argument(
        "--roi-poly",
        help="工作区多边形 x1,y1;x2,y2;...（与 --roi 二选一）",
    )
    parser.add_argument(
        "--url",
        help="视频流地址（如 ESP32-S3 的 http://192.168.4.1:81/stream），"
        "给定后忽略图片参数，进入实时识别",
    )
    parser.add_argument("--frames", type=int, default=0, help="流模式下处理帧数，0=一直跑")
    parser.add_argument("--snapshot", help="流模式：把最后一帧标注结果存到这里")
    parser.add_argument("--debug", action="store_true", help="输出额外调试信息")
    args = parser.parse_args()

    cfg = Config()
    if args.chroma_min is not None:
        cfg.chroma_min = args.chroma_min
    if args.max_width is not None:
        cfg.max_width = args.max_width
    if args.roi:
        cfg.roi = tuple(float(v) for v in args.roi.replace("，", ",").split(","))
    if args.roi_poly:
        cfg.roi_poly = [
            tuple(float(v) for v in pt.split(","))
            for pt in args.roi_poly.split(";")
            if pt.strip()
        ]
    calib = load_calibration(args.calib) if args.calib else None

    if args.url:
        run_stream(args.url, cfg, calib, args)
        return

    all_results = {}
    for path in args.images:
        bgr, dets = process(path, cfg, calib)
        print(f"{path}  ({bgr.shape[1]}x{bgr.shape[0]})  检出 {len(dets)} 个物体")
        print_table(dets)
        if args.debug:
            for i, d in enumerate(dets, 1):
                print(
                    f"    #{i} contour_pts={len(d.contour)} "
                    f"box={np.round(d.box_poly, 1).tolist()}"
                )
        all_results[path] = [d.to_dict() for d in dets]
        if args.out and len(args.images) == 1:
            cv2.imwrite(args.out, annotate(bgr, dets, cfg))
            print(f"  标注结果已保存: {args.out}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON 已保存: {args.json}")


if __name__ == "__main__":
    main()
