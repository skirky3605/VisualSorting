# 智能分拣（视觉识别 + 抓取姿态）

ESP32-S3 出图 → PC 端识别颜色/形状 → 输出机械臂抓取姿态（工作台 mm + 偏航角）。

## 目录结构

| 路径 | 说明 |
|---|---|
| `vision/` | 视觉算法代码（主程序、标定、视角评估、调参工具），详见 [vision/README.md](vision/README.md) |
| `test1.jpg` ~ `test4.jpg` | 四个候选机位的样张（同一批 7 件物料） |
| `out/` | 评估与标注结果输出（标注图、`views.csv`、`views.json`） |
| `legacy/main_v0_hsv.py` | 早期 HSV 阈值版本脚本（已归档，被 `vision/sorter_vision.py` 取代） |
| `.vscode/` | 编辑器配置（conda 环境管理） |

## 环境

conda 环境 `py310`（OpenCV 5.0 / numpy / scipy / torch-cpu），已装好，直接用：

```powershell
$py = "C:\Users\skirky\miniconda3\envs\py310\python.exe"
```

## 快速开始

```powershell
# 1) 单张图片识别 + 标注
& $py vision\sorter_vision.py test1.jpg --out out\test1_annotated.jpg --json out\test1.json

# 2) 四个候选机位定量比较（内置真值）
& $py vision\evaluate_views.py test1.jpg test2.jpg test3.jpg test4.jpg `
      --out-dir out --csv out\views.csv --json out\views.json

# 3) 真实工位标定（4 个角点像素 + 标定板实际尺寸）→ 输出工作台坐标与偏航角
& $py vision\calibrate.py --points "300,700;1180,690;1210,120;320,110" `
      --size-mm "520,430" --out calib.json --check test1.jpg --check-out out\grid.png

# 4) 接 ESP32-S3 视频流实时识别
& $py vision\sorter_vision.py --url http://192.168.4.1:81/stream --frames 300 --snapshot out\snap.png
```

> 在项目根目录用相对路径运行；PowerShell 把中文路径当参数传给 python.exe 会乱码。

## 四个机位的评估结论（详见 vision/README.md 第 5 节）

| 机位 | 检出 | 分类正确 | 工作区内误检 | 透视收敛比 | 备注 |
|---|---|---|---|---|---|
| **test1 近正俯视** | **7/7** | 6/7 | **0** | **1.143** | 推荐主相机位；三棱柱需靠小倾角/补光区分 |
| test3 中俯视≈40° | 7/7 | **7/7** | 0 | 1.155 | 识别最稳，但分辨率余量低、物料相邻会粘连 |
| test2 低角度大视野 | 6/7 | 5/7 | 0（区外 4 个误检） | 1.374 | 不推荐：物料粘连 + 背景杂物 |
| test4 低角度近拍 | 7/7 | 4/7 | 0（区外 1 个误检） | 1.503 | 不推荐：沿视线方向尺寸被压缩 39% |

**结论：相机装在 test1 的位置（近正俯视），保留 10°~25° 小倾角**，几何失真最小、
姿态最可信；三棱柱的歧义用倾斜露脊线或单侧补光 + 明暗双峰判据解决。
