# -*- coding: utf-8 -*-
# rp_stats.py
import os
import re
import numpy as np
import pandas as pd
from openpyxl import Workbook  # 仅用于 xlsx 引擎，可留可去

# --------- 读取 topo（沿用你的写法）---------
def read_ascii_topo(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 自动检测数据起点（跳过前面的非数值 header）
    data_start_index = 0
    for i, line in enumerate(lines[:10]):
        parts = line.strip().split()
        if all(part.replace('.', '', 1).replace('-', '', 1).isdigit() for part in parts):
            data_start_index = i
            break

    # 加载高程矩阵数据
    data = np.loadtxt(lines[data_start_index:])

    # 默认 header（如需真实 cellsize/原点，请在此解析）
    header = {
        "cellsize": 1.0,
        "xllcorner": 0.0,
        "yllcorner": 0.0,
    }
    return header, data


# --------- rp 曲线 ----------
def calculate_rp_curve(header, elevation_data, step=1.0):
    cellsize = header["cellsize"]
    nrows, ncols = elevation_data.shape
    total_area = nrows * ncols * (cellsize ** 2)

    max_height = float(np.max(elevation_data))
    heights = np.arange(0.0, max_height + 1e-9, step)  # 包含最大高度

    rp_values = []
    for h in heights:
        mask = (elevation_data >= h)
        building_area = np.sum(mask) * (cellsize ** 2)
        rp = building_area / total_area if total_area > 0 else np.nan
        rp_values.append(rp)

    return np.asarray(rp_values, dtype=float), heights, max_height


# --------- 计算 rp_ave / rp_std（两种方案）----------
def rp_stats_step1m(header, elevation_data, step=1.0):
    """
    以 1m（或给定 step）分辨率：z=0, step, 2*step, ... , Hmax
    返回：rp_ave, rp_std, rp_curve, z_levels
    """
    rp_curve, z_levels, _ = calculate_rp_curve(header, elevation_data, step=step)
    rp_ave = float(np.nanmean(rp_curve))
    rp_std = float(np.nanstd(rp_curve, ddof=0))
    return rp_ave, rp_std, rp_curve, z_levels


def rp_stats_equal8(header, elevation_data):
    """
    将 [0, Hmax] 等分为 8 个 z 点，计算 rp(z)；
    返回：rp_ave, rp_std, rp_curve(8), z_levels(8)
    """
    cellsize = header["cellsize"]
    nrows, ncols = elevation_data.shape
    total_area = nrows * ncols * (cellsize ** 2)

    Hmax = float(np.max(elevation_data))
    if Hmax <= 0:
        # 无建筑高度时：rp(z)=1（只有 z=0）
        z_levels = np.linspace(0.0, 0.0, 8)
        rp_vals = np.ones_like(z_levels, dtype=float)
        return float(np.nanmean(rp_vals)), float(np.nanstd(rp_vals, ddof=0)), rp_vals, z_levels

    z_levels = np.linspace(0.0, Hmax, 8, endpoint=True)
    rp_vals = []
    for z in z_levels:
        mask = (elevation_data >= z)
        building_area = np.sum(mask) * (cellsize ** 2)
        rp_vals.append(building_area / total_area if total_area > 0 else np.nan)
    rp_vals = np.asarray(rp_vals, dtype=float)

    rp_ave = float(np.nanmean(rp_vals))
    rp_std = float(np.nanstd(rp_vals, ddof=0))
    return rp_ave, rp_std, rp_vals, z_levels


# --------- 逐层 rf 曲线（对所有建筑截面积累计）----------
def calculate_rf_curve(header, elevation_data, z_levels, wind_axis="x"):
    """
    根据 topo 和给定 z_levels 计算逐层 rf(z)：

    在高度区间 [z_k, z_k+Δz_k) 内：
      - 对所有建筑在该风向上的“截面积”（投影截面）进行累计：
        先沿风向把高程矩阵压缩成一维的 max 高度 h_line，
        再计算每条扫描线在这一层的截面高度 overlap，
        Af(z_k) = Σ_j overlap_j * cellsize
      - rf(z_k) = Af(z_k) / (A_ground * Δz_k)

    wind_axis:
      - "x":   风沿 topo 的 x 方向吹（列方向），截面在 yz 平面 -> 每一行(y)一条扫描线
      - "y":   风沿 topo 的 y 方向吹（行方向），截面在 xz 平面 -> 每一列(x)一条扫描线
    """
    H = np.asarray(elevation_data, dtype=float)
    cellsize = float(header["cellsize"])
    nrows, ncols = H.shape
    ground_area = nrows * ncols * (cellsize ** 2)

    z_levels = np.asarray(z_levels, dtype=float)
    N = len(z_levels)
    if N == 0 or ground_area <= 0:
        return np.zeros_like(z_levels, dtype=float)

    # 每层厚度 Δz_k
    dz = np.zeros(N, dtype=float)
    if N == 1:
        dz[0] = 1.0
    else:
        dz[:-1] = np.diff(z_levels)
        dz[-1] = dz[-2]

    # 按风向选扫描线方向
    if wind_axis == "y":
        # 风沿 y 方向吹：沿行方向压缩，截面在 xz 平面
        h_line = np.max(H, axis=0)   # shape = (ncols,)
    else:
        # 默认风沿 x 方向吹：沿列方向压缩，截面在 yz 平面
        h_line = np.max(H, axis=1)   # shape = (nrows,)

    rf_vals = np.zeros(N, dtype=float)

    for k in range(N):
        z_k = z_levels[k]
        z_top_k = z_k + dz[k]

        # 每条扫描线在本层的截面高度：
        # overlap_j = max(0, min(h_line[j], z_top_k) - z_k)
        overlap = np.minimum(h_line, z_top_k) - z_k
        overlap = np.clip(overlap, a_min=0.0, a_max=None)

        # 本层总截面积：所有扫描线的 overlap 高度之和 × 单条线宽度 cellsize
        Af_k = np.sum(overlap) * cellsize

        if dz[k] > 0 and ground_area > 0:
            rf_vals[k] = Af_k / (ground_area * dz[k])
        else:
            rf_vals[k] = np.nan

    return rf_vals


# --------- 提取风向角（保持你的写法）----------
def extract_wind_angle(folder_name):
    match = re.search(r'(\d{2,3})$', folder_name)
    return int(match.group(1)) if match else 0


# --------- 读取 profile-U 数据 ----------
def read_profile_U(folder_path, folder_name):
    """
    读取每个案例文件夹下的 profile-<folder_name>.csv，
    按第一列（序号）排序，返回 U 列的值（np.ndarray 一维）。
    如果文件不存在，则返回 None。
    """
    profile_name = f"profile-{folder_name}.csv"
    profile_path = os.path.join(folder_path, profile_name)
    if not os.path.exists(profile_path):
        print(f"[WARN] 未找到 profile 文件: {profile_path}，该案例 U_* 将填 NaN")
        return None

    # 默认按逗号分隔，自动识别编码
    df = pd.read_csv(profile_path)

    # 第一列视为序号，用于排序
    first_col = df.columns[0]
    df = df.sort_values(by=first_col)

    if "U" not in df.columns:
        raise ValueError(f"{profile_path} 中未找到 'U' 列，请检查文件格式。")

    U_vals = df["U"].to_numpy(dtype=float)
    return U_vals


# --------- 批量处理（新增 rp / rf + U_* 输出）----------
def process_all_topos(root_dir, output_excel, method="step1m", step=1.0):
    """
    遍历 root_dir 下的案例文件夹：
      - topo 文件假定为 <folder>_topo
      - profile 文件为 profile-<folder>.csv，读取 U 列
      - method: "step1m" | "equal8"
      - 输出：
        基础信息 +
        rp_0..rp_N-1, rf_0..rf_N-1 +
        rp_ave, rp_std, rf_ave, rf_std +
        U_1..U_M
    """
    rows = []
    max_levels = 0      # rp / rf 层数最大值
    max_U_len = 0       # U 序列最大长度

    # 第一遍：收集数据，确定最大层数 / 最大 U 长度
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        topo_filename = f"{folder}_topo"
        topo_path = os.path.join(folder_path, topo_filename)

        header, elevation = read_ascii_topo(topo_path)
        wind_angle = extract_wind_angle(folder)

        if method == "equal8":
            rp_ave, rp_std, rp_curve, z_levels = rp_stats_equal8(header, elevation)
        else:  # "step1m"
            rp_ave, rp_std, rp_curve, z_levels = rp_stats_step1m(header, elevation, step=step)

        # 计算逐层 rf（与 rp 使用同一套 z_levels）
        # 这里默认 topo 已按风向旋转好，风沿 topo 的 x 方向吹
        rf_curve = calculate_rf_curve(header, elevation, z_levels, wind_axis="x")
        rf_ave = float(np.nanmean(rf_curve))
        rf_std = float(np.nanstd(rf_curve, ddof=0))

        # 读取 profile-U
        U_vals = read_profile_U(folder_path, folder)

        rows.append({
            "ID": folder,
            "风向": wind_angle,
            "H_max": float(np.max(elevation)),
            "levels(步数)": int(len(rp_curve)),
            "step": float(step) if method == "step1m" else np.nan,
            "rp_curve": rp_curve,
            "rp_ave": float(rp_ave),
            "rp_std": float(rp_std),
            "rf_curve": rf_curve,
            "rf_ave": float(rf_ave),
            "rf_std": float(rf_std),
            "U_vals": U_vals,
        })

        if len(rp_curve) > max_levels:
            max_levels = len(rp_curve)
        if U_vals is not None and len(U_vals) > max_U_len:
            max_U_len = len(U_vals)

    # 构建列名：rp_0..rp_{max_levels-1}, rf_0..rf_{max_levels-1}, U_1..U_{max_U_len}
    base_cols = ["ID", "风向", "H_max", "levels(步数)", "step"]
    rp_cols   = [f"rp_{i}" for i in range(max_levels)]
    rf_cols   = [f"rf_{i}" for i in range(max_levels)]
    U_cols    = [f"U_{i+1}" for i in range(max_U_len)]  # U 从 1 开始
    col_names = (
        base_cols
        + ["rp_ave", "rp_std", "rf_ave", "rf_std"]
        + rp_cols
        + rf_cols
        + U_cols
    )

    # 第二遍：展开 rp_* / rf_* / U_*
    records = []
    for item in rows:
        rec = {k: item[k] for k in base_cols}
        rec["rp_ave"] = item["rp_ave"]
        rec["rp_std"] = item["rp_std"]
        rec["rf_ave"] = item["rf_ave"]
        rec["rf_std"] = item["rf_std"]

        # rp_* 展开
        rp_vals = item["rp_curve"]
        for i in range(max_levels):
            rec[f"rp_{i}"] = rp_vals[i] if i < len(rp_vals) else np.nan

        # rf_* 展开
        rf_vals = item["rf_curve"]
        for i in range(max_levels):
            rec[f"rf_{i}"] = rf_vals[i] if i < len(rf_vals) else np.nan

        # U_* 展开（列名是 U_1..U_M）
        U_vals = item["U_vals"]
        for i in range(max_U_len):
            col_name = f"U_{i+1}"
            if U_vals is not None and i < len(U_vals):
                rec[col_name] = U_vals[i]
            else:
                rec[col_name] = np.nan

        records.append(rec)

    df = pd.DataFrame.from_records(records, columns=col_names)
    df.to_excel(output_excel, index=False)