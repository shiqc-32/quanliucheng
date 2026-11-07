# urban_morphology_summary.py
import os
import re
import glob
import time
import numpy as np
import pandas as pd
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator


# =========================================================
# 基础 IO
# =========================================================

def _auto_read_csv(path, sep=",", encoding=None):
    if encoding is not None:
        return pd.read_csv(path, sep=sep, encoding=encoding)

    last_err = None
    for enc in ["utf-8", "utf-8-sig", "gb18030"]:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _find_case_name(cfg, target_case="urban_morphology_summary"):
    if not isinstance(cfg, dict):
        raise ValueError("YAML 顶层必须是字典。")
    if target_case not in cfg:
        raise KeyError(f"YAML 中未找到 case: {target_case}")
    return target_case


def _get_case_cfg(cfg, target_case="urban_morphology_summary"):
    case_name = _find_case_name(cfg, target_case=target_case)
    case_cfg = cfg[case_name]
    if not isinstance(case_cfg, dict):
        raise ValueError(f"case '{case_name}' 的内容必须是字典。")
    return case_cfg


# =========================================================
# topo 读取
# =========================================================

def _load_topo_array(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)

    elif ext in [".txt", ".dat", ".asc", ".csv"]:
        try:
            if ext == ".csv":
                arr = np.loadtxt(path, delimiter=",")
            else:
                arr = np.loadtxt(path)
        except Exception:
            arr = np.loadtxt(path, delimiter=",")

    elif ext == "":
        try:
            arr = np.loadtxt(path)
        except Exception:
            try:
                arr = np.loadtxt(path, delimiter=",")
            except Exception as e:
                raise ValueError(f"无法按文本格式读取无扩展名 topo 文件: {path}") from e
    else:
        raise ValueError(f"暂不支持的 topo 文件格式: {path}")

    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"topo 必须是二维数组: {path}")
    return arr


def _find_topo_file(case_dir, folder_name, topo_suffix="_topo"):
    exact_no_ext = os.path.join(case_dir, f"{folder_name}{topo_suffix}")
    if os.path.isfile(exact_no_ext):
        return exact_no_ext

    patterns = [
        os.path.join(case_dir, f"{folder_name}{topo_suffix}.npy"),
        os.path.join(case_dir, f"{folder_name}{topo_suffix}.txt"),
        os.path.join(case_dir, f"{folder_name}{topo_suffix}.dat"),
        os.path.join(case_dir, f"{folder_name}{topo_suffix}.asc"),
        os.path.join(case_dir, f"{folder_name}{topo_suffix}.csv"),
    ]

    for p in patterns:
        if os.path.isfile(p):
            return p

    fuzzy = glob.glob(os.path.join(case_dir, f"{folder_name}{topo_suffix}*"))
    fuzzy = [p for p in fuzzy if os.path.isfile(p)]
    if len(fuzzy) == 1:
        return fuzzy[0]
    if len(fuzzy) > 1:
        raise FileExistsError(f"找到多个 topo 文件，请手动清理: {fuzzy}")

    raise FileNotFoundError(f"未找到 topo 文件: {folder_name}{topo_suffix}")


def _build_valid_mask(H, nodata_value=None):
    H = np.asarray(H, dtype=float)
    valid_mask = np.isfinite(H)
    if nodata_value is not None:
        valid_mask &= (H != nodata_value)
    return valid_mask


def _build_street_mask(H, valid_mask):
    return valid_mask & (H <= 0)


def _extract_wind_dir_from_folder(folder_name):
    m = re.search(r"(\d{2})$", str(folder_name))
    if m is None:
        raise ValueError(f"无法从文件夹名末两位提取风向: {folder_name}")
    return float(int(m.group(1)))


# =========================================================
# 纯计算函数
# =========================================================

def _as_float_array(H):
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        raise ValueError("H must be a 2D array.")
    return H


def _make_valid_mask(H, valid_mask=None):
    H = _as_float_array(H)
    if valid_mask is None:
        return np.ones_like(H, dtype=bool)

    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != H.shape:
        raise ValueError("valid_mask must have the same shape as H.")
    return valid_mask


def _cell_area_from_mask(valid_mask, dx, dy):
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be > 0.")
    return float(np.count_nonzero(valid_mask) * dx * dy)


def _building_mask(H, valid_mask=None):
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)
    return (H > 0) & valid_mask


def _label_buildings_8(H, valid_mask=None):
    H = _as_float_array(H)
    bmask = _building_mask(H, valid_mask)
    structure = ndimage.generate_binary_structure(2, 2)
    labels, num = ndimage.label(bmask, structure=structure)
    return labels, num


def _representative_height_hmax(H_obj):
    vals = np.asarray(H_obj, dtype=float)
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    return float(np.max(vals))


def _pixel_corner_coords(mask_obj, dx, dy):
    rows, cols = np.where(mask_obj)
    if rows.size == 0:
        return np.empty((0, 2), dtype=float)

    corners = []
    for r, c in zip(rows, cols):
        x0 = c * dx
        x1 = (c + 1) * dx
        y1 = -r * dy
        y0 = -(r + 1) * dy
        corners.extend([
            (x0, y0),
            (x0, y1),
            (x1, y0),
            (x1, y1),
        ])
    return np.asarray(corners, dtype=float)


def _projected_width(mask_obj, dx, dy, wind_dir_deg):
    corners = _pixel_corner_coords(mask_obj, dx, dy)
    if corners.shape[0] == 0:
        return 0.0

    theta = np.deg2rad(wind_dir_deg)
    normal = np.array([-np.sin(theta), np.cos(theta)], dtype=float)

    proj = corners @ normal
    width = float(np.max(proj) - np.min(proj))
    return width


def _extract_building_metrics(H, dx, dy, valid_mask=None, wind_dir_deg=0.0):
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)

    labels, num = _label_buildings_8(H, valid_mask=valid_mask)
    pixel_area = dx * dy
    metrics = []

    for lab in range(1, num + 1):
        obj_mask = (labels == lab)
        H_obj = H[obj_mask]

        K = float(np.count_nonzero(obj_mask) * pixel_area)
        h = _representative_height_hmax(H_obj)
        V = float(np.sum(H_obj[H_obj > 0]) * pixel_area)
        w = _projected_width(obj_mask, dx, dy, wind_dir_deg)
        S = float(h * w)

        metrics.append({
            "label": lab,
            "K": K,
            "h": h,
            "V": V,
            "w": w,
            "S": S,
        })

    return metrics


def calc_building_density_A(H, dx, dy, valid_mask=None):
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)

    X = _cell_area_from_mask(valid_mask, dx, dy)
    if X <= 0:
        raise ValueError("Cell area X must be > 0.")

    K_sum = float(np.count_nonzero((H > 0) & valid_mask) * dx * dy)
    return K_sum / X

def _calc_lp_debug(H, valid_mask, lp_ori=None):
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)

    n_total = int(H.size)
    n_valid = int(np.count_nonzero(valid_mask))
    n_building = int(np.count_nonzero((H > 0) & valid_mask))
    n_zero = int(np.count_nonzero((H == 0) & valid_mask))
    n_negative = int(np.count_nonzero((H < 0) & valid_mask))
    n_nodata = int(n_total - n_valid)

    A_manual = np.nan if n_valid == 0 else (n_building / n_valid)

    uniq_vals = np.unique(H[valid_mask]) if n_valid > 0 else np.array([])
    if uniq_vals.size <= 10:
        uniq_summary = ",".join(map(lambda x: f"{x:g}", uniq_vals))
    else:
        uniq_summary = f"min={np.min(uniq_vals):g}, max={np.max(uniq_vals):g}, n_unique={uniq_vals.size}"

    out = {
        "dbg_shape": f"{H.shape[0]}x{H.shape[1]}",
        "dbg_total_cells": n_total,
        "dbg_valid_cells": n_valid,
        "dbg_building_cells": n_building,
        "dbg_zero_cells": n_zero,
        "dbg_negative_cells": n_negative,
        "dbg_nodata_cells": n_nodata,
        "dbg_A_manual": A_manual,
        "dbg_unique_vals": uniq_summary,
    }

    if lp_ori is not None and pd.notna(lp_ori):
        out["dbg_lp_ori"] = float(lp_ori)
        out["dbg_A_minus_lp_ori"] = float(A_manual - float(lp_ori))
        out["dbg_abs_A_minus_lp_ori"] = float(abs(A_manual - float(lp_ori)))
    else:
        out["dbg_lp_ori"] = np.nan
        out["dbg_A_minus_lp_ori"] = np.nan
        out["dbg_abs_A_minus_lp_ori"] = np.nan

    return out


def calc_building_aspect_ratio_B(H, dx, dy, valid_mask=None):
    H = _as_float_array(H)
    metrics = _extract_building_metrics(
        H=H, dx=dx, dy=dy, valid_mask=valid_mask, wind_dir_deg=0.0
    )
    if len(metrics) == 0:
        return 0.0

    S_sum = float(sum(m["S"] for m in metrics))
    K_sum = float(sum(m["K"] for m in metrics))
    if K_sum <= 0:
        return 0.0
    return S_sum / K_sum


def calc_frontal_area_density_C(H, dx, dy, wind_dir_deg, valid_mask=None):
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)

    X = _cell_area_from_mask(valid_mask, dx, dy)
    if X <= 0:
        raise ValueError("Cell area X must be > 0.")

    metrics = _extract_building_metrics(
        H=H, dx=dx, dy=dy, valid_mask=valid_mask, wind_dir_deg=wind_dir_deg
    )
    if len(metrics) == 0:
        return 0.0

    S_sum = float(sum(m["S"] for m in metrics))
    return S_sum / X


def calc_compactness_D(H, dx, dy, valid_mask=None):
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)

    X = _cell_area_from_mask(valid_mask, dx, dy)
    if X <= 0:
        raise ValueError("Cell area X must be > 0.")

    H_use = np.where(valid_mask, H, 0.0)
    H_use = np.where(H_use > 0, H_use, 0.0)

    V_sum = float(np.sum(H_use) * dx * dy)
    return V_sum / X


def calc_porosity_E_formula(H, dx, dy, valid_mask=None):
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)

    X = _cell_area_from_mask(valid_mask, dx, dy)
    if X <= 0:
        raise ValueError("Cell area X must be > 0.")

    H_use = np.where(valid_mask, H, 0.0)
    H_use = np.where(H_use > 0, H_use, 0.0)

    if np.count_nonzero(H_use > 0) == 0:
        return 0.0

    V_sum = float(np.sum(H_use) * dx * dy)
    h_max = float(np.max(H_use))
    if h_max <= 0:
        return 0.0

    return V_sum / (X * h_max)


def calc_height_variation_F(H, dx, dy, valid_mask=None):
    H = _as_float_array(H)
    metrics = _extract_building_metrics(
        H=H, dx=dx, dy=dy, valid_mask=valid_mask, wind_dir_deg=0.0
    )

    heights = np.array([m["h"] for m in metrics], dtype=float)
    n = heights.size

    if n == 0 or n == 1:
        return np.nan

    return float(np.std(heights, ddof=1))


@njit(cache=True)
def _scan_max_elevation_angle_core(
    H,
    row0,
    col0,
    theta_rad,
    dx,
    dy,
    max_radius,
    step_length,
    base_height,
):
    nrow, ncol = H.shape

    x0 = (col0 + 0.5) * dx
    y0 = -(row0 + 0.5) * dy

    nsteps = int(np.floor(max_radius / step_length))
    max_angle = 0.0

    last_rr = -10**9
    last_cc = -10**9

    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    for k in range(1, nsteps + 1):
        s = k * step_length
        x = x0 + s * cos_t
        y = y0 + s * sin_t

        cc = int(np.floor(x / dx))
        rr = int(np.floor(-y / dy))

        if rr < 0 or rr >= nrow or cc < 0 or cc >= ncol:
            break

        if rr == last_rr and cc == last_cc:
            continue
        last_rr = rr
        last_cc = cc

        hj = H[rr, cc]
        if hj <= base_height:
            continue

        alpha = np.arctan2(hj - base_height, s)
        if alpha > max_angle:
            max_angle = alpha

    return max_angle


def scan_max_elevation_angle(
    H,
    row0,
    col0,
    theta_rad,
    dx,
    dy,
    max_radius,
    step_length=None,
    base_height=0.0,
):
    H = np.ascontiguousarray(_as_float_array(H), dtype=np.float64)

    nrow, ncol = H.shape
    if row0 < 0 or row0 >= nrow or col0 < 0 or col0 >= ncol:
        raise IndexError("row0/col0 out of bounds.")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be > 0.")
    if max_radius <= 0:
        raise ValueError("max_radius must be > 0.")

    if step_length is None:
        step_length = min(dx, dy) / 2.0
    step_length = float(step_length)

    if step_length <= 0:
        raise ValueError("step_length must be > 0.")

    return float(
        _scan_max_elevation_angle_core(
            H=H,
            row0=int(row0),
            col0=int(col0),
            theta_rad=float(theta_rad),
            dx=float(dx),
            dy=float(dy),
            max_radius=float(max_radius),
            step_length=step_length,
            base_height=float(base_height),
        )
    )


@njit(cache=True)
def _calc_sky_openness_G_for_point_core(
    H,
    row0,
    col0,
    dx,
    dy,
    n_dirs,
    max_radius,
    step_length,
    base_height,
):
    gamma = np.zeros(n_dirs, dtype=np.float64)

    for i in range(n_dirs):
        theta = 2.0 * np.pi * i / n_dirs
        gamma[i] = _scan_max_elevation_angle_core(
            H=H,
            row0=row0,
            col0=col0,
            theta_rad=theta,
            dx=dx,
            dy=dy,
            max_radius=max_radius,
            step_length=step_length,
            base_height=base_height,
        )

    G = 1.0 - np.mean(np.sin(gamma))
    return G, gamma


def calc_sky_openness_G_for_point(
    H,
    row0,
    col0,
    dx,
    dy,
    n_dirs=72,
    max_radius=100.0,
    step_length=None,
    base_height=0.0,
):
    H = np.ascontiguousarray(_as_float_array(H), dtype=np.float64)

    nrow, ncol = H.shape
    if row0 < 0 or row0 >= nrow or col0 < 0 or col0 >= ncol:
        raise IndexError("row0/col0 out of bounds.")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be > 0.")
    if n_dirs <= 0:
        raise ValueError("n_dirs must be > 0.")
    if max_radius <= 0:
        raise ValueError("max_radius must be > 0.")

    if step_length is None:
        step_length = min(dx, dy) / 2.0
    step_length = float(step_length)

    if step_length <= 0:
        raise ValueError("step_length must be > 0.")

    G, gamma = _calc_sky_openness_G_for_point_core(
        H=H,
        row0=int(row0),
        col0=int(col0),
        dx=float(dx),
        dy=float(dy),
        n_dirs=int(n_dirs),
        max_radius=float(max_radius),
        step_length=step_length,
        base_height=float(base_height),
    )
    return float(G), np.asarray(gamma, dtype=float)


def calc_sky_openness_G_map(
    H,
    street_mask,
    dx,
    dy,
    n_dirs=72,
    max_radius=100.0,
    step_length=None,
    base_height=0.0,
):
    H = np.ascontiguousarray(_as_float_array(H), dtype=np.float64)
    street_mask = np.asarray(street_mask, dtype=bool)

    if street_mask.shape != H.shape:
        raise ValueError("street_mask must have the same shape as H.")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be > 0.")
    if n_dirs <= 0:
        raise ValueError("n_dirs must be > 0.")
    if max_radius <= 0:
        raise ValueError("max_radius must be > 0.")

    if step_length is None:
        step_length = min(dx, dy) / 2.0
    step_length = float(step_length)

    if step_length <= 0:
        raise ValueError("step_length must be > 0.")

    nrow, ncol = H.shape
    G_map = np.full((nrow, ncol), np.nan, dtype=float)

    rows, cols = np.where(street_mask)
    for r, c in zip(rows, cols):
        G_val, _ = _calc_sky_openness_G_for_point_core(
            H=H,
            row0=int(r),
            col0=int(c),
            dx=float(dx),
            dy=float(dy),
            n_dirs=int(n_dirs),
            max_radius=float(max_radius),
            step_length=step_length,
            base_height=float(base_height),
        )
        G_map[r, c] = float(G_val)

    return G_map, None


def _extract_building_boundaries(H, valid_mask=None):
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)

    labels, num = _label_buildings_8(H, valid_mask=valid_mask)
    boundary_masks = {}
    structure = ndimage.generate_binary_structure(2, 2)

    for lab in range(1, num + 1):
        obj_mask = (labels == lab)
        eroded = ndimage.binary_erosion(obj_mask, structure=structure, border_value=0)
        boundary = obj_mask & (~eroded)
        boundary_masks[lab] = boundary

    return labels, boundary_masks


def _mask_to_center_coords(mask_obj, dx, dy):
    rows, cols = np.where(mask_obj)
    if rows.size == 0:
        return np.empty((0, 2), dtype=float)

    x = (cols + 0.5) * dx
    y = -(rows + 0.5) * dy
    return np.column_stack([x, y]).astype(float)


def _pairwise_min_distance(coords_a, coords_b):
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return np.nan

    diff = coords_a[:, None, :] - coords_b[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    return float(np.sqrt(np.min(dist2)))


def calc_building_neighbor_distance_matrix(H, dx, dy, valid_mask=None):
    H = _as_float_array(H)
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be > 0.")

    labels, boundary_masks = _extract_building_boundaries(H, valid_mask=valid_mask)
    building_ids = np.array(sorted(boundary_masks.keys()), dtype=int)
    n = building_ids.size

    D = np.full((n, n), np.nan, dtype=float)
    if n == 0:
        return labels, building_ids, D

    boundary_coords = {
        bid: _mask_to_center_coords(boundary_masks[bid], dx, dy)
        for bid in building_ids
    }

    for i in range(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            bid_i = building_ids[i]
            bid_j = building_ids[j]
            dij = _pairwise_min_distance(boundary_coords[bid_i], boundary_coords[bid_j])
            D[i, j] = dij
            D[j, i] = dij

    return labels, building_ids, D


def calc_building_neighbor_proximity_per_building(H, dx, dy, valid_mask=None):
    labels, building_ids, D = calc_building_neighbor_distance_matrix(
        H=H, dx=dx, dy=dy, valid_mask=valid_mask
    )

    n = building_ids.size
    result = []

    if n == 0:
        return result, D
    if n == 1:
        result.append({"label": int(building_ids[0]), "d_bar": np.nan})
        return result, D

    for i, bid in enumerate(building_ids):
        row = D[i, :]
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        d_bar = float(np.mean(row[mask]))
        result.append({"label": int(bid), "d_bar": d_bar})

    return result, D


def calc_building_neighbor_proximity_H(H, dx, dy, valid_mask=None):
    per_building, D = calc_building_neighbor_proximity_per_building(
        H=H, dx=dx, dy=dy, valid_mask=valid_mask
    )

    n = len(per_building)
    if n == 0 or n == 1:
        return np.nan, per_building, D

    d_bars = np.array([item["d_bar"] for item in per_building], dtype=float)
    H_neighbor = float(np.mean(d_bars))
    return H_neighbor, per_building, D


# =========================================================
# 参数读取与执行
# =========================================================

def _prepare_metadata_table(case_cfg):
    inp = case_cfg["input"]
    cols = case_cfg["columns"]

    sep = inp.get("sep", ",")
    encoding = inp.get("encoding", None)

    metadata = _auto_read_csv(inp["metadata_csv"], sep=sep, encoding=encoding)
    flowdata = _auto_read_csv(inp["flowdata_csv"], sep=sep, encoding=encoding)
    mapping = _auto_read_csv(inp["case_name_mapping_csv"], sep=sep, encoding=encoding)
    urban = _auto_read_csv(inp["urban_model_input_csv"], sep=sep, encoding=encoding)

    metadata_sub = metadata[[
        cols["metadata_name"],
        cols["metadata_config"],
        cols["metadata_lp"],
        cols["metadata_lf"],
        cols["metadata_hmean"],
        cols["metadata_hstd"],
        cols["metadata_hmax"],
        cols["metadata_dxdy"],
    ]].copy()

    metadata_sub["fg"] = metadata_sub[cols["metadata_config"]].astype(str).str.strip().str.lower().str[0].map({
        "i": 0,
        "r": 1,
    })

    metadata_sub = metadata_sub.rename(columns={
        cols["metadata_name"]: "NameE",
        cols["metadata_lp"]: "lp_ori",
        cols["metadata_lf"]: "lf_ori",
        cols["metadata_hmean"]: "have_ori",
        cols["metadata_hstd"]: "hstd_ori",
        cols["metadata_hmax"]: "hmax_ori",
        cols["metadata_dxdy"]: "dxdy",
    })

    metadata_sub = metadata_sub[[
        "NameE", "fg", "lp_ori", "lf_ori", "have_ori", "hstd_ori", "hmax_ori", "dxdy"
    ]]

    mapping_sub = mapping[[
        cols["mapping_meta_name"],
        cols["mapping_folder_score"],
    ]].copy().rename(columns={
        cols["mapping_meta_name"]: "meta_name",
        cols["mapping_folder_score"]: "folder_score",
    })

    urban_sub = urban[[
        cols["urban_case_name"],
        cols["urban_rp_std"],
        cols["urban_rp_ave"],
    ]].copy().rename(columns={
        cols["urban_case_name"]: "meta_name",
        cols["urban_rp_std"]: "rp_std",
        cols["urban_rp_ave"]: "rp_ave",
    })

    flowdata_sub = flowdata[[
        cols["flowdata_name"],
        cols["flowdata_col1"],
        cols["flowdata_col2"],
    ]].copy().rename(columns={
        cols["flowdata_name"]: "NameE",
        cols["flowdata_col1"]: "d",
        cols["flowdata_col2"]: "z0",
    })

    df = mapping_sub.merge(metadata_sub, left_on="meta_name", right_on="NameE", how="left")
    df = df.merge(urban_sub, on="meta_name", how="left")
    df = df.merge(flowdata_sub, on="NameE", how="left")
    return df


def _resolve_case_folder(folder_score, ideal_root, real_root):
    folder_name = str(folder_score).strip()
    ideal_dir = os.path.join(ideal_root, folder_name)
    real_dir = os.path.join(real_root, folder_name)

    if os.path.isdir(ideal_dir) and os.path.isdir(real_dir):
        raise FileExistsError(f"ideal 和 real 同时存在同名 case: {folder_name}")
    if os.path.isdir(ideal_dir):
        return ideal_dir
    if os.path.isdir(real_dir):
        return real_dir

    raise FileNotFoundError(f"在 ideal/real 下均未找到 case 文件夹: {folder_name}")


def _calc_g_stats(G_map, street_mask):
    vals = G_map[street_mask]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {
            "G_25": np.nan,
            "G_50": np.nan,
            "G_75": np.nan,
            "G_std": np.nan,
        }

    return {
        "G_25": float(np.percentile(vals, 25)),
        "G_50": float(np.percentile(vals, 50)),
        "G_75": float(np.percentile(vals, 75)),
        "G_std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
    }


def _calc_case_metrics(H, valid_mask, dx, dy, wind_dir_deg, comp_cfg):
    street_mask = _build_street_mask(H, valid_mask)

    A = calc_building_density_A(H, dx=dx, dy=dy, valid_mask=valid_mask)
    B = calc_building_aspect_ratio_B(H, dx=dx, dy=dy, valid_mask=valid_mask)
    C = calc_frontal_area_density_C(H, dx=dx, dy=dy, wind_dir_deg=wind_dir_deg, valid_mask=valid_mask)
    D = calc_compactness_D(H, dx=dx, dy=dy, valid_mask=valid_mask)
    E = calc_porosity_E_formula(H, dx=dx, dy=dy, valid_mask=valid_mask)
    F = calc_height_variation_F(H, dx=dx, dy=dy, valid_mask=valid_mask)

    G_map, _ = calc_sky_openness_G_map(
        H=H,
        street_mask=street_mask,
        dx=dx,
        dy=dy,
        n_dirs=int(comp_cfg.get("n_dirs", 72)),
        max_radius=float(comp_cfg.get("max_radius", 100.0)),
        step_length=comp_cfg.get("step_length", None),
        base_height=float(comp_cfg.get("base_height", 0.0)),
    )
    G_stats = _calc_g_stats(G_map, street_mask)

    H_neighbor, _, _ = calc_building_neighbor_proximity_H(
        H=H, dx=dx, dy=dy, valid_mask=valid_mask
    )

    out = {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "E": E,
        "F": F,
        "H_neighbor": H_neighbor,
        "wind_dir_deg": wind_dir_deg,
    }
    out.update(G_stats)
    return out


def _process_one_case(args):
    (
        idx,
        total_cases,
        row_dict,
        ideal_root,
        real_root,
        topo_suffix,
        nodata_value,
        dx,
        dy,
        compute_cfg,
    ) = args

    meta_name = row_dict["meta_name"]
    folder_score = row_dict["folder_score"]
    row_out = dict(row_dict)

    verbose_case_log = bool(compute_cfg.get("verbose_case_log", False))

    try:
        if verbose_case_log:
            print(
                f"[{idx}/{total_cases}] START | meta_name={meta_name} | folder_score={folder_score}",
                flush=True
            )

        case_dir = _resolve_case_folder(folder_score, ideal_root, real_root)
        topo_path = _find_topo_file(case_dir, str(folder_score).strip(), topo_suffix=topo_suffix)
        H = _load_topo_array(topo_path)
        valid_mask = _build_valid_mask(H, nodata_value=nodata_value)
        wind_dir_deg = _extract_wind_dir_from_folder(str(folder_score).strip())


        metric_dict = _calc_case_metrics(
            H=H,
            valid_mask=valid_mask,
            dx=dx,
            dy=dy,
            wind_dir_deg=wind_dir_deg,
            comp_cfg=compute_cfg,
        )

        lp_debug = _calc_lp_debug(H, valid_mask, lp_ori=row_dict.get("lp_ori"))

        row_out.update(metric_dict)
        row_out.update(lp_debug)

        row_out["dbg_topo_path"] = topo_path
        row_out["dbg_dx"] = dx
        row_out["dbg_dy"] = dy
        row_out["status"] = "ok"

        if pd.notna(row_out.get("dbg_abs_A_minus_lp_ori")) and row_out["dbg_abs_A_minus_lp_ori"] > 1e-8:
            print(
                f"[LP_MISMATCH] meta_name={meta_name} | folder_score={folder_score} | "
                f"lp_ori={row_out['dbg_lp_ori']:.10f} | A={row_out['A']:.10f} | "
                f"building={row_out['dbg_building_cells']} | valid={row_out['dbg_valid_cells']} | "
                f"shape={row_out['dbg_shape']} | topo={topo_path}",
                flush=True
            )
        row_out["__order__"] = idx

        if verbose_case_log:
            print(
                f"[{idx}/{total_cases}] OK | meta_name={meta_name} | "
                f"folder_score={folder_score} | wind_dir={wind_dir_deg:.0f}",
                flush=True
            )

    except Exception as e:
        row_out.update({
            "A": np.nan,
            "B": np.nan,
            "C": np.nan,
            "D": np.nan,
            "E": np.nan,
            "F": np.nan,
            "G_25": np.nan,
            "G_50": np.nan,
            "G_75": np.nan,
            "G_std": np.nan,
            "H_neighbor": np.nan,
            "wind_dir_deg": np.nan,
            "status": f"error: {e}",
            "__order__": idx,
        })

        print(
            f"[{idx}/{total_cases}] ERROR | meta_name={meta_name} | "
            f"folder_score={folder_score} | {e}",
            flush=True
        )

    return row_out


def urban_morphology_summary(cfg):
    case_cfg = _get_case_cfg(cfg, target_case="urban_morphology_summary")

    inp = case_cfg["input"]
    out_cfg = case_cfg["output"]
    comp_cfg = case_cfg["compute"]


    nodata_value = inp.get("nodata_value", None)
    topo_suffix = inp.get("topo_suffix", "_topo")

    ideal_root = inp["ideal_root"]
    real_root = inp["real_root"]

    cfg_n_jobs = int(comp_cfg.get("n_jobs", 1))
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    n_jobs = max(1, min(cfg_n_jobs, slurm_cpus))
    progress_every = int(comp_cfg.get("progress_every", 10))

    _ensure_dir(out_cfg["out_dir"])

    df = _prepare_metadata_table(case_cfg)
    total_cases = len(df)

    print(
        f"[Start] 共需处理 {total_cases} 个 case | "
        f"cfg_n_jobs={cfg_n_jobs} | SLURM_CPUS_PER_TASK={slurm_cpus} | "
        f"actual_n_jobs={n_jobs} | numba={NUMBA_AVAILABLE}",
        flush=True
    )

    task_args = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        dx = float(row["dxdy"])
        dy = float(row["dxdy"])
        
        task_args.append((
            idx,
            total_cases,
            row.to_dict(),
            ideal_root,
            real_root,
            topo_suffix,
            nodata_value,
            dx,
            dy,
            comp_cfg,
        ))

    metric_rows = []
    n_ok = 0
    n_err = 0
    t0 = time.time()

    if n_jobs == 1:
        for k, args in enumerate(task_args, start=1):
            row = _process_one_case(args)
            metric_rows.append(row)

            if str(row.get("status", "")).startswith("ok"):
                n_ok += 1
            else:
                n_err += 1

            if (k == 1) or (k % progress_every == 0) or (k == total_cases):
                elapsed = time.time() - t0
                rate = k / elapsed if elapsed > 0 else 0.0
                print(
                    f"[Progress] {k}/{total_cases} done | ok={n_ok} | err={n_err} | "
                    f"elapsed={elapsed:.1f}s | rate={rate:.2f} case/s",
                    flush=True
                )
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_process_one_case, args) for args in task_args]

            for k, fut in enumerate(as_completed(futures), start=1):
                row = fut.result()
                metric_rows.append(row)

                if str(row.get("status", "")).startswith("ok"):
                    n_ok += 1
                else:
                    n_err += 1

                if (k == 1) or (k % progress_every == 0) or (k == total_cases):
                    elapsed = time.time() - t0
                    rate = k / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[Progress] {k}/{total_cases} done | ok={n_ok} | err={n_err} | "
                        f"elapsed={elapsed:.1f}s | rate={rate:.2f} case/s",
                        flush=True
                    )

    out_df = pd.DataFrame(metric_rows).sort_values("__order__").drop(columns="__order__")

    desired_cols = [
        "meta_name",
        "folder_score",
        "NameE",
        "fg",
        "lp_ori",
        "lf_ori",
        "have_ori",
        "hstd_ori",
        "hmax_ori",
        "dxdy",
        "rp_std",
        "rp_ave",
        "d",
        "z0",
        "wind_dir_deg",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G_25",
        "G_50",
        "G_75",
        "G_std",
        "H_neighbor",
        "status",
    ]
    existing_cols = [c for c in desired_cols if c in out_df.columns]
    other_cols = [c for c in out_df.columns if c not in existing_cols]
    out_df = out_df[existing_cols + other_cols]

    float_format = out_cfg.get("float_format", "%.10f")
    out_df.to_csv(out_cfg["out_csv"], index=False, float_format=float_format)

    total_elapsed = time.time() - t0
    print(
        f"[Finish] 结果已写出: {out_cfg['out_csv']} | "
        f"ok={n_ok} | err={n_err} | total_elapsed={total_elapsed:.1f}s",
        flush=True
    )