"""
Single-topography nested-square urban morphology calculator.

Input table columns:
    x(km), y(km), topo(m)

For one 2 km x 2 km topography, calculate the original A-H morphology
metrics in concentric square regions with configurable side lengths.

Definitions retained from the supplied urban_morphology_summary.py:
    A  building plan-area density
    B  building aspect-ratio-like index: sum(h_i * w_i) / sum(K_i)
    C  frontal-area density for a specified wind direction
    D  building volume per unit plan area
    E  volume / (plan area * maximum height)
    F  sample standard deviation of building representative heights
    G  sky-openness quartiles and standard deviation on street cells
    H  mean building-neighbour proximity
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

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
# Configuration and I/O
# =========================================================


def _get_case_cfg(
    cfg: Dict[str, Any],
    target_case: str = "nested_square_morphology",
) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ValueError("YAML 顶层必须是字典。")
    if target_case not in cfg:
        raise KeyError(f"YAML 中未找到 case: {target_case}")

    case_cfg = cfg[target_case]
    if not isinstance(case_cfg, dict):
        raise ValueError(f"case '{target_case}' 的内容必须是字典。")
    return case_cfg


def _ensure_parent_dir(path: str | os.PathLike[str]) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def _read_topo_table(
    path: str | os.PathLike[str],
    sep: Optional[str] = None,
    encoding: Optional[str] = None,
    skip_rows: int = 0,
) -> pd.DataFrame:
    """
    Read a three-column topo table.

    For .dat/.txt/.asc files, automatic mode prioritises arbitrary whitespace
    (``sep=r"\\s+"``). This is required for tables such as::

        x(km)     y(km)   topo(m)
        0.00      0.00       4.0

    ``pandas.read_csv(sep=None)`` must not be used for this format because it
    may treat every individual space as a delimiter and create many empty
    ``Unnamed`` columns.
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到 topo 文件: {path}")

    if int(skip_rows) < 0:
        raise ValueError("skip_rows 不能小于 0。")

    encodings: Iterable[Optional[str]]
    if encoding is not None:
        encodings = [encoding]
    else:
        encodings = ["utf-8", "utf-8-sig", "gb18030"]

    sep_text = "" if sep is None else str(sep).strip()
    auto_sep = sep_text.lower() in {"", "auto", "none", "null"}
    extension = Path(path).suffix.lower()

    if auto_sep:
        # DAT/TXT/ASC topo tables are normally aligned with one or more spaces.
        if extension in {".dat", ".txt", ".asc", ""}:
            separator_candidates = [r"\s+", ",", "\t"]
        elif extension == ".csv":
            separator_candidates = [",", r"\s+", "\t"]
        else:
            separator_candidates = [r"\s+", ",", "\t"]
    else:
        if sep_text == r"\t" or sep_text.lower() == "tab":
            separator_candidates = ["\t"]
        elif sep_text.lower() in {"whitespace", "space", "spaces", r"\s+"}:
            separator_candidates = [r"\s+"]
        else:
            separator_candidates = [sep_text]

    errors = []
    for enc in encodings:
        for separator in separator_candidates:
            try:
                frame = pd.read_csv(
                    path,
                    sep=separator,
                    engine="python",
                    encoding=enc,
                    skiprows=int(skip_rows),
                    skipinitialspace=True,
                )

                # Remove only completely empty columns introduced by malformed
                # delimiters or trailing separators.
                empty_columns = [
                    col for col in frame.columns
                    if str(col).startswith("Unnamed:") and frame[col].isna().all()
                ]
                if empty_columns:
                    frame = frame.drop(columns=empty_columns)

                if frame.shape[1] < 3:
                    raise ValueError(
                        f"仅解析出 {frame.shape[1]} 列，期望至少 3 列；"
                        f"当前分隔符={separator!r}"
                    )

                print(
                    f"[Read topo] path={path} | sep={separator!r} | "
                    f"encoding={enc} | rows={len(frame)} | "
                    f"columns={list(frame.columns)}",
                    flush=True,
                )
                return frame
            except Exception as exc:
                errors.append(
                    f"encoding={enc}, sep={separator!r}: {type(exc).__name__}: {exc}"
                )

    details = "\n".join(errors[-9:])
    raise ValueError(
        f"无法读取 topo 文件: {path}。已尝试的读取方式如下：\n{details}"
    )


def _normalise_column_name(name: Any) -> str:
    return str(name).strip().replace("（", "(").replace("）", ")")


def _resolve_column(df: pd.DataFrame, requested: str) -> str:
    """Resolve a configured column name after stripping spaces/full-width brackets."""
    normalised = {_normalise_column_name(col): col for col in df.columns}
    key = _normalise_column_name(requested)
    if key not in normalised:
        raise KeyError(
            f"输入文件中没有列 '{requested}'。实际列名为: {list(df.columns)}"
        )
    return normalised[key]


def _infer_regular_spacing(values: np.ndarray, axis_name: str) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError(f"{axis_name} 至少需要两个不同坐标。")

    diffs = np.diff(np.sort(values))
    positive = diffs[diffs > 0]
    if positive.size == 0:
        raise ValueError(f"{axis_name} 坐标没有正的网格步长。")

    spacing = float(np.median(positive))
    tolerance = max(1e-10, abs(spacing) * 1e-6)
    if not np.allclose(positive, spacing, rtol=1e-6, atol=tolerance):
        raise ValueError(
            f"{axis_name} 坐标不是规则网格。"
            f"最小步长={positive.min():.12g}, 最大步长={positive.max():.12g}"
        )
    return spacing


def _build_topo_grid(
    table: pd.DataFrame,
    x_column: str,
    y_column: str,
    height_column: str,
    expected_dx_m: Optional[float] = None,
    expected_dy_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Convert the x-y-height table into a 2-D array.

    x is sorted ascending. y is sorted descending so that increasing row index
    corresponds to decreasing physical y, matching the geometry convention used
    by the original program.
    """
    x_col = _resolve_column(table, x_column)
    y_col = _resolve_column(table, y_column)
    h_col = _resolve_column(table, height_column)

    data = table[[x_col, y_col, h_col]].copy()
    data.columns = ["x_km", "y_km", "height_m"]

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="raise")

    if data.isna().any().any():
        raise ValueError("topo 文件包含空值。")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("topo 文件包含 NaN 或无穷值。")
    if data.duplicated(subset=["x_km", "y_km"]).any():
        duplicated = data.loc[
            data.duplicated(subset=["x_km", "y_km"], keep=False),
            ["x_km", "y_km"],
        ].head()
        raise ValueError(f"存在重复的 (x, y) 坐标，例如:\n{duplicated}")

    x_values = np.sort(data["x_km"].unique().astype(float))
    y_values = np.sort(data["y_km"].unique().astype(float))[::-1]

    dx_km = _infer_regular_spacing(x_values, "x")
    dy_km = _infer_regular_spacing(y_values, "y")
    dx_m = dx_km * 1000.0
    dy_m = dy_km * 1000.0

    check_tol_m = max(1e-6, min(dx_m, dy_m) * 1e-5)
    if expected_dx_m is not None and not np.isclose(
        dx_m, float(expected_dx_m), rtol=1e-6, atol=check_tol_m
    ):
        raise ValueError(
            f"x 步长与 YAML 不一致: 文件={dx_m:g} m, YAML={expected_dx_m:g} m"
        )
    if expected_dy_m is not None and not np.isclose(
        dy_m, float(expected_dy_m), rtol=1e-6, atol=check_tol_m
    ):
        raise ValueError(
            f"y 步长与 YAML 不一致: 文件={dy_m:g} m, YAML={expected_dy_m:g} m"
        )

    pivot = data.pivot(index="y_km", columns="x_km", values="height_m")
    pivot = pivot.reindex(index=y_values, columns=x_values)

    if pivot.isna().any().any():
        missing_count = int(pivot.isna().sum().sum())
        raise ValueError(
            f"规则网格中缺少 {missing_count} 个 (x, y) 数据点，无法重建完整矩阵。"
        )

    H = np.ascontiguousarray(pivot.to_numpy(dtype=float))
    return H, x_values, y_values, dx_m, dy_m


# =========================================================
# Core morphology functions retained from the original code
# =========================================================


def _as_float_array(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        raise ValueError("H 必须是二维数组。")
    return H


def _make_valid_mask(H: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    H = _as_float_array(H)
    if valid_mask is None:
        return np.ones_like(H, dtype=bool)

    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != H.shape:
        raise ValueError("valid_mask 与 H 的形状必须一致。")
    return valid_mask


def _cell_area_from_mask(valid_mask: np.ndarray, dx: float, dy: float) -> float:
    if dx <= 0 or dy <= 0:
        raise ValueError("dx 和 dy 必须大于 0。")
    return float(np.count_nonzero(valid_mask) * dx * dy)


def _building_mask(
    H: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    building_threshold_m: float = 0.0,
) -> np.ndarray:
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)
    return (H > building_threshold_m) & valid_mask


def _label_buildings_8(
    H: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    building_threshold_m: float = 0.0,
) -> Tuple[np.ndarray, int]:
    bmask = _building_mask(H, valid_mask, building_threshold_m)
    structure = ndimage.generate_binary_structure(2, 2)
    labels, num = ndimage.label(bmask, structure=structure)
    return labels, int(num)


def _representative_height_hmax(H_obj: np.ndarray, building_threshold_m: float) -> float:
    vals = np.asarray(H_obj, dtype=float)
    vals = vals[vals > building_threshold_m]
    if vals.size == 0:
        return 0.0
    return float(np.max(vals))


def _pixel_corner_coords(mask_obj: np.ndarray, dx: float, dy: float) -> np.ndarray:
    rows, cols = np.where(mask_obj)
    if rows.size == 0:
        return np.empty((0, 2), dtype=float)

    corners = np.empty((rows.size * 4, 2), dtype=float)
    x0 = cols * dx
    x1 = (cols + 1) * dx
    y1 = -rows * dy
    y0 = -(rows + 1) * dy

    corners[0::4, 0] = x0
    corners[0::4, 1] = y0
    corners[1::4, 0] = x0
    corners[1::4, 1] = y1
    corners[2::4, 0] = x1
    corners[2::4, 1] = y0
    corners[3::4, 0] = x1
    corners[3::4, 1] = y1
    return corners


def _projected_width(
    mask_obj: np.ndarray,
    dx: float,
    dy: float,
    wind_dir_deg: float,
) -> float:
    corners = _pixel_corner_coords(mask_obj, dx, dy)
    if corners.shape[0] == 0:
        return 0.0

    theta = np.deg2rad(wind_dir_deg)
    normal = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    projection = corners @ normal
    return float(np.max(projection) - np.min(projection))


def _extract_building_metrics(
    H: np.ndarray,
    dx: float,
    dy: float,
    valid_mask: Optional[np.ndarray] = None,
    wind_dir_deg: float = 0.0,
    building_threshold_m: float = 0.0,
) -> list[Dict[str, float]]:
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)
    labels, num = _label_buildings_8(
        H,
        valid_mask=valid_mask,
        building_threshold_m=building_threshold_m,
    )

    pixel_area = dx * dy
    metrics: list[Dict[str, float]] = []

    for label_id in range(1, num + 1):
        obj_mask = labels == label_id
        H_obj = H[obj_mask]

        K = float(np.count_nonzero(obj_mask) * pixel_area)
        h = _representative_height_hmax(H_obj, building_threshold_m)
        positive = H_obj[H_obj > building_threshold_m]
        V = float(np.sum(positive) * pixel_area)
        w = _projected_width(obj_mask, dx, dy, wind_dir_deg)
        S = float(h * w)

        metrics.append(
            {
                "label": float(label_id),
                "K": K,
                "h": h,
                "V": V,
                "w": w,
                "S": S,
            }
        )

    return metrics


def calc_building_density_A(
    H: np.ndarray,
    dx: float,
    dy: float,
    valid_mask: Optional[np.ndarray] = None,
    building_threshold_m: float = 0.0,
) -> float:
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)
    area = _cell_area_from_mask(valid_mask, dx, dy)
    if area <= 0:
        raise ValueError("计算区域面积必须大于 0。")

    building_area = float(
        np.count_nonzero((H > building_threshold_m) & valid_mask) * dx * dy
    )
    return building_area / area


def calc_building_aspect_ratio_B(
    H: np.ndarray,
    dx: float,
    dy: float,
    valid_mask: Optional[np.ndarray] = None,
    building_threshold_m: float = 0.0,
) -> float:
    metrics = _extract_building_metrics(
        H=H,
        dx=dx,
        dy=dy,
        valid_mask=valid_mask,
        wind_dir_deg=0.0,
        building_threshold_m=building_threshold_m,
    )
    if not metrics:
        return 0.0

    S_sum = float(sum(item["S"] for item in metrics))
    K_sum = float(sum(item["K"] for item in metrics))
    return 0.0 if K_sum <= 0 else S_sum / K_sum


def calc_frontal_area_density_C(
    H: np.ndarray,
    dx: float,
    dy: float,
    wind_dir_deg: float,
    valid_mask: Optional[np.ndarray] = None,
    building_threshold_m: float = 0.0,
) -> float:
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)
    area = _cell_area_from_mask(valid_mask, dx, dy)
    if area <= 0:
        raise ValueError("计算区域面积必须大于 0。")

    metrics = _extract_building_metrics(
        H=H,
        dx=dx,
        dy=dy,
        valid_mask=valid_mask,
        wind_dir_deg=wind_dir_deg,
        building_threshold_m=building_threshold_m,
    )
    if not metrics:
        return 0.0
    return float(sum(item["S"] for item in metrics)) / area


def calc_compactness_D(
    H: np.ndarray,
    dx: float,
    dy: float,
    valid_mask: Optional[np.ndarray] = None,
    building_threshold_m: float = 0.0,
) -> float:
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)
    area = _cell_area_from_mask(valid_mask, dx, dy)
    if area <= 0:
        raise ValueError("计算区域面积必须大于 0。")

    H_use = np.where(valid_mask & (H > building_threshold_m), H, 0.0)
    volume = float(np.sum(H_use) * dx * dy)
    return volume / area


def calc_porosity_E_formula(
    H: np.ndarray,
    dx: float,
    dy: float,
    valid_mask: Optional[np.ndarray] = None,
    building_threshold_m: float = 0.0,
) -> float:
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)
    area = _cell_area_from_mask(valid_mask, dx, dy)
    if area <= 0:
        raise ValueError("计算区域面积必须大于 0。")

    H_use = np.where(valid_mask & (H > building_threshold_m), H, 0.0)
    if np.count_nonzero(H_use > building_threshold_m) == 0:
        return 0.0

    volume = float(np.sum(H_use) * dx * dy)
    h_max = float(np.max(H_use))
    return 0.0 if h_max <= 0 else volume / (area * h_max)


def calc_height_variation_F(
    H: np.ndarray,
    dx: float,
    dy: float,
    valid_mask: Optional[np.ndarray] = None,
    building_threshold_m: float = 0.0,
) -> float:
    metrics = _extract_building_metrics(
        H=H,
        dx=dx,
        dy=dy,
        valid_mask=valid_mask,
        wind_dir_deg=0.0,
        building_threshold_m=building_threshold_m,
    )
    heights = np.asarray([item["h"] for item in metrics], dtype=float)
    if heights.size < 2:
        return np.nan
    return float(np.std(heights, ddof=1))


# =========================================================
# G: sky openness, computed once on the full 2 km topography
# =========================================================


@njit(cache=True)
def _scan_max_elevation_angle_core(
    H: np.ndarray,
    row0: int,
    col0: int,
    theta_rad: float,
    dx: float,
    dy: float,
    max_radius: float,
    step_length: float,
    base_height: float,
    building_threshold_m: float,
) -> float:
    nrow, ncol = H.shape
    x0 = (col0 + 0.5) * dx
    y0 = -(row0 + 0.5) * dy

    nsteps = int(np.floor(max_radius / step_length))
    max_angle = 0.0
    last_row = -10**9
    last_col = -10**9

    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    for step_index in range(1, nsteps + 1):
        distance = step_index * step_length
        x = x0 + distance * cos_theta
        y = y0 + distance * sin_theta

        col = int(np.floor(x / dx))
        row = int(np.floor(-y / dy))

        if row < 0 or row >= nrow or col < 0 or col >= ncol:
            break
        if row == last_row and col == last_col:
            continue

        last_row = row
        last_col = col
        height = H[row, col]
        if height <= max(base_height, building_threshold_m):
            continue

        angle = np.arctan2(height - base_height, distance)
        if angle > max_angle:
            max_angle = angle

    return max_angle


@njit(cache=True)
def _calc_sky_openness_for_point_core(
    H: np.ndarray,
    row0: int,
    col0: int,
    dx: float,
    dy: float,
    n_dirs: int,
    max_radius: float,
    step_length: float,
    base_height: float,
    building_threshold_m: float,
) -> float:
    sin_sum = 0.0
    for direction_index in range(n_dirs):
        theta = 2.0 * np.pi * direction_index / n_dirs
        gamma = _scan_max_elevation_angle_core(
            H,
            row0,
            col0,
            theta,
            dx,
            dy,
            max_radius,
            step_length,
            base_height,
            building_threshold_m,
        )
        sin_sum += np.sin(gamma)

    return 1.0 - sin_sum / n_dirs


def calc_sky_openness_G_map(
    H: np.ndarray,
    street_mask: np.ndarray,
    dx: float,
    dy: float,
    n_dirs: int = 72,
    max_radius: float = 100.0,
    step_length: Optional[float] = None,
    base_height: float = 0.0,
    building_threshold_m: float = 0.0,
    progress_every: int = 2000,
) -> np.ndarray:
    H = np.ascontiguousarray(_as_float_array(H), dtype=np.float64)
    street_mask = np.asarray(street_mask, dtype=bool)
    if street_mask.shape != H.shape:
        raise ValueError("street_mask 与 H 的形状必须一致。")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx 和 dy 必须大于 0。")
    if n_dirs <= 0:
        raise ValueError("n_dirs 必须大于 0。")
    if max_radius <= 0:
        raise ValueError("max_radius 必须大于 0。")

    if step_length is None:
        step_length = min(dx, dy) / 2.0
    step_length = float(step_length)
    if step_length <= 0:
        raise ValueError("step_length 必须大于 0。")

    G_map = np.full(H.shape, np.nan, dtype=float)
    rows, cols = np.where(street_mask)
    total = rows.size

    for index, (row, col) in enumerate(zip(rows, cols), start=1):
        G_map[row, col] = float(
            _calc_sky_openness_for_point_core(
                H,
                int(row),
                int(col),
                float(dx),
                float(dy),
                int(n_dirs),
                float(max_radius),
                float(step_length),
                float(base_height),
                float(building_threshold_m),
            )
        )

        if progress_every > 0 and (
            index == 1 or index % progress_every == 0 or index == total
        ):
            print(f"[G progress] {index}/{total} street cells", flush=True)

    return G_map


def _calc_g_stats(G_map: np.ndarray, target_street_mask: np.ndarray) -> Dict[str, float]:
    values = G_map[target_street_mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "G_25": np.nan,
            "G_50": np.nan,
            "G_75": np.nan,
            "G_std": np.nan,
            "G_n_street": 0,
        }

    return {
        "G_25": float(np.percentile(values, 25)),
        "G_50": float(np.percentile(values, 50)),
        "G_75": float(np.percentile(values, 75)),
        "G_std": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
        "G_n_street": int(values.size),
    }


# =========================================================
# H: building-neighbour proximity
# =========================================================


def _extract_building_boundaries(
    H: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    building_threshold_m: float = 0.0,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    H = _as_float_array(H)
    valid_mask = _make_valid_mask(H, valid_mask)
    labels, num = _label_buildings_8(
        H,
        valid_mask=valid_mask,
        building_threshold_m=building_threshold_m,
    )

    structure = ndimage.generate_binary_structure(2, 2)
    boundary_masks: Dict[int, np.ndarray] = {}
    for label_id in range(1, num + 1):
        obj_mask = labels == label_id
        eroded = ndimage.binary_erosion(obj_mask, structure=structure, border_value=0)
        boundary_masks[label_id] = obj_mask & (~eroded)

    return labels, boundary_masks


def _mask_to_center_coords(mask_obj: np.ndarray, dx: float, dy: float) -> np.ndarray:
    rows, cols = np.where(mask_obj)
    if rows.size == 0:
        return np.empty((0, 2), dtype=float)

    x = (cols + 0.5) * dx
    y = -(rows + 0.5) * dy
    return np.column_stack([x, y]).astype(float)


def _pairwise_min_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return np.nan

    # Same minimum Euclidean boundary-cell-centre distance as the original
    # broadcasting implementation, but without constructing a huge 3-D array.
    if coords_a.shape[0] <= coords_b.shape[0]:
        tree = cKDTree(coords_b)
        distances, _ = tree.query(coords_a, k=1)
    else:
        tree = cKDTree(coords_a)
        distances, _ = tree.query(coords_b, k=1)
    return float(np.min(distances))


def calc_building_neighbor_proximity_H(
    H: np.ndarray,
    dx: float,
    dy: float,
    valid_mask: Optional[np.ndarray] = None,
    building_threshold_m: float = 0.0,
) -> Tuple[float, int]:
    _, boundary_masks = _extract_building_boundaries(
        H,
        valid_mask=valid_mask,
        building_threshold_m=building_threshold_m,
    )
    building_ids = np.asarray(sorted(boundary_masks), dtype=int)
    n_buildings = int(building_ids.size)

    if n_buildings < 2:
        return np.nan, n_buildings

    boundary_coords = {
        int(building_id): _mask_to_center_coords(
            boundary_masks[int(building_id)], dx, dy
        )
        for building_id in building_ids
    }

    distance_sum_per_building = np.zeros(n_buildings, dtype=float)
    for i in range(n_buildings):
        for j in range(i + 1, n_buildings):
            distance = _pairwise_min_distance(
                boundary_coords[int(building_ids[i])],
                boundary_coords[int(building_ids[j])],
            )
            distance_sum_per_building[i] += distance
            distance_sum_per_building[j] += distance

    mean_distance_per_building = distance_sum_per_building / (n_buildings - 1)
    return float(np.mean(mean_distance_per_building)), n_buildings


# =========================================================
# Nested-square selection and execution
# =========================================================


def _axis_mask(
    values_km: np.ndarray,
    center_km: float,
    side_length_m: float,
    include_boundary: bool,
) -> np.ndarray:
    half_width_km = side_length_m / 2000.0
    lower = center_km - half_width_km
    upper = center_km + half_width_km
    tolerance = max(1e-12, np.ptp(values_km) * 1e-12)

    if include_boundary:
        return (values_km >= lower - tolerance) & (values_km <= upper + tolerance)
    return (values_km >= lower - tolerance) & (values_km < upper - tolerance)


def _build_square_selection(
    x_values_km: np.ndarray,
    y_values_km: np.ndarray,
    center_x_km: float,
    center_y_km: float,
    side_length_m: float,
    include_boundary: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mask = _axis_mask(
        x_values_km,
        center_x_km,
        side_length_m,
        include_boundary,
    )
    y_mask = _axis_mask(
        y_values_km,
        center_y_km,
        side_length_m,
        include_boundary,
    )

    if not np.any(x_mask) or not np.any(y_mask):
        raise ValueError(f"边长 {side_length_m:g} m 的正方形没有选中任何网格点。")

    full_mask = np.outer(y_mask, x_mask)
    return x_mask, y_mask, full_mask


def _calculate_subdomain_metrics(
    H_sub: np.ndarray,
    dx_m: float,
    dy_m: float,
    wind_dir_deg: float,
    building_threshold_m: float,
) -> Dict[str, float]:
    valid_mask = np.ones_like(H_sub, dtype=bool)

    A = calc_building_density_A(
        H_sub,
        dx_m,
        dy_m,
        valid_mask,
        building_threshold_m,
    )
    B = calc_building_aspect_ratio_B(
        H_sub,
        dx_m,
        dy_m,
        valid_mask,
        building_threshold_m,
    )
    C = calc_frontal_area_density_C(
        H_sub,
        dx_m,
        dy_m,
        wind_dir_deg,
        valid_mask,
        building_threshold_m,
    )
    D = calc_compactness_D(
        H_sub,
        dx_m,
        dy_m,
        valid_mask,
        building_threshold_m,
    )
    E = calc_porosity_E_formula(
        H_sub,
        dx_m,
        dy_m,
        valid_mask,
        building_threshold_m,
    )
    F = calc_height_variation_F(
        H_sub,
        dx_m,
        dy_m,
        valid_mask,
        building_threshold_m,
    )
    H_neighbor, n_buildings = calc_building_neighbor_proximity_H(
        H_sub,
        dx_m,
        dy_m,
        valid_mask,
        building_threshold_m,
    )

    building_cells = int(np.count_nonzero(H_sub > building_threshold_m))
    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "E": E,
        "F": F,
        "H_neighbor": H_neighbor,
        "n_buildings": n_buildings,
        "n_building_cells": building_cells,
    }

def _read_original_morphology_by_side(
    input_cfg: Dict[str, Any],
    required_side_lengths_m: Iterable[float],
) -> Dict[float, Dict[str, float]]:
    """
    从原始形态参数 CSV 中，按照 side_length_km 匹配不同尺度的
    lp、lf、have、hstd、hmax。

    返回格式
    --------
    {
        100.0: {
            "lp_ori": ...,
            "lf_ori": ...,
            "have_ori": ...,
            "hstd_ori": ...,
            "hmax_ori": ...,
        },
        ...
    }
    """
    original_csv = input_cfg.get("original_morphology_csv")

    if original_csv is None or str(original_csv).strip() == "":
        raise KeyError(
            "YAML 的 input 中缺少 original_morphology_csv。"
        )

    original_path = Path(original_csv).expanduser()

    if not original_path.is_absolute():
        original_path = Path.cwd() / original_path

    original_path = original_path.resolve()

    if not original_path.is_file():
        raise FileNotFoundError(
            f"未找到原始形态参数 CSV：{original_path}"
        )

    encoding_cfg = input_cfg.get(
        "original_morphology_encoding",
        None,
    )

    if encoding_cfg is None:
        encoding_candidates = [
            "utf-8",
            "utf-8-sig",
            "gb18030",
        ]
    else:
        encoding_candidates = [str(encoding_cfg)]

    sep_cfg = input_cfg.get(
        "original_morphology_sep",
        ",",
    )

    sep_text = "" if sep_cfg is None else str(sep_cfg).strip()
    sep_lower = sep_text.lower()

    if sep_lower in {"", "auto", "none", "null"}:
        separator_candidates = [
            ",",
            r"\s+",
            "\t",
        ]
    elif sep_lower in {
        "whitespace",
        "space",
        "spaces",
        r"\s+",
    }:
        separator_candidates = [r"\s+"]
    elif sep_lower in {"tab", r"\t"}:
        separator_candidates = ["\t"]
    else:
        separator_candidates = [sep_text]

    skip_rows = int(
        input_cfg.get(
            "original_morphology_skip_rows",
            0,
        )
    )

    read_errors = []
    original_table = None
    used_separator = None
    used_encoding = None

    for encoding in encoding_candidates:
        for separator in separator_candidates:
            try:
                frame = pd.read_csv(
                    original_path,
                    sep=separator,
                    engine="python",
                    encoding=encoding,
                    skiprows=skip_rows,
                    skipinitialspace=True,
                )

                empty_columns = [
                    column
                    for column in frame.columns
                    if str(column).startswith("Unnamed:")
                    and frame[column].isna().all()
                ]

                if empty_columns:
                    frame = frame.drop(columns=empty_columns)

                if frame.empty:
                    raise ValueError("读取到的表格为空。")

                original_table = frame
                used_separator = separator
                used_encoding = encoding
                break

            except Exception as exc:
                read_errors.append(
                    f"encoding={encoding}, "
                    f"sep={separator!r}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if original_table is not None:
            break

    if original_table is None:
        error_detail = "\n".join(read_errors[-9:])
        raise ValueError(
            f"无法读取原始形态参数 CSV：{original_path}\n"
            f"{error_detail}"
        )

    print(
        f"[Read original morphology] "
        f"path={original_path} | "
        f"sep={used_separator!r} | "
        f"encoding={used_encoding} | "
        f"rows={len(original_table)} | "
        f"columns={list(original_table.columns)}",
        flush=True,
    )

    side_column = _resolve_column(
        original_table,
        input_cfg.get(
            "original_side_length_column",
            "side_length_km",
        ),
    )

    lp_column = _resolve_column(
        original_table,
        input_cfg.get(
            "original_lp_column",
            "lp",
        ),
    )

    lf_column = _resolve_column(
        original_table,
        input_cfg.get(
            "original_lf_column",
            "lf",
        ),
    )

    have_column = _resolve_column(
        original_table,
        input_cfg.get(
            "original_have_column",
            "have",
        ),
    )

    hstd_column = _resolve_column(
        original_table,
        input_cfg.get(
            "original_hstd_column",
            "hstd",
        ),
    )

    hmax_column = _resolve_column(
        original_table,
        input_cfg.get(
            "original_hmax_column",
            "hmax",
        ),
    )

    morphology = original_table[
        [
            side_column,
            lp_column,
            lf_column,
            have_column,
            hstd_column,
            hmax_column,
        ]
    ].copy()

    morphology.columns = [
        "source_side_length",
        "lp_ori",
        "lf_ori",
        "have_ori",
        "hstd_ori",
        "hmax_ori",
    ]

    for column in morphology.columns:
        morphology[column] = pd.to_numeric(
            morphology[column],
            errors="raise",
        )

    if morphology.isna().any().any():
        null_columns = morphology.columns[
            morphology.isna().any()
        ].tolist()

        raise ValueError(
            "原始形态参数 CSV 中存在空值，涉及列："
            f"{null_columns}"
        )

    if not np.isfinite(
        morphology.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "原始形态参数 CSV 中存在 NaN 或无穷值。"
        )

    side_unit = str(
        input_cfg.get(
            "original_side_length_unit",
            "km",
        )
    ).strip().lower()

    if side_unit == "km":
        morphology["source_side_length_m"] = (
            morphology["source_side_length"] * 1000.0
        )
    elif side_unit == "m":
        morphology["source_side_length_m"] = (
            morphology["source_side_length"]
        )
    else:
        raise ValueError(
            "original_side_length_unit 仅支持 'km' 或 'm'，"
            f"当前值为：{side_unit!r}"
        )

    source_side_values_m = morphology[
        "source_side_length_m"
    ].to_numpy(dtype=float)

    original_by_side: Dict[
        float,
        Dict[str, float],
    ] = {}

    for required_side_m in required_side_lengths_m:
        required_side_m = float(required_side_m)

        tolerance_m = max(
            1e-6,
            abs(required_side_m) * 1e-9,
        )

        matched_indices = np.where(
            np.isclose(
                source_side_values_m,
                required_side_m,
                rtol=0.0,
                atol=tolerance_m,
            )
        )[0]

        if matched_indices.size == 0:
            available_km = (
                source_side_values_m / 1000.0
            ).tolist()

            raise KeyError(
                f"原始形态参数 CSV 中未找到边长 "
                f"{required_side_m / 1000.0:g} km。"
                f"可用边长为：{available_km}"
            )

        if matched_indices.size > 1:
            raise ValueError(
                f"原始形态参数 CSV 中边长 "
                f"{required_side_m / 1000.0:g} km "
                f"出现了 {matched_indices.size} 次，"
                "每种边长只能保留一行。"
            )

        source_row = morphology.iloc[
            int(matched_indices[0])
        ]

        original_by_side[required_side_m] = {
            "lp_ori": float(source_row["lp_ori"]),
            "lf_ori": float(source_row["lf_ori"]),
            "have_ori": float(source_row["have_ori"]),
            "hstd_ori": float(source_row["hstd_ori"]),
            "hmax_ori": float(source_row["hmax_ori"]),
        }

        print(
            f"[Original matched] "
            f"side={required_side_m / 1000.0:g} km | "
            f"lp_ori={source_row['lp_ori']:.10g} | "
            f"lf_ori={source_row['lf_ori']:.10g} | "
            f"have_ori={source_row['have_ori']:.10g} | "
            f"hstd_ori={source_row['hstd_ori']:.10g} | "
            f"hmax_ori={source_row['hmax_ori']:.10g}",
            flush=True,
        )

    return original_by_side

def nested_square_morphology(
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    主计算函数。

    计算一个 topo 文件中不同边长的同心正方形城市形态参数，
    并根据 side_length_km 从外部 CSV 中匹配原始形态参数。
    """
    case_cfg = _get_case_cfg(cfg)

    input_cfg = case_cfg["input"]
    output_cfg = case_cfg["output"]
    region_cfg = case_cfg["regions"]
    compute_cfg = case_cfg["compute"]

    # =====================================================
    # 1. 读取 topo 文件
    # =====================================================

    topo_csv = input_cfg["topo_csv"]

    table = _read_topo_table(
        topo_csv,
        sep=input_cfg.get("sep", None),
        encoding=input_cfg.get("encoding", None),
        skip_rows=int(
            input_cfg.get("skip_rows", 0)
        ),
    )

    (
        H_full,
        x_values_km,
        y_values_km,
        dx_m,
        dy_m,
    ) = _build_topo_grid(
        table=table,
        x_column=input_cfg.get(
            "x_column",
            "x(km)",
        ),
        y_column=input_cfg.get(
            "y_column",
            "y(km)",
        ),
        height_column=input_cfg.get(
            "height_column",
            "topo(m)",
        ),
        expected_dx_m=input_cfg.get(
            "expected_dx_m",
            None,
        ),
        expected_dy_m=input_cfg.get(
            "expected_dy_m",
            None,
        ),
    )

    # =====================================================
    # 2. 确定区域中心
    # =====================================================

    center_x_km = region_cfg.get(
        "center_x_km",
        None,
    )

    center_y_km = region_cfg.get(
        "center_y_km",
        None,
    )

    if center_x_km is None:
        center_x_km = float(
            (
                x_values_km.min()
                + x_values_km.max()
            )
            / 2.0
        )
    else:
        center_x_km = float(center_x_km)

    if center_y_km is None:
        center_y_km = float(
            (
                y_values_km.min()
                + y_values_km.max()
            )
            / 2.0
        )
    else:
        center_y_km = float(center_y_km)

    # =====================================================
    # 3. 读取正方形边长
    # =====================================================

    side_lengths_m = [
        float(value)
        for value in region_cfg[
            "side_lengths_m"
        ]
    ]

    if (
        not side_lengths_m
        or any(
            value <= 0
            for value in side_lengths_m
        )
    ):
        raise ValueError(
            "regions.side_lengths_m 必须包含正数。"
        )

    # 去重并从小到大排列
    side_lengths_m = sorted(
        dict.fromkeys(side_lengths_m)
    )

    include_boundary = bool(
        region_cfg.get(
            "include_boundary",
            True,
        )
    )

    # =====================================================
    # 4. 计算参数
    # =====================================================

    building_threshold_m = float(
        compute_cfg.get(
            "building_threshold_m",
            0.0,
        )
    )

    wind_dir_deg = float(
        compute_cfg.get(
            "wind_dir_deg",
            0.0,
        )
    )

    print(
        "[Input] "
        f"shape={H_full.shape}, "
        f"dx={dx_m:g} m, "
        f"dy={dy_m:g} m, "
        f"x=[{x_values_km.min():g}, "
        f"{x_values_km.max():g}] km, "
        f"y=[{y_values_km.min():g}, "
        f"{y_values_km.max():g}] km",
        flush=True,
    )

    print(
        f"[Center] "
        f"x={center_x_km:g} km, "
        f"y={center_y_km:g} km | "
        f"wind={wind_dir_deg:g} deg | "
        f"numba={NUMBA_AVAILABLE}",
        flush=True,
    )

    # =====================================================
    # 5. 读取不同尺度的原始形态参数
    # =====================================================

    original_by_side = (
        _read_original_morphology_by_side(
            input_cfg=input_cfg,
            required_side_lengths_m=side_lengths_m,
        )
    )

    # =====================================================
    # 6. 在完整 topo 上计算 G
    # =====================================================

    full_street_mask = (
        H_full <= building_threshold_m
    )

    g_t0 = time.time()

    G_map = calc_sky_openness_G_map(
        H=H_full,
        street_mask=full_street_mask,
        dx=dx_m,
        dy=dy_m,
        n_dirs=int(
            compute_cfg.get(
                "n_dirs",
                72,
            )
        ),
        max_radius=float(
            compute_cfg.get(
                "max_radius_m",
                100.0,
            )
        ),
        step_length=compute_cfg.get(
            "step_length_m",
            None,
        ),
        base_height=float(
            compute_cfg.get(
                "base_height_m",
                0.0,
            )
        ),
        building_threshold_m=(
            building_threshold_m
        ),
        progress_every=int(
            compute_cfg.get(
                "g_progress_every",
                2000,
            )
        ),
    )

    print(
        f"[G finish] "
        f"elapsed={time.time() - g_t0:.1f} s",
        flush=True,
    )

    # =====================================================
    # 7. 可选保存 G_map
    # =====================================================

    save_g_map = bool(
        output_cfg.get(
            "save_g_map",
            False,
        )
    )

    if save_g_map:
        g_map_npy = output_cfg.get(
            "g_map_npy",
            None,
        )

        if not g_map_npy:
            output_csv_path = Path(
                output_cfg["out_csv"]
            )

            g_map_npy = str(
                output_csv_path.with_name(
                    output_csv_path.stem
                    + "_G_map.npy"
                )
            )

        _ensure_parent_dir(g_map_npy)
        np.save(g_map_npy, G_map)

        print(
            f"[Output] G map: {g_map_npy}",
            flush=True,
        )

    # =====================================================
    # 8. 分别计算五种边长
    # =====================================================

    result_rows: list[
        Dict[str, Any]
    ] = []

    for side_length_m in side_lengths_m:
        side_length_m = float(
            side_length_m
        )

        side_length_km = (
            side_length_m / 1000.0
        )

        (
            x_mask,
            y_mask,
            square_full_mask,
        ) = _build_square_selection(
            x_values_km=x_values_km,
            y_values_km=y_values_km,
            center_x_km=center_x_km,
            center_y_km=center_y_km,
            side_length_m=side_length_m,
            include_boundary=include_boundary,
        )

        H_sub = H_full[
            np.ix_(
                y_mask,
                x_mask,
            )
        ]

        selected_x = x_values_km[
            x_mask
        ]

        selected_y = y_values_km[
            y_mask
        ]

        # 中心点和边界仅打印，不写入 CSV
        print(
            f"[Square geometry] "
            f"side={side_length_km:g} km | "
            f"center=({center_x_km:g}, "
            f"{center_y_km:g}) km | "
            f"x=[{selected_x.min():g}, "
            f"{selected_x.max():g}] km | "
            f"y=[{selected_y.min():g}, "
            f"{selected_y.max():g}] km",
            flush=True,
        )

        metrics = _calculate_subdomain_metrics(
            H_sub=H_sub,
            dx_m=dx_m,
            dy_m=dy_m,
            wind_dir_deg=wind_dir_deg,
            building_threshold_m=(
                building_threshold_m
            ),
        )

        g_stats = _calc_g_stats(
            G_map,
            target_street_mask=(
                square_full_mask
                & full_street_mask
            ),
        )

        n_rows, n_cols = H_sub.shape

        represented_area_m2 = float(
            n_rows
            * n_cols
            * dx_m
            * dy_m
        )

        original_values = (
            original_by_side[
                side_length_m
            ]
        )

        row: Dict[str, Any] = {
            # 使用 km，不再输出 side_length_m
            "side_length_km": (
                side_length_km
            ),

            # 从原始 CSV 中按边长匹配
            "lp_ori": original_values[
                "lp_ori"
            ],
            "lf_ori": original_values[
                "lf_ori"
            ],
            "have_ori": original_values[
                "have_ori"
            ],
            "hstd_ori": original_values[
                "hstd_ori"
            ],
            "hmax_ori": original_values[
                "hmax_ori"
            ],

            # 网格信息
            "n_x": int(n_cols),
            "n_y": int(n_rows),
            "n_grid_cells": int(
                n_rows * n_cols
            ),
            "represented_area_m2": (
                represented_area_m2
            ),
            "dx_m": dx_m,
            "dy_m": dy_m,
            "wind_dir_deg": (
                wind_dir_deg
            ),
        }

        row.update(metrics)
        row.update(g_stats)

        result_rows.append(row)

        print(
            f"[Square result] "
            f"side={side_length_km:g} km | "
            f"shape={H_sub.shape} | "
            f"buildings="
            f"{metrics['n_buildings']} | "
            f"lp_ori="
            f"{original_values['lp_ori']:.6f} | "
            f"lf_ori="
            f"{original_values['lf_ori']:.6f} | "
            f"A={metrics['A']:.6f} | "
            f"C={metrics['C']:.6f}",
            flush=True,
        )

    # =====================================================
    # 9. 整理最终输出顺序
    # =====================================================

    result = pd.DataFrame(
        result_rows
    )

    column_order = [
        "side_length_km",

        # 原始形态参数
        "lp_ori",
        "lf_ori",
        "have_ori",
        "hstd_ori",
        "hmax_ori",

        # 网格信息
        "n_x",
        "n_y",
        "n_grid_cells",
        "represented_area_m2",
        "dx_m",
        "dy_m",
        "wind_dir_deg",

        # 建筑信息
        "n_buildings",
        "n_building_cells",

        # A-H 指标
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
        "G_n_street",
        "H_neighbor",
    ]

    missing_output_columns = [
        column
        for column in column_order
        if column not in result.columns
    ]

    if missing_output_columns:
        raise KeyError(
            "最终结果中缺少以下列："
            f"{missing_output_columns}"
        )

    result = result[
        column_order
    ]

    # =====================================================
    # 10. 写出 CSV
    # =====================================================

    out_csv = Path(
        output_cfg["out_csv"]
    ).expanduser()

    if not out_csv.is_absolute():
        out_csv = (
            Path.cwd()
            / out_csv
        )

    out_csv = out_csv.resolve()

    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        out_csv,
        index=False,
        float_format=output_cfg.get(
            "float_format",
            "%.10f",
        ),
        encoding=output_cfg.get(
            "encoding",
            "utf-8-sig",
        ),
    )

    if not out_csv.is_file():
        raise IOError(
            f"CSV 写出失败，未找到输出文件："
            f"{out_csv}"
        )

    print(
        f"[Finish] 结果已写出：{out_csv}",
        flush=True,
    )

    print(
        f"[Finish] 输出列："
        f"{list(result.columns)}",
        flush=True,
    )

    print(
        f"[Finish] 文件大小："
        f"{out_csv.stat().st_size} bytes",
        flush=True,
    )

    return result