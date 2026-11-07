# -*- coding: utf-8 -*-
"""
pipeline_kanda.py

UrbanTALES 剖面数据上复现 Kanda(2013) Method 1（统一高度带版）：
- Z_kanda = [Hmax + 0.2*Hmean, Hmax + Hmean]
- u*(z) = sqrt(|Mom(z)|)，在 Z_kanda 内聚合得到 u*_eff（默认 median）
- 固定 u*_eff 与 kappa=0.4，在 Z_kanda 内对数律最小二乘拟合 d, z0
- Step4：计算 dMom/dz 梯度指标（G 与无量纲 G_star）
- Step5：输出稳定性指标（CV_u 与 CV_Mom）以及 QC 分级

【重要】本版本已按你新增的 YAML 小模块 `urban_recal:` 读取参数与列名映射。
也就是说：本代码只从 cfg["urban_recal"] 读取本流程所需配置，不再依赖你全局扁平 columns。

对外唯一入口：
    Kanda_method_to_Urban(cfg: dict) -> pandas.DataFrame

你在 main.py 里只需要：
    from pipeline_kanda import Kanda_method_to_Urban
    Kanda_method_to_Urban(cfg)

"""

from __future__ import annotations

import os
import glob
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

# 用于非线性最小二乘拟合（d, z0）
from scipy.optimize import least_squares


# =============================================================================
# 0. 通用工具函数：统计、差分
# =============================================================================

def _safe_float(x) -> float:
    """尽量把输入转成 float；失败则返回 nan。"""
    try:
        return float(x)
    except Exception:
        return float("nan")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE"""
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2（以观测方差为基准）"""
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _cv(x: np.ndarray) -> float:
    """
    变异系数 CV = std / mean
    注意：x 应为正量或已取绝对值（例如 |Mom| 或 u*）。
    """
    if x.size == 0:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if mu == 0:
        return float("nan")
    return float(sd / mu)


def _central_diff(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    对离散点计算中心差分近似 dy/dz：
    - 两端点用前/后向差分
    - 中间点用中心差分
    """
    n = y.size
    if n < 2:
        return np.array([], dtype=float)

    dydz = np.empty(n, dtype=float)

    dz0 = (z[1] - z[0])
    dydz[0] = (y[1] - y[0]) / dz0 if dz0 != 0 else np.nan

    for i in range(1, n - 1):
        dz = (z[i + 1] - z[i - 1])
        dydz[i] = (y[i + 1] - y[i - 1]) / dz if dz != 0 else np.nan

    dzn = (z[-1] - z[-2])
    dydz[-1] = (y[-1] - y[-2]) / dzn if dzn != 0 else np.nan

    return dydz


# =============================================================================
# 1. Kanda Method 1：核心数学与拟合
# =============================================================================

@dataclass
class FitResult:
    """保存最小二乘拟合的输出"""
    d: float
    z0: float
    r2: float
    rmse: float
    success: bool
    message: str


def _z_kanda(hmax: float, hmean: float) -> Tuple[float, float]:
    """
    Kanda(2013) 统一对数拟合高度带：
        Z_kanda = [Hmax + 0.2*Hmean,  Hmax + Hmean]
    """
    z_low = hmax + 0.2 * hmean
    z_high = hmax + 1.0 * hmean
    return float(z_low), float(z_high)


def _select_band(df: pd.DataFrame, z_col: str, z_low: float, z_high: float) -> pd.DataFrame:
    """从 profile 数据中截取 z ∈ [z_low, z_high] 的层点，并按 z 升序排序。"""
    band = df[(df[z_col] >= z_low) & (df[z_col] <= z_high)].copy()
    band = band.sort_values(z_col).reset_index(drop=True)
    return band


def _ustar_from_mom(mom: np.ndarray) -> np.ndarray:
    """
    用总动量通量 Mom（运动学形式，m^2 s^-2）构造摩擦速度剖面：
        u*(z) = sqrt(|Mom(z)|)
    """
    return np.sqrt(np.abs(mom))


def _predict_u(z: np.ndarray, d: float, z0: float, ustar_eff: float, kappa: float) -> np.ndarray:
    """
    对数律预测（沿主风向分量）：
        u(z) = (u*eff / kappa) * ln( (z - d) / z0 )
    要求：z - d > 0 且 z0 > 0。
    """
    return (ustar_eff / kappa) * np.log((z - d) / z0)


def _fit_d_z0(
    z: np.ndarray,
    u: np.ndarray,
    ustar_eff: float,
    kappa: float,
    z0_bounds: Tuple[float, float],
    d_upper_frac: float,
    d0_frac_zlow: float,
    z00_frac: float,
) -> FitResult:
    """
    固定 u*eff 后，非线性最小二乘拟合 d 与 z0。

    约束：
    - z0 ∈ [z0_min, z0_max] 且 z0 > 0
    - d ∈ [0, d_upper_frac * min(z)]，以确保 z-d 始终为正
    """
    if z.size < 3:
        return FitResult(np.nan, np.nan, np.nan, np.nan, False, "Too few points for fit")

    z_low = float(np.min(z))

    # d 的边界：0 ~ 0.9*min(z)（可由配置调节）
    d_min = 0.0
    d_max = float(d_upper_frac) * z_low
    if d_max <= d_min:
        return FitResult(np.nan, np.nan, np.nan, np.nan, False, "Invalid d bounds")

    z0_min, z0_max = map(float, z0_bounds)
    if z0_min <= 0 or z0_max <= z0_min:
        return FitResult(np.nan, np.nan, np.nan, np.nan, False, "Invalid z0 bounds")

    # 初值策略：提高批处理稳定性
    d0 = float(d0_frac_zlow) * z_low
    d0 = min(max(d0, d_min + 1e-9), d_max - 1e-9)

    z00 = float(z00_frac) * max(z_low - d0, 1e-6)
    z00 = min(max(z00, z0_min), z0_max)

    # 残差函数：最小化 u_obs - u_pred
    def residuals(x: np.ndarray) -> np.ndarray:
        d, z0 = float(x[0]), float(x[1])

        # 防止 log 输入非正
        if z0 <= 0:
            return np.ones_like(u) * 1e6
        if np.any(z - d <= 0):
            return np.ones_like(u) * 1e6

        uhat = _predict_u(z, d, z0, ustar_eff, kappa)
        return (u - uhat)

    x0 = np.array([d0, z00], dtype=float)
    lb = np.array([d_min, z0_min], dtype=float)
    ub = np.array([d_max, z0_max], dtype=float)

    try:
        sol = least_squares(
            residuals,
            x0,
            bounds=(lb, ub),
            method="trf",
            loss="linear",   # 你若将来需要稳健拟合可改为 soft_l1 等
        )
        d_hat, z0_hat = float(sol.x[0]), float(sol.x[1])
        u_pred = _predict_u(z, d_hat, z0_hat, ustar_eff, kappa)
        fit_rmse = _rmse(u, u_pred)
        fit_r2 = _r2(u, u_pred)
        msg = sol.message if isinstance(sol.message, str) else "OK"
        return FitResult(d_hat, z0_hat, fit_r2, fit_rmse, bool(sol.success), msg)
    except Exception as e:
        return FitResult(np.nan, np.nan, np.nan, np.nan, False, f"Fit failed: {e}")


def _qc_classify(
    n_fit: int,
    cv_u: float,
    r2_val: float,
    d: float,
    z0: float,
    z_low: float,
    z0_bounds: Tuple[float, float],
    cv_green: float,
    cv_red: float,
    r2_green: float,
    r2_red: float,
    nmin: int,
    fit_success: bool,
) -> Tuple[str, str]:
    """
    QC 分级逻辑（建议默认）：
    - FAIL: 拟合失败 或 参数为 NaN
    - RED : 点数不足、CV过大、R2过低、或参数越界/不合理
    - GREEN: CV<=cv_green 且 R2>=r2_green 且其它检查通过
    - AMBER: 介于 GREEN 与 RED 之间

    返回：
    - qc_level: GREEN/AMBER/RED/FAIL
    - qc_flags: 用 ; 拼接的标记字符串，便于后续筛选与排查
    """
    flags: List[str] = []

    # 点数检查
    if n_fit < nmin:
        flags.append("INSUFF_POINTS")

    # 拟合成功与否
    if not fit_success:
        flags.append("FIT_FAIL")

    # 参数物理合理性检查
    if not np.isfinite(d) or not np.isfinite(z0):
        flags.append("PARAM_NAN")
    else:
        if d <= 0:
            flags.append("D_NONPOS")
        if d >= z_low:
            flags.append("D_TOO_LARGE")

        z0_min, z0_max = z0_bounds
        if z0 <= 0:
            flags.append("Z0_NONPOS")
        if (z0 < z0_min) or (z0 > z0_max):
            flags.append("Z0_OUT_OF_BOUNDS")

    # 稳定性指标
    if np.isfinite(cv_u):
        if cv_u > cv_red:
            flags.append("CVU_HIGH")
        elif cv_u > cv_green:
            flags.append("CVU_MED")
    else:
        flags.append("CVU_NAN")

    # 拟合优度
    if np.isfinite(r2_val):
        if r2_val < r2_red:
            flags.append("R2_LOW")
        elif r2_val < r2_green:
            flags.append("R2_MED")
    else:
        flags.append("R2_NAN")

    # 分级：优先 FAIL
    if ("FIT_FAIL" in flags) or ("PARAM_NAN" in flags):
        qc_level = "FAIL"
    elif ("INSUFF_POINTS" in flags) or ("CVU_HIGH" in flags) or ("R2_LOW" in flags) or ("Z0_OUT_OF_BOUNDS" in flags) or ("D_TOO_LARGE" in flags) or ("D_NONPOS" in flags) or ("Z0_NONPOS" in flags):
        qc_level = "RED"
    else:
        # 没有致命问题，再区分 GREEN/AMBER
        if (np.isfinite(cv_u) and cv_u <= cv_green) and (np.isfinite(r2_val) and r2_val >= r2_green):
            qc_level = "GREEN"
        else:
            qc_level = "AMBER"

    qc_flags = ";".join(flags) if flags else ""
    return qc_level, qc_flags


# =============================================================================
# 2. I/O：读 CSV、读 meta/cnm、定位 profile
# =============================================================================

def _read_csv(path: str, sep: str = ",", encoding: Optional[str] = None) -> pd.DataFrame:
    """
    读 CSV 的统一入口（带兜底编码）：
    - 如果 encoding 为 None，会依次尝试 utf-8 / utf-8-sig / gb18030
    - sep 从配置传入（默认逗号）
    """
    # 用户显式指定 encoding：直接用；失败再 fallback
    tried: List[str] = []
    if encoding:
        tried.append(encoding)

    # 常用兜底编码
    for enc in ["utf-8", "utf-8-sig", "gb18030"]:
        if enc not in tried:
            tried.append(enc)

    last_err = None
    for enc in tried:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    # 全失败：抛出最后一个错误
    raise last_err  # type: ignore


def _load_meta(meta_csv: str, meta_cols: Dict[str, str], sep: str = ",", encoding: Optional[str] = None) -> pd.DataFrame:
    """
    读取 meta_csv 并抽取必要字段：
    - meta_name
    - hmax
    - hmean
    可选字段：
    - lp
    - lf
    - hstd
    """
    df = _read_csv(meta_csv, sep=sep, encoding=encoding)

    # 必需列
    need = [meta_cols["meta_name"], meta_cols["hmax"], meta_cols["hmean"]]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"meta_csv 缺少必要列：{c}，当前列：{list(df.columns)}")

    # 可选列（有就读，没有就填 NaN）
    opt_keys = ["lp", "lf", "hstd"]
    opt_src_cols = {}
    for k in opt_keys:
        src = meta_cols.get(k, None)
        if src and (src in df.columns):
            opt_src_cols[k] = src

    keep_cols = need + list(opt_src_cols.values())
    out = df[keep_cols].copy()

    rename_map = {
        meta_cols["meta_name"]: "meta_name",
        meta_cols["hmax"]: "hmax",
        meta_cols["hmean"]: "hmean",
    }
    for k, src in opt_src_cols.items():
        rename_map[src] = k  # 直接命名为 lp/lf

    out = out.rename(columns=rename_map)

    out["meta_name"] = out["meta_name"].astype(str).str.strip()
    out["hmax"] = pd.to_numeric(out["hmax"], errors="coerce")
    out["hmean"] = pd.to_numeric(out["hmean"], errors="coerce")

    # 可选列数值化（若不存在则补 NaN）
    for k in ["lp", "lf", "hstd"]:
        if k in out.columns:
            out[k] = pd.to_numeric(out[k], errors="coerce")
        else:
            out[k] = np.nan

    return out


def _load_cnm(cnm_path: str, cnm_cols: Dict[str, str], sep: str = ",", encoding: Optional[str] = None) -> pd.DataFrame:
    """
    读取 meta_name <-> folder_name 映射表
    必须包含：
    - meta_name
    - folder_name
    """
    df = _read_csv(cnm_path, sep=sep, encoding=encoding)

    need = [cnm_cols["meta_name"], cnm_cols["folder_name"]]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"cnm_path 缺少必要列：{c}，当前列：{list(df.columns)}")

    out = df[[cnm_cols["meta_name"], cnm_cols["folder_name"]]].copy()
    out = out.rename(columns={
        cnm_cols["meta_name"]: "meta_name",
        cnm_cols["folder_name"]: "folder_name",
    })

    out["meta_name"] = out["meta_name"].astype(str).str.strip()
    out["folder_name"] = out["folder_name"].astype(str).str.strip()
    return out


def _build_case_table(meta_df: pd.DataFrame, cnm_df: pd.DataFrame) -> pd.DataFrame:
    """
    将 meta 与 cnm 做 merge，得到统一 case 表：
    meta_name, folder_name, hmax, hmean
    """
    merged = cnm_df.merge(meta_df, on="meta_name", how="left")
    return merged


def _profile_path_in_folder(folder_path: str, folder_name: str, profile_pattern: str) -> Optional[str]:
    """
    在 case 文件夹内定位 profile 文件：
    1) 优先使用 profile_pattern（支持 {folder_name} 或 {folder}）
    2) 找不到则兜底搜索 profile-*.csv
    """
    filename = profile_pattern.format(folder_name=folder_name, folder=folder_name)
    cand = os.path.join(folder_path, filename)
    if os.path.isfile(cand):
        return cand

    hits = glob.glob(os.path.join(folder_path, "profile-*.csv"))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # 多个候选：优先包含 folder_name 的
        for h in hits:
            if folder_name in os.path.basename(h):
                return h
        return hits[0]
    return None


def _discover_jobs(
    root: str,
    fg: int,
    case_table: pd.DataFrame,
    profile_pattern: str,
) -> List[Dict[str, Any]]:
    """
    根据 case_table 在 root 下构造任务列表 jobs。
    fg: ideal=0 / real=1
    """
    jobs: List[Dict[str, Any]] = []

    if not os.path.isdir(root):
        warnings.warn(f"root 不存在或不是目录：{root}")
        return jobs

    for _, row in case_table.iterrows():
        meta_name = str(row["meta_name"])
        folder_name = str(row["folder_name"])
        hmax = _safe_float(row.get("hmax", np.nan))
        hmean = _safe_float(row.get("hmean", np.nan))
        lp = _safe_float(row.get("lp", np.nan))
        lf = _safe_float(row.get("lf", np.nan))
        hstd = _safe_float(row.get("hstd", np.nan))

        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            # 该 root 下没有这个 case，就不生成 job，避免 ideal/real 双遍历导致翻倍
            continue

        profile_path = _profile_path_in_folder(folder_path, folder_name, profile_pattern)

        jobs.append({
            "meta_name": meta_name,
            "folder_name": folder_name,
            "fg": int(fg),
            "hmax": hmax,
            "hmean": hmean,
            "folder_path": folder_path,
            "profile_path": profile_path,
            "lp": lp,
            "lf": lf,
            "hstd": hstd,
        })
    print(f"[INFO] jobs: total={len(jobs)}, ideal={sum(j['fg']==0 for j in jobs)}, real={sum(j['fg']==1 for j in jobs)}")
    return jobs


def _read_profile(
    profile_path: str,
    profile_cols: Dict[str, str],
    sep: str = ",",
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """
    读取 profile CSV 并做列名统一与数值转换。
    必须列：z, u, Mom
    可选列：Tuw, Duw, v, U
    """
    df = _read_csv(profile_path, sep=sep, encoding=encoding)

    # 将用户配置的列名映射到内部标准列名：z,u,Mom,Tuw,Duw,v,U
    rename_map = {}
    for key in ["z", "u", "Mom", "Tuw", "Duw", "v", "U"]:
        colname = profile_cols.get(key, key)
        if colname in df.columns:
            rename_map[colname] = key
    df = df.rename(columns=rename_map)

    # 检查必要列
    for need in ["z", "u", "Mom"]:
        if need not in df.columns:
            raise ValueError(f"profile 缺少必要列 {need}，文件：{profile_path}，当前列：{list(df.columns)}")

    # 数值化
    for c in ["z", "u", "Mom", "Tuw", "Duw", "v", "U"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 丢弃必要列缺失行
    df = df.dropna(subset=["z", "u", "Mom"]).copy()
    df = df.sort_values("z").reset_index(drop=True)
    return df


# =============================================================================
# 3. 单 case：完整流程（含 Step4/Step5 + QC）
# =============================================================================

def _run_case(
    profile_df: Optional[pd.DataFrame],
    meta_name: str,
    fg: int,
    hmax: float,
    hmean: float,
    compute: Dict[str, Any],
) -> Dict[str, Any]:
    """
    对单个 case 执行完整流程，并返回一行结果（dict）。
    任何异常尽量不中断批处理，通过 qc_level/qc_flags 标记。
    """

    # -------------------- 从 compute 读取参数（未给则用默认建议） --------------------
    kappa = float(compute.get("kappa", 0.4))
    nmin = int(compute.get("nmin", 10))

    # QC 阈值
    cv_green = float(compute.get("cv_green", 0.10))
    cv_red = float(compute.get("cv_red", 0.15))
    r2_green = float(compute.get("r2_green", 0.90))
    r2_red = float(compute.get("r2_red", 0.85))

    # 拟合边界与初值策略
    z0_min = float(compute.get("z0_min", 1e-6))
    z0_max = float(compute.get("z0_max", 1e2))
    d_upper_frac = float(compute.get("d_upper_frac", 0.9))
    d0_frac_zlow = float(compute.get("d0_frac_zlow", 0.7))
    z00_frac = float(compute.get("z00_frac", 0.1))

    # u*eff 聚合方式
    ustar_agg = str(compute.get("ustar_agg", "median")).lower().strip()

    # -------------------- 初始化输出行（保证任何情况下都有返回） --------------------
    out: Dict[str, Any] = {
        "meta_name": meta_name,     # 输出用 case 名（按 meta_name）
        "fg": int(fg),              # ideal=0 / real=1
        "hmax": hmax,
        "hmean": hmean,
        "z_low": np.nan,
        "z_high": np.nan,
        "n_fit": 0,
        "ustar_eff": np.nan,
        "d": np.nan,
        "z0": np.nan,
        "r2": np.nan,
        "rmse": np.nan,
        "cv_u": np.nan,
        "cv_mom": np.nan,
        "G": np.nan,
        "G_star": np.nan,
        "fit_success": False,
        "fit_message": "",
        "qc_level": "FAIL",
        "qc_flags": "",
    }

    # -------------------- 1) meta 检查：hmax/hmean --------------------
    if not (np.isfinite(hmax) and np.isfinite(hmean)):
        out["qc_flags"] = "MISSING_META_HEIGHT"
        return out

    # -------------------- 2) profile 检查 --------------------
    if profile_df is None or profile_df.empty:
        out["qc_flags"] = "MISSING_PROFILE"
        return out

    # -------------------- 3) 计算 Z_kanda 并截取 band --------------------
    z_low, z_high = _z_kanda(hmax, hmean)
    out["z_low"], out["z_high"] = z_low, z_high

    band = _select_band(profile_df, "z", z_low, z_high)
    n_fit = int(len(band))
    out["n_fit"] = n_fit

    if n_fit < 2:
        out["qc_flags"] = "NO_POINTS_IN_ZKANDA"
        return out

    # -------------------- 4) Step2：u*(z) 与 u*eff --------------------
    mom = band["Mom"].to_numpy(dtype=float)
    ustar = _ustar_from_mom(mom)

    if ustar_agg == "mean":
        ustar_eff = float(np.mean(ustar))
    else:
        # 默认 median（更稳健）
        ustar_eff = float(np.median(ustar))
    out["ustar_eff"] = ustar_eff

    # -------------------- 5) Step5：稳定性指标（CV） --------------------
    out["cv_u"] = _cv(ustar)
    out["cv_mom"] = _cv(np.abs(mom))

    # -------------------- 6) Step4：通量梯度检验（dMom/dz） --------------------
    z_arr = band["z"].to_numpy(dtype=float)
    dMom_dz = _central_diff(mom, z_arr)

    if dMom_dz.size > 0:
        G = float(np.nanmedian(np.abs(dMom_dz)))  # 汇总梯度（median）
        out["G"] = G

        dz_mean = float(np.nanmean(np.diff(z_arr))) if z_arr.size >= 2 else np.nan
        mom_mean = float(np.nanmean(np.abs(mom)))
        if np.isfinite(dz_mean) and np.isfinite(mom_mean) and mom_mean > 0:
            out["G_star"] = float(G * dz_mean / mom_mean)

    # -------------------- 7) Step3：固定 u*eff 拟合 d 与 z0 --------------------
    u_obs = band["u"].to_numpy(dtype=float)

    fit = _fit_d_z0(
        z=z_arr,
        u=u_obs,
        ustar_eff=ustar_eff,
        kappa=kappa,
        z0_bounds=(z0_min, z0_max),
        d_upper_frac=d_upper_frac,
        d0_frac_zlow=d0_frac_zlow,
        z00_frac=z00_frac,
    )

    out["d"] = fit.d
    out["z0"] = fit.z0
    out["r2"] = fit.r2
    out["rmse"] = fit.rmse
    out["fit_success"] = bool(fit.success)
    out["fit_message"] = fit.message

    # -------------------- 8) QC：分级与标记 --------------------
    qc_level, qc_flags = _qc_classify(
        n_fit=n_fit,
        cv_u=out["cv_u"],
        r2_val=out["r2"],
        d=out["d"],
        z0=out["z0"],
        z_low=float(np.min(z_arr)) if z_arr.size > 0 else np.nan,
        z0_bounds=(z0_min, z0_max),
        cv_green=cv_green,
        cv_red=cv_red,
        r2_green=r2_green,
        r2_red=r2_red,
        nmin=nmin,
        fit_success=out["fit_success"],
    )
    out["qc_level"] = qc_level
    out["qc_flags"] = qc_flags

    return out

def _resolve_path(value: Any, paths: Dict[str, Any]) -> Any:
    """
    解析类似 '@paths.meta_csv' 这样的引用：
    - 若 value 是字符串且以 '@paths.' 开头，则从 paths 中取对应键并返回
    - 否则原样返回
    """
    if isinstance(value, str) and value.startswith("@paths."):
        key = value[len("@paths."):]
        return paths.get(key, value)
    return value



# =============================================================================
# 4. 总入口：批处理 ideal/real，输出结果
# =============================================================================

def Kanda_method_to_Urban(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    总入口函数（main.py 一行调用即可）：

        df = Kanda_method_to_Urban(cfg)

    本函数将：
    1) 从 cfg["urban_recal"] 读取本流程专用配置（root/profile/output/columns/compute）
    2) 读取 meta_csv 与 cnm_path，建立 meta_name <-> folder_name 映射
    3) 对 ideal_root 与 real_root 下所有 case 批处理计算
    4) 将结果写入输出目录（若配置了 out_dir），并返回结果 DataFrame
    """

    # -------------------- 0) 读取配置块 --------------------
    if "urban_recal" not in cfg:
        raise KeyError("cfg 中缺少 `urban_recal` 模块。请按我们约定的 YAML 结构添加。")

    urban = cfg["urban_recal"]
    paths = cfg.get("paths", {})  # 用于兼容：若你仍想用 paths.urban_recal_path 可兜底

    # root/profile/output/columns/compute 都从 urban_recal 读取
    root_cfg = urban.get("root", {})
    profile_cfg = urban.get("profile", {})
    output_cfg = urban.get("output", {})
    cols_cfg = urban.get("columns", {})
    compute = urban.get("compute", {})

    # -------------------- 1) 输入路径 --------------------
    ideal_root = _resolve_path(root_cfg.get("ideal_root", ""), paths)
    real_root  = _resolve_path(root_cfg.get("real_root", ""), paths)
    meta_csv   = _resolve_path(root_cfg.get("meta_csv", ""), paths)
    cnm_path   = _resolve_path(root_cfg.get("cnm_path", ""), paths)

    # profile 文件名模板
    profile_pattern = profile_cfg.get("profile_pattern", "profile-{folder_name}.csv")
    # profile 读取参数（分隔符/编码）
    profile_sep = profile_cfg.get("sep", ",")
    profile_encoding = profile_cfg.get("encoding", None)

    # meta/cnm 读取参数（可选：你如果需要，也可以在 YAML 里加 meta_encoding/map_encoding）
    meta_sep = profile_cfg.get("sep", ",")  # 这里沿用同一个 sep；你也可以单独配置
    meta_encoding = compute.get("meta_encoding", None)  # 兼容预留
    map_encoding = compute.get("map_encoding", None)    # 兼容预留

    # -------------------- 2) 输出路径 --------------------
    # 优先使用 urban_recal.output.out_dir；若为空，兜底用 paths.urban_recal_path
    output_dir = _resolve_path(output_cfg.get("out_dir", ""), paths) or paths.get("urban_recal_path", "")
    output_name = output_cfg.get("out_name", "kanda_method1_results.csv")

    # 输出开关
    save_split_fg = bool(output_cfg.get("save_split_fg", True))
    save_qc_summary = bool(output_cfg.get("save_qc_summary", True))
    save_profile_path = bool(output_cfg.get("save_profile_path", True))
    float_format = output_cfg.get("float_format", None)

    # -------------------- 3) 列名映射（专供本流程使用） --------------------
    profile_cols = cols_cfg.get("profile", {"z": "z", "u": "u", "Mom": "Mom", "Tuw": "Tuw", "Duw": "Duw", "v": "v", "U": "U"})
    meta_cols = cols_cfg.get("meta", {"meta_name": "meta_name", "hmax": "hmax", "hmean": "hmean"})
    cnm_cols = cols_cfg.get("cnm", {"meta_name": "meta_name", "folder_name": "folder_name"})

    # -------------------- 4) 读取 meta 与 cnm 并构造 case_table --------------------
    meta_df = _load_meta(meta_csv, meta_cols, sep=meta_sep, encoding=meta_encoding)
    cnm_df = _load_cnm(cnm_path, cnm_cols, sep=meta_sep, encoding=map_encoding)
    case_table = _build_case_table(meta_df, cnm_df)

    # -------------------- 5) 构造 ideal/real 的 jobs 列表 --------------------
    jobs: List[Dict[str, Any]] = []
    if ideal_root:
        jobs.extend(_discover_jobs(ideal_root, fg=0, case_table=case_table, profile_pattern=profile_pattern))
    if real_root:
        jobs.extend(_discover_jobs(real_root, fg=1, case_table=case_table, profile_pattern=profile_pattern))

    # -------------------- 6) 批处理计算 --------------------
    results: List[Dict[str, Any]] = []

    for job in jobs:
        meta_name = job["meta_name"]
        fg = int(job["fg"])
        hmax = float(job["hmax"]) if np.isfinite(job["hmax"]) else float("nan")
        hmean = float(job["hmean"]) if np.isfinite(job["hmean"]) else float("nan")
        lp = _safe_float(job.get("lp", np.nan))
        lf = _safe_float(job.get("lf", np.nan))
        hstd = _safe_float(job.get("hstd", np.nan))
        profile_path = job.get("profile_path", None)
        profile_df = None

        if profile_path is not None and os.path.isfile(profile_path):
            try:
                # 读取 profile 并统一列名
                profile_df = _read_profile(
                    profile_path=profile_path,
                    profile_cols=profile_cols,
                    sep=profile_sep,
                    encoding=profile_encoding,
                )
            except Exception as e:
                # profile 读取失败：不中断，返回 FAIL 结果
                row = {
                    "meta_name": meta_name,
                    "fg": fg,
                    "hmax": hmax,
                    "hmean": hmean,
                    "lp": lp,
                    "lf": lf,
                    "hstd": hstd,
                    "z_low": np.nan,
                    "z_high": np.nan,
                    "n_fit": 0,
                    "ustar_eff": np.nan,
                    "d": np.nan,
                    "z0": np.nan,
                    "r2": np.nan,
                    "rmse": np.nan,
                    "cv_u": np.nan,
                    "cv_mom": np.nan,
                    "G": np.nan,
                    "G_star": np.nan,
                    "fit_success": False,
                    "fit_message": f"profile_read_error: {e}",
                    "qc_level": "FAIL",
                    "qc_flags": "PROFILE_READ_ERROR",
                }
                if save_profile_path:
                    row["profile_path"] = profile_path
                results.append(row)
                continue

        # 正常跑单 case（profile_df 为空也会被 QC 标记）
        row = _run_case(
            profile_df=profile_df,
            meta_name=meta_name,
            fg=fg,
            hmax=hmax,
            hmean=hmean,
            compute=compute,
        )

        # 追加 meta 中的形态参数，保证输出可直接用于验证
        row["lp"] = lp
        row["lf"] = lf
        row["hstd"] = hstd

        if save_profile_path:
            row["profile_path"] = profile_path if profile_path else ""

        results.append(row)

    results_df = pd.DataFrame(results)

    # -------------------- 7) 保存输出（如果配置了 out_dir） --------------------
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_name)

        # float_format 用于控制输出的小数位（若为 None，则 pandas 默认）
        results_df.to_csv(
            out_path,
            index=False,
            encoding="utf-8-sig",
            float_format=float_format if float_format else None,
        )

        # 额外输出：ideal / real 分开
        if save_split_fg:
            results_df[results_df["fg"] == 0].to_csv(
                os.path.join(output_dir, "kanda_method1_ideal.csv"),
                index=False,
                encoding="utf-8-sig",
                float_format=float_format if float_format else None,
            )
            results_df[results_df["fg"] == 1].to_csv(
                os.path.join(output_dir, "kanda_method1_real.csv"),
                index=False,
                encoding="utf-8-sig",
                float_format=float_format if float_format else None,
            )

        # 额外输出：QC 汇总
        if save_qc_summary:
            qc_sum = results_df.groupby(["fg", "qc_level"]).size().reset_index(name="count")
            qc_sum.to_csv(
                os.path.join(output_dir, "kanda_method1_qc_summary.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    return results_df
