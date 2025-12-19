# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Set, Tuple


def _resolve_path(cfg: Dict[str, Any], val: Any) -> Any:
    """解析 '@paths.xxx' 写法。"""
    if isinstance(val, str) and val.startswith("@paths."):
        key = val.split(".", 1)[1]
        return cfg["paths"][key]
    return val


def _ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def _safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)
    return s


def _parse_fg_keep(fg_keep: Any) -> Optional[Set[int]]:
    """
    支持 fg_keep 写法：
      - 0 / 1
      - "0" / "1"
      - "both" / "all" / None
      - [0] / [1] / [0,1]
    返回：None 表示不筛选；否则返回 set({0,1})
    """
    if fg_keep is None:
        return None
    if isinstance(fg_keep, str):
        s = fg_keep.strip().lower()
        if s in ("both", "all", "01", "0,1", "0|1"):
            return {0, 1}
        if s in ("0", "fg0"):
            return {0}
        if s in ("1", "fg1"):
            return {1}
        parts = [p.strip() for p in s.replace("|", ",").split(",") if p.strip()]
        vals: Set[int] = set()
        for p in parts:
            try:
                vals.add(int(float(p)))
            except Exception:
                pass
        return vals if vals else None

    if isinstance(fg_keep, (int, float, np.integer, np.floating)):
        return {int(fg_keep)}

    if isinstance(fg_keep, (list, tuple, set)):
        vals: Set[int] = set()
        for v in fg_keep:
            try:
                vals.add(int(float(v)))
            except Exception:
                pass
        return vals if vals else None

    return None


def _kde_1d_numpy(x: np.ndarray, grid: np.ndarray, bw_method: Optional[Any] = None) -> np.ndarray:
    """
    纯 numpy 版 1D KDE（避免依赖 scipy）。
    - bw_method: None / "scott" / "silverman" / float
    """
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.zeros_like(grid)

    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if std <= 0:
        return np.zeros_like(grid)

    if bw_method is None or (isinstance(bw_method, str) and bw_method.lower() == "scott"):
        h = std * (n ** (-1.0 / 5.0))
    elif isinstance(bw_method, str) and bw_method.lower() == "silverman":
        h = 0.9 * std * (n ** (-1.0 / 5.0))
    else:
        try:
            h = float(bw_method)
        except Exception:
            h = std * (n ** (-1.0 / 5.0))

    if h <= 0:
        return np.zeros_like(grid)

    # KDE: mean Gaussian kernels
    z = (grid[:, None] - x[None, :]) / h
    dens = np.exp(-0.5 * z * z).mean(axis=1) / (h * np.sqrt(2.0 * np.pi))
    return dens


def _compute_stats(df: pd.DataFrame, cols: List[str], quantiles: List[float]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        n_total = int(len(s))
        n_valid = int(s.notna().sum())
        row = {
            "column": c,
            "n_total": n_total,
            "n_valid": n_valid,
            "missing": n_total - n_valid,
            "mean": float(s.mean()) if n_valid else np.nan,
            "std": float(s.std(ddof=1)) if n_valid > 1 else np.nan,
            "min": float(s.min()) if n_valid else np.nan,
            "max": float(s.max()) if n_valid else np.nan,
        }
        if n_valid:
            qs = s.quantile(quantiles)
            for q, v in qs.items():
                row[f"q{q:.2f}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_hist_kde(
    s: pd.Series,
    out_png: str,
    *,
    bins: int = 60,
    figsize: Tuple[float, float] = (4, 3),
    dpi: int = 150,
    kde_enable: bool = True,
    kde_grid: int = 400,
    kde_bw_method: Optional[Any] = None,
    title: Optional[str] = None,
):
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
    if x.size == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # 更密的直方图：density=True 便于叠加 KDE
    ax.hist(x, bins=bins, density=True)

    if kde_enable and x.size >= 5:
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        grid = np.linspace(xmin, xmax, int(kde_grid))
        dens = _kde_1d_numpy(x, grid, bw_method=kde_bw_method)
        ax.plot(grid, dens)

    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_xlabel(s.name)
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)

    _ensure_dir(os.path.dirname(out_png))
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _run_one_dist_check(
    df: pd.DataFrame,
    out_dir: str,
    *,
    columns: Optional[List[str]],
    quantiles: List[float],
    plot_enable: bool,
    bins: int,
    figsize: Tuple[float, float],
    dpi: int,
    kde_enable: bool,
    kde_grid: int,
    kde_bw_method: Optional[Any],
    tag: str,
):
    _ensure_dir(out_dir)

    # 自动列选择：如果 columns 为空 -> 选所有数值列
    if not columns:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in num_cols if c.lower() not in ("fg",)]
    else:
        # 只保留存在的列
        columns = [c for c in columns if c in df.columns]

    if not columns:
        print(f"[dist_check] {tag}: 没有可检查列，跳过")
        return

    # 统计输出
    stats_df = _compute_stats(df, columns, quantiles)
    stats_path = os.path.join(out_dir, f"stats_summary_{_safe_name(tag)}.csv")
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"[dist_check] {tag}: stats -> {os.path.abspath(stats_path)}")

    # 绘图输出
    if plot_enable:
        fig_dir = os.path.join(out_dir, f"figs_{_safe_name(tag)}")
        _ensure_dir(fig_dir)

        for c in columns:
            out_png = os.path.join(fig_dir, f"{_safe_name(c)}.png")
            _plot_hist_kde(
                df[c],
                out_png,
                bins=bins,
                figsize=figsize,
                dpi=dpi,
                kde_enable=kde_enable,
                kde_grid=kde_grid,
                kde_bw_method=kde_bw_method,
                title=f"{c} ({tag})",
            )
        print(f"[dist_check] {tag}: figs -> {os.path.abspath(fig_dir)}")


def run_dist_check_from_cfg(cfg: Dict[str, Any]) -> None:
    """
    从 cfg['dist_check'] 读取配置，输出：
      - stats_summary_*.csv
      - figs_*/xxx.png（直方图更密 + KDE 曲线）
    并支持按 fg 分组/筛选（fg_filter）。
    """
    sec = cfg.get("dist_check", None)
    if not sec:
        print("[dist_check] dist_check 未配置，跳过")
        return
    if sec.get("enable") is False:
        print("[dist_check] dist_check.enable = False，跳过")
        return

    in_csv = _resolve_path(cfg, sec.get("in_csv"))
    out_dir = _resolve_path(cfg, sec.get("out_dir"))
    if not in_csv:
        raise ValueError("[dist_check] dist_check.in_csv 未设置")
    if not out_dir:
        raise ValueError("[dist_check] dist_check.out_dir 未设置")

    print("[dist_check] 读取 CSV:", os.path.abspath(in_csv))
    df = pd.read_csv(in_csv)

    # columns
    columns = sec.get("columns", None)

    # stats
    stats_cfg = sec.get("stats", {}) or {}
    quantiles = stats_cfg.get("quantiles", [0.05, 0.25, 0.5, 0.75, 0.95])
    quantiles = [float(q) for q in quantiles]

    # plot
    plot_cfg = sec.get("plot", {}) or {}
    plot_enable = bool(plot_cfg.get("enable", True))
    bins = int(plot_cfg.get("bins", 60))
    figsize = tuple(plot_cfg.get("figsize", [4, 3]))
    dpi = int(plot_cfg.get("dpi", 150))

    kde_cfg = plot_cfg.get("kde", {}) or {}
    kde_enable = bool(kde_cfg.get("enable", True))
    kde_grid = int(kde_cfg.get("grid", 400))
    kde_bw_method = kde_cfg.get("bw_method", None)

    # fg filter
    fg_cfg = sec.get("fg_filter", {}) or {}
    fg_enable = bool(fg_cfg.get("enable", False))
    fg_col = fg_cfg.get("fg_col", "fg")
    fg_keep = fg_cfg.get("keep", None)          # 0/1/[0]/[1]/[0,1]/both/None
    output_mode = fg_cfg.get("output_mode", "split").strip().lower()

    _ensure_dir(out_dir)

    # 不启用 fg：直接跑一套
    if not fg_enable:
        _run_one_dist_check(
            df, out_dir,
            columns=columns,
            quantiles=quantiles,
            plot_enable=plot_enable,
            bins=bins,
            figsize=figsize,
            dpi=dpi,
            kde_enable=kde_enable,
            kde_grid=kde_grid,
            kde_bw_method=kde_bw_method,
            tag="all",
        )
        return

    # 启用 fg：筛选/分组
    if fg_col not in df.columns:
        raise ValueError(f"[dist_check] fg_filter.enable=True 但 fg_col='{fg_col}' 不在 CSV 列中")

    df[fg_col] = pd.to_numeric(df[fg_col], errors="coerce")
    keep_set = _parse_fg_keep(fg_keep)

    if keep_set is None:
        # keep 为空：不筛选，仍可 split
        keep_set = {0, 1}

    if output_mode == "single":
        sub = df[df[fg_col].isin(list(keep_set))].copy()
        tag = f"fg_{'_'.join(map(str, sorted(list(keep_set))))}"
        _run_one_dist_check(
            sub, out_dir,
            columns=columns,
            quantiles=quantiles,
            plot_enable=plot_enable,
            bins=bins,
            figsize=figsize,
            dpi=dpi,
            kde_enable=kde_enable,
            kde_grid=kde_grid,
            kde_bw_method=kde_bw_method,
            tag=tag,
        )
    else:
        # split：每个 fg 单独输出
        for g in sorted(list(keep_set)):
            sub = df[df[fg_col] == g].copy()
            tag = f"fg_{g}"
            _run_one_dist_check(
                sub, out_dir,
                columns=columns,
                quantiles=quantiles,
                plot_enable=plot_enable,
                bins=bins,
                figsize=figsize,
                dpi=dpi,
                kde_enable=kde_enable,
                kde_grid=kde_grid,
                kde_bw_method=kde_bw_method,
                tag=tag,
            )
