
# plot_rec.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, Dict, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 解析 @paths.xxx 引用（供 main.py 在 yaml.safe_load 后调用）
# ============================================================
def _get_by_dotted_key(root: Dict[str, Any], dotted: str) -> Any:
    cur: Any = root
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"无法解析引用 '@{dotted}'：缺少键 '{part}'。")
        cur = cur[part]
    return cur


def resolve_cfg_refs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度递归解析 cfg 中所有字符串引用：
      - "@paths.xxx" -> cfg["paths"]["xxx"]
      - 更一般地支持 "@a.b.c"

    注意：你在 main.py 中保留 cfg = yaml.safe_load(f) 不变，
    只需在那之后调用：cfg = resolve_cfg_refs(cfg)
    """
    def _resolve_one(v: Any, root: Dict[str, Any]) -> Any:
        if isinstance(v, str) and v.startswith("@"):
            return _get_by_dotted_key(root, v[1:])
        return v

    def walk(obj: Any, root: Dict[str, Any]) -> Any:
        if isinstance(obj, dict):
            return {k: walk(v, root) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v, root) for v in obj]
        return _resolve_one(obj, root)

    out = cfg
    for _ in range(10):  # 防止循环引用
        new_out = walk(out, out)
        if new_out == out:
            break
        out = new_out

    return out


# ============================================================
# 绘图：两份数据 + fg 分类 + 两张图
# ============================================================
def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"不支持的文件类型: {ext}（仅支持 csv/xlsx/xls）")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _require_columns(df: pd.DataFrame, cols: list, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] 缺少必要列: {missing}. 当前列: {list(df.columns)}")


def _to_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _clean_fg(df: pd.DataFrame, fg_col: str, name: str) -> pd.DataFrame:
    df = df.copy()
    df[fg_col] = pd.to_numeric(df[fg_col], errors="coerce")
    df = df[df[fg_col].isin([0, 1])].copy()
    if df.empty:
        raise ValueError(f"[{name}] 过滤后 fg=0/1 数据为空，请检查 {fg_col} 列。")
    df[fg_col] = df[fg_col].astype(int)
    return df


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = num / den
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _scatter_group(
    ax,
    x: str,
    y: str,
    df: pd.DataFrame,
    marker: str,
    filled: bool,
    label: str,
    size: float = 28.0,
):
    """
    不指定固定颜色：使用 matplotlib 默认 color cycle。
    filled=True  -> 实心（idealized, fg=0）
    filled=False -> 空心（realistic, fg=1）
    """
    color = ax._get_lines.get_next_color()
    xvals = df[x].to_numpy()
    yvals = df[y].to_numpy()

    if filled:
        ax.scatter(
            xvals,
            yvals,
            s=size,
            marker=marker,
            c=[color],
            edgecolors=[color],
            linewidths=0.8,
            label=label,
        )
    else:
        ax.scatter(
            xvals,
            yvals,
            s=size,
            marker=marker,
            facecolors="none",
            edgecolors=[color],
            linewidths=0.9,
            label=label,
        )


def _plot_four_groups(
    ax,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    x: str,
    y: str,
    label1: str,
    label2: str,
    marker1: str = "o",
    marker2: str = "s",
    fg_col: str = "fg",
):
    """
    4 组（数据来源 × fg）：
      - df1 fg=0 (实心, marker1)
      - df1 fg=1 (空心, marker1)
      - df2 fg=0 (实心, marker2)
      - df2 fg=1 (空心, marker2)
    """
    g10 = df1[df1[fg_col] == 0]
    g11 = df1[df1[fg_col] == 1]
    g20 = df2[df2[fg_col] == 0]
    g21 = df2[df2[fg_col] == 1]

    if len(g10) > 0:
        _scatter_group(ax, x, y, g10, marker=marker1, filled=True,
                       label=f"{label1} | idealized (fg=0)")
    if len(g11) > 0:
        _scatter_group(ax, x, y, g11, marker=marker1, filled=False,
                       label=f"{label1} | realistic (fg=1)")
    if len(g20) > 0:
        _scatter_group(ax, x, y, g20, marker=marker2, filled=True,
                       label=f"{label2} | idealized (fg=0)")
    if len(g21) > 0:
        _scatter_group(ax, x, y, g21, marker=marker2, filled=False,
                       label=f"{label2} | realistic (fg=1)")

    ax.grid(True, linewidth=0.6, alpha=0.4)


def _apply_limits(ax, lim_cfg: Optional[Dict[str, Any]]):
    """
    lim_cfg:
      xlim: [xmin, xmax]
      ylim: [ymin, ymax]
    任意一个不提供则不设置。
    """
    if not lim_cfg:
        return

    xlim = lim_cfg.get("xlim")
    ylim = lim_cfg.get("ylim")

    if isinstance(xlim, (list, tuple)) and len(xlim) == 2:
        ax.set_xlim(xlim[0], xlim[1])
    if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
        ax.set_ylim(ylim[0], ylim[1])


def _merge_unique_legend(fig, axes, loc="upper center", ncol=2):
    handles: list = []
    labels: list = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l

    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h

    if uniq:
        fig.legend(list(uniq.values()), list(uniq.keys()), loc=loc, ncol=ncol, frameon=False)


def plot_two_datasets(cfg: Dict[str, Any]) -> None:
    """
    cfg 结构要求：
      cfg["plot_two_datasets"] 下包含 data1, data2, outdir 等参数
      支持 @paths.xxx（需先在 main 中调用 resolve_cfg_refs(cfg)）

    输出两张图：
      Figure1_three_subplots.png
      Figure2_two_subplots.png
    """
    sec = cfg.get("plot_two_datasets", {})
    if not isinstance(sec, dict):
        raise ValueError("cfg['plot_two_datasets'] 必须是字典。")

    data1 = sec["data1"]
    data2 = sec["data2"]
    outdir = sec.get("outdir", ".")
    os.makedirs(outdir, exist_ok=True)

    label1 = sec.get("label1", "Dataset1")
    label2 = sec.get("label2", "Dataset2")
    marker1 = sec.get("marker1", "o")
    marker2 = sec.get("marker2", "s")

    dpi = int(sec.get("dpi", 300))
    fig1_name = sec.get("fig1_name", "Figure1_three_subplots.png")
    fig2_name = sec.get("fig2_name", "Figure2_two_subplots.png")

    # 坐标轴范围（可选）
    axis_limits = sec.get("axis_limits", {}) or {}

    # 列名映射（可选）
    cols_cfg = sec.get("columns", {}) or {}

    def col(key: str) -> str:
        return str(cols_cfg.get(key, key)).strip().lower()

    lp = col("lp")
    lf = col("lf")
    have = col("have")
    hstd = col("hstd")
    hmax = col("hmax")
    d = col("d")
    z0 = col("z0")
    fg = col("fg")

    # 读表 + 列名标准化
    df1 = _normalize_columns(_read_table(data1))
    df2 = _normalize_columns(_read_table(data2))

    needed = [lp, lf, have, hstd, hmax, d, z0, fg]
    _require_columns(df1, needed, "data1")
    _require_columns(df2, needed, "data2")

    # 数值化 + fg 清洗
    df1 = _to_numeric(df1, needed)
    df2 = _to_numeric(df2, needed)
    df1 = _clean_fg(df1, fg, "data1")
    df2 = _clean_fg(df2, fg, "data2")

    # 派生量
    df1 = df1.copy()
    df2 = df2.copy()
    df1["d_over_hmax"] = _safe_ratio(df1[d], df1[hmax])
    df2["d_over_hmax"] = _safe_ratio(df2[d], df2[hmax])
    df1["z0_over_have"] = _safe_ratio(df1[z0], df1[have])
    df2["z0_over_have"] = _safe_ratio(df2[z0], df2[have])

    # ========= Figure 1: 1x3 =========
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.6))

    # 子图1：lp vs lf
    ax = axes1[0]
    _plot_four_groups(ax, df1, df2, lp, lf, label1, label2, marker1, marker2, fg_col=fg)
    ax.set_xlabel("lp")
    ax.set_ylabel("lf")
    ax.set_title("lp vs lf")
    _apply_limits(ax, axis_limits.get("fig1", {}).get("ax1_lp_lf"))

    # 子图2：have vs hstd
    ax = axes1[1]
    _plot_four_groups(ax, df1, df2, have, hstd, label1, label2, marker1, marker2, fg_col=fg)
    ax.set_xlabel("have")
    ax.set_ylabel("hstd")
    ax.set_title("have vs hstd")
    _apply_limits(ax, axis_limits.get("fig1", {}).get("ax2_have_hstd"))

    # 子图3：hstd vs hmax
    ax = axes1[2]
    _plot_four_groups(ax, df1, df2, hstd, hmax, label1, label2, marker1, marker2, fg_col=fg)
    ax.set_xlabel("hstd")
    ax.set_ylabel("hmax")
    ax.set_title("hstd vs hmax")
    _apply_limits(ax, axis_limits.get("fig1", {}).get("ax3_hstd_hmax"))

    _merge_unique_legend(fig1, axes1, loc="upper center", ncol=2)
    fig1.tight_layout(rect=[0, 0, 1, 0.88])
    out1 = os.path.join(outdir, fig1_name)
    fig1.savefig(out1, dpi=dpi)
    plt.close(fig1)

    # ========= Figure 2: 1x2 =========
    fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # 子图1：lp vs d/hmax
    ax = axes2[0]
    _plot_four_groups(
        ax,
        df1.dropna(subset=["d_over_hmax"]),
        df2.dropna(subset=["d_over_hmax"]),
        lp,
        "d_over_hmax",
        label1,
        label2,
        marker1,
        marker2,
        fg_col=fg,
    )
    ax.set_xlabel("lp")
    ax.set_ylabel("d / hmax")
    ax.set_title("lp vs d/hmax")
    _apply_limits(ax, axis_limits.get("fig2", {}).get("ax1_lp_d_over_hmax"))

    # 子图2：lp vs z0/have
    ax = axes2[1]
    _plot_four_groups(
        ax,
        df1.dropna(subset=["z0_over_have"]),
        df2.dropna(subset=["z0_over_have"]),
        lp,
        "z0_over_have",
        label1,
        label2,
        marker1,
        marker2,
        fg_col=fg,
    )
    ax.set_xlabel("lp")
    ax.set_ylabel("z0 / have")
    ax.set_title("lp vs z0/have")
    _apply_limits(ax, axis_limits.get("fig2", {}).get("ax2_lp_z0_over_have"))

    _merge_unique_legend(fig2, axes2, loc="upper center", ncol=2)
    fig2.tight_layout(rect=[0, 0, 1, 0.88])
    out2 = os.path.join(outdir, fig2_name)
    fig2.savefig(out2, dpi=dpi)
    plt.close(fig2)

    print("已输出图片：")
    print(" -", out1)
    print(" -", out2)
