
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _resolve_path(cfg, val):
    """解析 '@paths.xxx' 写法。"""
    if isinstance(val, str) and val.startswith("@paths."):
        key = val.split(".", 1)[1]
        return cfg["paths"][key]
    return val


def plot_rur_compare(in_csv, out_path):
    print("[d_z0_plot] 读取 CSV:", os.path.abspath(in_csv))
    df = pd.read_csv(in_csv)

    pairs = [
        ("  d_LES(m)",  "  d_mac(m)",    "lnd of Macdonald"),
        ("  d_LES(m)",  "  d_kanda(m)",  "lnd of Kanda"),
        ("  d_LES(m)",  "  d_rur(m)",    "lnd of RUR"),
        ("  z0_LES(m)", "  z0_mac(m)",   "lnz₀ of Macdonald"),
        ("  z0_LES(m)", "  z0_kanda(m)", "lnz₀ of Kanda"),
        ("  z0_LES(m)", "  z0_rur(m)",   "lnz₀ of RUR"),
    ]
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for idx, (obs_col, model_col, ylab) in enumerate(pairs):
        row = 0 if idx < 3 else 1
        col = idx % 3
        ax = axes[row, col]

        # ===== 关键修改开始：把字符串自动跳过 =====
        # 先取两列
        sub = df[[obs_col, model_col]].copy()

        # 强制转换为数值，转换不了的（字符串等）变为 NaN
        for c in (obs_col, model_col):
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

        # 去掉任一列为 NaN 的行（包含有字符串的点会被丢掉）
        sub = sub.dropna()

        # 只保留正值，避免 ln 出问题
        sub = sub[(sub[obs_col] > 0) & (sub[model_col] > 0)]

        # 如果这一对列没有有效数据，则跳过这个子图
        if sub.empty:
            ax.text(
                0.5, 0.5, "No valid data",
                transform=ax.transAxes,
                ha="center", va="center"
            )
            ax.set_axis_off()
            continue
        # ===== 关键修改结束 =====

        x = np.log(sub[obs_col].to_numpy())
        y = np.log(sub[model_col].to_numpy())

        ax.scatter(x, y, s=20)

        xmin = min(x.min(), y.min())
        xmax = max(x.max(), y.max())
        ax.plot([xmin, xmax], [xmin, xmax],
                linestyle='-', linewidth=0.8, color='0.8')

        # 避免除以 0 的保护（极端情况下只有一个点）
        denom = (x * x).sum()
        if denom == 0:
            a = 1.0
        else:
            a = (x * y).sum() / denom

        ax.plot([xmin, xmax], [a * xmin, a * xmax], linewidth=1.0)

        diff = y - x
        mbe = diff.mean()
        rmse = np.sqrt((diff ** 2).mean())

        ax.text(
            0.05, 0.95,
            f"Y = {a:.2f} * X\nMBE = {mbe:.3f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes,
            ha="left", va="top"
        )

        if row == 0:
            ax.set_xlabel("lnd of observation")
        else:
            ax.set_xlabel("lnz₀ of observation")
        ax.set_ylabel(ylab)

        ax.text(
            0.03, 0.07, letters[idx],
            transform=ax.transAxes,
            ha="left", va="bottom"
        )

    plt.tight_layout()

    # 确保目录存在
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    abs_out = os.path.abspath(out_path)
    print("[d_z0_plot] 保存图片到:", abs_out)
    plt.savefig(abs_out, dpi=300)
    plt.close(fig)

    if os.path.exists(abs_out):
        print("[d_z0_plot] 保存成功 ✅")
    else:
        print("[d_z0_plot] 警告：保存后未找到文件 ❌")


def run_d_z0_plot_from_cfg(cfg):
    """
    从 cfg['plots']['d_z0_compare'] 读取配置并画图。
    """
    plots_cfg = cfg.get("plots", {})
    dz0_cfg = plots_cfg.get("d_z0_compare")
    if dz0_cfg is None:
        print("[d_z0_plot] plots.d_z0_compare 未配置，跳过绘图")
        return

    if dz0_cfg.get("enable") is False:
        print("[d_z0_plot] plots.d_z0_compare.enable = False，跳过绘图")
        return

    in_csv = _resolve_path(cfg, dz0_cfg.get("in_csv"))
    out_png = _resolve_path(cfg, dz0_cfg.get("out_png", "d_z0_compare.png"))

    if in_csv is None:
        raise ValueError("plots.d_z0_compare.in_csv 未在 yaml 中设置")

    plot_rur_compare(in_csv, out_png)