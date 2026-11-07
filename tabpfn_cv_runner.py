# -*- coding: utf-8 -*-
# tabpfn_cv_runner.py
#
# 功能说明（双数据集增强版）：
# - 单 CSV：按 test_split 划分训练/测试；CV 在训练集上；最终模型评估在测试集上。
# - 多 CSV 合并：合并后按 test_split 划分；同上。
# - 双 CSV（A=source, B=external）：
#   1) 从 A 内部再划分：A_fit(80%) 训练 + A_holdout(20%) 内部测试
#   2) CV 只在 A_fit 上做
#   3) 用 A_fit 拟合一次最终模型，同时评估 A_holdout 与 B（external）
#
# 你原有的计算逻辑尽量保持不变；仅在“双 CSV”路径追加了 A 内部 80/20 + external 测试的流程，
# 并新增了 tag 版落盘函数，避免同一任务下多套测试结果覆盖。

import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tabpfn import TabPFNRegressor

# 统一的 X/Y & 切分与预处理工具
from xy_blocks import (
    build_xy_from_csv,
    build_xy_from_two_csvs,
    step_drop_constant,
    step_fillna_numeric,
    step_dropna_in,
    load_dataframe,
    inspect_df,
    preprocess_df,
    make_X_y,
    split_indices,
    split_by_indices,
)

# 尝试导入 SHAP（可选）
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    shap = None
    _HAS_SHAP = False

LOG_SHIFT = 1.0  # log 变换使用 log(y + LOG_SHIFT)


# ================== 工具与通用 ==================

def _get(cfg: Dict[str, Any], dotted: Any, default=None, *, ctx: str = ""):
    """
    支持 '@paths.xxx' 引用；若找不到对应 key，则报错提示，避免后面传 None 给 read_csv。
    ctx: 仅用于报错时标记是哪个字段（例如 'kanda_single.in_csv'）
    """
    if isinstance(dotted, str) and dotted.startswith("@"):
        m = re.match(r"@([^\.]+)\.(.+)", dotted)
        if m:
            root, key = m.group(1), m.group(2)
            sub = cfg.get(root, {})
            if key not in sub:
                raise ValueError(
                    f"_get 解析 {ctx or dotted} 失败：cfg['{root}'] 中不存在键 '{key}'。"
                    f"\n已存在的键为：{list(sub.keys())}"
                )
            return sub[key]
        else:
            raise ValueError(f"_get 解析引用字符串失败：{dotted}（格式应为 @root.key）")

    if dotted is None:
        if default is None:
            raise ValueError(f"_get 收到 None 且未提供 default（ctx={ctx}）。")
        return default
    return dotted


def _ensure_dir(path: str) -> None:
    d = os.path.abspath(path)
    os.makedirs(d, exist_ok=True)


def _io_params(cfg: Dict[str, Any]) -> Tuple[str, str]:
    params = cfg.get("params", {})
    return params.get("sep", ","), params.get("encoding", "utf-8-sig")


def _pick_device(cfg: Dict[str, Any]) -> str:
    dev = str(cfg.get("params", {}).get("device", "auto")).lower()
    if dev == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ================== 分箱重采样（只作用在训练集） ==================

def apply_binning_resample_on_target(
    X_tr: np.ndarray,
    y_tr: pd.Series,
    *,
    n_bins: int = 10,
    mode: str = "undersample",        # "undersample" 或 "balance"
    max_per_bin: Optional[int] = None,
    strategy: str = "quantile",       # "quantile" 或 "uniform"
    seed: int = 42,
) -> Tuple[np.ndarray, pd.Series]:
    """
    对训练集按目标 y_tr 分箱，并进行重采样：

    - strategy="quantile"  : 按分位数划分等频区间
    - strategy="uniform"   : 按 min~max 均匀划分

    - mode="undersample"   : 每个 bin 至多 max_per_bin，超过则随机下采样
    - mode="balance"       : 目标是让所有 bin 接近同一目标数量：
                             大于该数量则下采样，小于则有放回采样补足

    返回重采样后的 X_tr_new, y_tr_new
    """
    y = y_tr.to_numpy(dtype=float)

    # 样本太少或 y 几乎不变就不搞这些花活
    if len(y) < n_bins or np.allclose(y.min(), y.max()):
        print(f"[Binning] y 变化太小或样本数不足（n={len(y)}），跳过分箱重采样。")
        return X_tr, y_tr

    # 计算分箱边界
    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(y, qs)
    elif strategy == "uniform":
        edges = np.linspace(y.min(), y.max(), n_bins + 1)
    else:
        raise ValueError(f"[Binning] 未知 strategy={strategy}，应为 'quantile' 或 'uniform'。")

    # 极端情况下，边界都重合
    if np.allclose(edges[0], edges[-1]):
        print(f"[Binning] 分箱边界 collapse 到同一点，跳过分箱重采样。")
        return X_tr, y_tr

    # 使用中间边界做 digitize
    bin_idx = np.digitize(y, edges[1:-1], right=False)

    rng = np.random.default_rng(seed)
    unique_bins, counts = np.unique(bin_idx, return_counts=True)
    count_map = dict(zip(unique_bins, counts))

    print(f"[Binning] 原始各 bin 样本数：{count_map}")

    if max_per_bin is None:
        max_per_bin = int(np.median(list(count_map.values())))
        max_per_bin = max(max_per_bin, 1)
        print(f"[Binning] max_per_bin 未指定，自动使用中位数：{max_per_bin}")

    selected_indices: List[int] = []

    if mode == "undersample":
        for b in unique_bins:
            idxs = np.where(bin_idx == b)[0]
            if len(idxs) > max_per_bin:
                choose = rng.choice(idxs, size=max_per_bin, replace=False)
            else:
                choose = idxs
            selected_indices.append(choose)

    elif mode == "balance":
        target_n = min(max_per_bin, max(count_map.values()))
        target_n = max(target_n, 1)
        print(f"[Binning] balance 模式，每个 bin 目标样本数：{target_n}")
        for b in unique_bins:
            idxs = np.where(bin_idx == b)[0]
            if len(idxs) >= target_n:
                choose = rng.choice(idxs, size=target_n, replace=False)
            else:
                choose = rng.choice(idxs, size=target_n, replace=True)
            selected_indices.append(choose)
    else:
        raise ValueError(f"[Binning] 未知 mode={mode}，应为 'undersample' 或 'balance'。")

    selected_indices = np.concatenate(selected_indices)
    rng.shuffle(selected_indices)

    X_new = X_tr[selected_indices]
    y_new = y_tr.iloc[selected_indices].reset_index(drop=True)

    # 再统计一下新分布
    y_new_np = y_new.to_numpy(dtype=float)
    bin_idx_new = np.digitize(y_new_np, edges[1:-1], right=False)
    unique_bins_new, counts_new = np.unique(bin_idx_new, return_counts=True)
    count_map_new = dict(zip(unique_bins_new, counts_new))

    print(f"[Binning] 重采样后各 bin 样本数：{count_map_new}，总样本数由 {len(y_tr)} → {len(y_new)}")

    return X_new, y_new


# ================== 训练评估 ==================

def run_kfold_cv(
    X_tr: np.ndarray, y_tr: pd.Series, kf_num: int,
    device: str, log_mode: bool, seed: int
) -> Dict[str, Any]:
    """
    在训练集上做 K 折交叉验证；每折内标准化，返回每折指标与均值±方差。
    """
    if log_mode:
        y_min = float(y_tr.min())
        if y_min <= -LOG_SHIFT:
            raise ValueError(
                f"log_mode=True 时，训练集 y 的最小值为 {y_min:.4f}，"
                f"需要满足 y > {-LOG_SHIFT} 才能使用 log(y + {LOG_SHIFT}) 变换。"
            )

    kf = KFold(n_splits=int(kf_num), shuffle=True, random_state=seed)
    fold_rows: List[Dict[str, float]] = []

    y_tr_np = y_tr.to_numpy(dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
        scaler = StandardScaler().fit(X_tr[tr_idx])
        Xtr = scaler.transform(X_tr[tr_idx])
        Xval = scaler.transform(X_tr[val_idx])

        if log_mode:
            y_tr_fold = np.log(y_tr_np[tr_idx] + LOG_SHIFT)
            y_val_fold = np.log(y_tr_np[val_idx] + LOG_SHIFT)
        else:
            y_tr_fold = y_tr_np[tr_idx]
            y_val_fold = y_tr_np[val_idx]

        model = TabPFNRegressor(device=device, random_state=seed)
        model.fit(Xtr, y_tr_fold)
        y_val_pred = model.predict(Xval)

        if log_mode:
            y_val_pred = np.exp(y_val_pred) - LOG_SHIFT
            y_val_eval = np.exp(y_val_fold) - LOG_SHIFT
        else:
            y_val_eval = y_val_fold

        r2 = r2_score(y_val_eval, y_val_pred)
        mae = mean_absolute_error(y_val_eval, y_val_pred)
        mse = mean_squared_error(y_val_eval, y_val_pred)
        fold_rows.append({"fold": fold + 1, "r2": r2, "mae": mae, "mse": mse})

    metrics_df = pd.DataFrame(fold_rows)
    summary = {
        "cv_r2_mean": metrics_df["r2"].mean(),   "cv_r2_std":  metrics_df["r2"].std(ddof=1),
        "cv_mae_mean": metrics_df["mae"].mean(), "cv_mae_std": metrics_df["mae"].std(ddof=1),
        "cv_mse_mean": metrics_df["mse"].mean(), "cv_mse_std": metrics_df["mse"].std(ddof=1),
    }
    return {"fold_metrics": fold_rows, "summary": summary}


def fit_final_and_eval_holdout(
    X_tr: np.ndarray, y_tr: pd.Series,
    X_te: np.ndarray, y_te: pd.Series,
    device: str, log_mode: bool, seed: int,
    *,
    feature_names: Optional[List[str]] = None,
    do_shap: bool = True,
    shap_max_samples: int = 200,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    用全训练集拟合最终模型（标准化 fit 在训练集），并在留出测试集评估。
    返回 test_metrics 与 y_pred_test。
    若 do_shap=True 且已安装 shap，则在测试集上做一次 SHAP 分析，
    并将每个特征的 mean(|SHAP|) 以 'shap_<feature>' 列写入 test_metrics。
    注意：log_mode=True 时，SHAP 是针对模型输出（log(y+LOG_SHIFT)）的。
    """
    if log_mode:
        y_tr_min = float(y_tr.min())
        y_te_min = float(y_te.min())
        if y_tr_min <= -LOG_SHIFT or y_te_min <= -LOG_SHIFT:
            raise ValueError(
                f"log_mode=True 且使用 log(y + {LOG_SHIFT}) 时，"
                f"要求训练集和测试集目标满足 y > {-LOG_SHIFT}。"
                f"当前 y_train_min={y_tr_min:.4f}, y_test_min={y_te_min:.4f}"
            )

    scaler = StandardScaler().fit(X_tr)
    Xtr_full = scaler.transform(X_tr)
    Xte_full = scaler.transform(X_te)

    y_tr_np = y_tr.to_numpy(dtype=float)

    if log_mode:
        ytr = np.log(y_tr_np + LOG_SHIFT)
    else:
        ytr = y_tr_np

    model = TabPFNRegressor(device=device, random_state=seed)
    model.fit(Xtr_full, ytr)

    y_pred = model.predict(Xte_full)
    if log_mode:
        y_pred = np.exp(y_pred) - LOG_SHIFT

    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)

    test_metrics: Dict[str, float] = {
        "r2": float(r2),
        "mae": float(mae),
        "mse": float(mse),
        "n_test": int(len(y_te)),
    }

    # ========== SHAP 分析 ==========
    if do_shap and _HAS_SHAP:
        try:
            n_bg = min(100, Xtr_full.shape[0])
            bg_idx = np.random.choice(Xtr_full.shape[0], size=n_bg, replace=False)
            background = Xtr_full[bg_idx]

            n_te = Xte_full.shape[0]
            if n_te > shap_max_samples:
                te_idx = np.random.choice(n_te, size=shap_max_samples, replace=False)
                X_shap = Xte_full[te_idx]
            else:
                X_shap = Xte_full

            def f_pred(x):
                return model.predict(x)

            explainer = shap.KernelExplainer(f_pred, background)
            shap_values = explainer.shap_values(X_shap)

            shap_values = np.array(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[0]

            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            if feature_names is not None and len(feature_names) == len(mean_abs_shap):
                for fname, val in zip(feature_names, mean_abs_shap):
                    test_metrics[f"shap_{fname}"] = float(val)
            else:
                for i, val in enumerate(mean_abs_shap):
                    test_metrics[f"shap_f{i}"] = float(val)

        except Exception as e:
            print(f"[WARN] SHAP 计算失败：{e}")

    elif do_shap and not _HAS_SHAP:
        print("[WARN] 未安装 shap，跳过 SHAP 分析。请先 `pip install shap`。")

    return test_metrics, np.asarray(y_pred, dtype=float).ravel()


def fit_once_and_eval_two_sets(
    X_fit: np.ndarray, y_fit: pd.Series,
    X_eval1: np.ndarray, y_eval1: pd.Series,
    X_eval2: np.ndarray, y_eval2: pd.Series,
    *,
    device: str, log_mode: bool, seed: int,
    do_shap: bool = True,
    shap_max_samples: int = 200,
    feature_names: Optional[List[str]] = None,
    shap_on: str = "external",   # "external" or "holdout"
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, float], np.ndarray]:
    """
    在 X_fit/y_fit 上拟合一次（scaler+TabPFN），然后分别对 eval1/eval2 做预测与评估。
    返回：(metrics1, pred1, metrics2, pred2)

    说明：
    - CV 不在这里做，CV 由上层对 X_fit/y_fit 调 run_kfold_cv。
    - log_mode=True 时内部拟合在 log(y+LOG_SHIFT) 空间，输出预测会 inverse 回原空间。
    - SHAP 默认只在 external 上做一次（避免重复耗时），可通过 shap_on 切换。
    """
    if log_mode:
        y_fit_min = float(y_fit.min())
        y_e1_min = float(y_eval1.min())
        y_e2_min = float(y_eval2.min())
        if y_fit_min <= -LOG_SHIFT or y_e1_min <= -LOG_SHIFT or y_e2_min <= -LOG_SHIFT:
            raise ValueError(
                f"log_mode=True 且使用 log(y + {LOG_SHIFT}) 时，要求所有目标满足 y > {-LOG_SHIFT}。"
                f"当前 y_fit_min={y_fit_min:.4f}, y_eval1_min={y_e1_min:.4f}, y_eval2_min={y_e2_min:.4f}"
            )

    scaler = StandardScaler().fit(X_fit)
    Xfit_s = scaler.transform(X_fit)
    Xe1_s = scaler.transform(X_eval1)
    Xe2_s = scaler.transform(X_eval2)

    y_fit_np = y_fit.to_numpy(dtype=float)
    if log_mode:
        yfit = np.log(y_fit_np + LOG_SHIFT)
    else:
        yfit = y_fit_np

    model = TabPFNRegressor(device=device, random_state=seed)
    model.fit(Xfit_s, yfit)

    def _predict(Xs: np.ndarray) -> np.ndarray:
        pred = model.predict(Xs)
        if log_mode:
            pred = np.exp(pred) - LOG_SHIFT
        return np.asarray(pred, dtype=float).ravel()

    pred1 = _predict(Xe1_s)
    pred2 = _predict(Xe2_s)

    def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return {"r2": float(r2), "mae": float(mae), "mse": float(mse), "n_test": int(len(y_true))}

    m1 = _metrics(y_eval1, pred1)
    m2 = _metrics(y_eval2, pred2)

    # ========== SHAP（只做一次，默认 external） ==========
    if do_shap and _HAS_SHAP:
        try:
            X_pool = Xe1_s if shap_on == "holdout" else Xe2_s

            n_bg = min(100, Xfit_s.shape[0])
            bg_idx = np.random.choice(Xfit_s.shape[0], size=n_bg, replace=False)
            background = Xfit_s[bg_idx]

            n_pool = X_pool.shape[0]
            if n_pool > shap_max_samples:
                idx = np.random.choice(n_pool, size=shap_max_samples, replace=False)
                X_shap = X_pool[idx]
            else:
                X_shap = X_pool

            def f_pred(x):
                return model.predict(x)

            explainer = shap.KernelExplainer(f_pred, background)
            shap_values = explainer.shap_values(X_shap)

            shap_values = np.array(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[0]

            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            tgt = m1 if shap_on == "holdout" else m2
            if feature_names is not None and len(feature_names) == len(mean_abs_shap):
                for fname, val in zip(feature_names, mean_abs_shap):
                    tgt[f"shap_{fname}"] = float(val)
            else:
                for i, val in enumerate(mean_abs_shap):
                    tgt[f"shap_f{i}"] = float(val)

        except Exception as e:
            print(f"[WARN] SHAP 计算失败：{e}")
    elif do_shap and not _HAS_SHAP:
        print("[WARN] 未安装 shap，跳过 SHAP 分析。请先 `pip install shap`。")

    return m1, pred1, m2, pred2


# ================== 可视化：预测 vs 真实散点图 ==================

def plot_pred_vs_true_scatter(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    task_label: str,
    mode_tag: str,
    split_desc: str,
    kf_num: int,
    cv_summary: Dict[str, float],
    test_metrics: Dict[str, float],
    out_path: str,
) -> None:
    """
    绘制预测值 vs 真实值散点图，并保存为 PNG 文件，不改变任何计算逻辑。
    """
    y_true_np = y_true.to_numpy(dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float).ravel()

    if y_true_np.size == 0:
        print(f"[Plot] 空测试集，跳过绘图：{out_path}")
        return

    # 线性拟合：预测值 = m * 真实值 + b
    try:
        m, b = np.polyfit(y_true_np, y_pred_np, 1)
    except Exception as e:
        print(f"[Plot] polyfit 失败：{e}，使用 m=1, b=0")
        m, b = 1.0, 0.0

    v_min = float(min(y_true_np.min(), y_pred_np.min()))
    v_max = float(max(y_true_np.max(), y_pred_np.max()))
    margin = 0.05 * (v_max - v_min) if v_max > v_min else 1.0
    x_min = v_min - margin
    x_max = v_max + margin

    x_line = np.linspace(x_min, x_max, 200)

    cv_r2_mean = float(cv_summary.get("cv_r2_mean", np.nan))
    cv_r2_std = float(cv_summary.get("cv_r2_std", np.nan))
    if np.isfinite(cv_r2_mean) and abs(cv_r2_mean) > 1e-12 and np.isfinite(cv_r2_std):
        cov_r2 = 100.0 * cv_r2_std / abs(cv_r2_mean)
    else:
        cov_r2 = np.nan

    rmse = np.sqrt(float(test_metrics.get("mse", np.nan)))
    test_r2 = float(test_metrics.get("r2", np.nan))
    mae = float(test_metrics.get("mae", np.nan))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.scatter(y_true_np, y_pred_np, alpha=0.8, label="Data Points")
    ax.plot(x_line, x_line, linestyle="--", linewidth=1.5, label="Ideal Fit (y=x)")
    ax.plot(x_line, m * x_line + b, linewidth=1.5, label=f"Trend (m={m:.2f})")

    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")

    title1 = f"Task: {task_label} ({mode_tag})"
    title2 = (
        f"Model Stability ({kf_num} – Fold CV)  "
        f"Mean R²: {cv_r2_mean:.3f} | Std: {cv_r2_std:.3f}  "
        f"CoV (R²): {cov_r2:.2f}%"
    )
    fig.suptitle(f"{title1}\n{title2}", fontsize=12)

    ax.set_title(f"Evaluation on {split_desc}")

    box_text = (
        f"Split : {split_desc}\n"
        f"Test R²: {test_r2:.3f}\n"
        f"RMSE: {rmse:.3f}\n"
        f"MAE: {mae:.3f}"
    )
    ax.text(
        0.05,
        0.95,
        box_text,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ================== 落盘 ==================

def save_cv_metrics(out_dir: str, prefix: str, mode: str, cv_metrics: Dict[str, Any], enc: str) -> str:
    path = os.path.join(out_dir, f"{prefix}_{mode}_cv_metrics.csv")
    pd.DataFrame(cv_metrics["fold_metrics"]).to_csv(path, index=False, encoding=enc)
    return path


def append_cv_summary(
    out_dir: str, prefix: str, mode: str, cv_summary: Dict[str, float],
    meta: Dict[str, Any], enc: str
) -> str:
    row = {**{"mode": mode}, **meta, **cv_summary}
    path = os.path.join(out_dir, f"{prefix}_cv_summary.csv")
    header_needed = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, index=False, mode="a", header=header_needed, encoding=enc)
    return path


def save_test_metrics(out_dir: str, prefix: str, mode: str, test_metrics: Dict[str, float], split_desc: str, enc: str) -> str:
    row = {**test_metrics, "split": split_desc}
    path = os.path.join(out_dir, f"{prefix}_{mode}_test_metrics.csv")
    pd.DataFrame([row]).to_csv(path, index=False, encoding=enc)
    return path


def save_test_preds(out_dir: str, prefix: str, mode: str, y_te: pd.Series, y_pred: np.ndarray, enc: str) -> str:
    path = os.path.join(out_dir, f"{prefix}_{mode}_test_preds.csv")
    pd.DataFrame({"y_true": y_te.to_numpy(dtype=float), "y_pred": np.asarray(y_pred, dtype=float).ravel()}).to_csv(
        path, index=False, encoding=enc
    )
    return path


def save_test_metrics_tag(
    out_dir: str, prefix: str, mode: str,
    test_metrics: Dict[str, float], tag: str, enc: str
) -> str:
    """
    带 tag 的测试指标保存，避免同一任务多次评估（holdout + external）互相覆盖。
    """
    row = {**test_metrics, "split": tag}
    safe_tag = str(tag).replace("/", "_")
    path = os.path.join(out_dir, f"{prefix}_{mode}_test_metrics_{safe_tag}.csv")
    pd.DataFrame([row]).to_csv(path, index=False, encoding=enc)
    return path


def save_test_preds_tag(
    out_dir: str, prefix: str, mode: str,
    y_true: pd.Series, y_pred: np.ndarray, tag: str, enc: str
) -> str:
    """
    带 tag 的预测保存。
    """
    safe_tag = str(tag).replace("/", "_")
    path = os.path.join(out_dir, f"{prefix}_{mode}_test_preds_{safe_tag}.csv")
    pd.DataFrame({"y_true": y_true.to_numpy(dtype=float), "y_pred": np.asarray(y_pred, dtype=float).ravel()}).to_csv(
        path, index=False, encoding=enc
    )
    return path


def save_used_features(out_dir: str, prefix: str, used_features: List[str]) -> str:
    path = os.path.join(out_dir, f"{prefix}_used_features.txt")
    with open(path, "w", encoding="utf-8") as f:
        for c in used_features:
            f.write(f"{c}\n")
    return path


# ================== 直接吃已分好 X/Y 的入口（可选） ==================

def run_tabpfn_with_xy(
    X_tr: np.ndarray, y_tr: pd.Series,
    X_te: np.ndarray, y_te: pd.Series,
    *,
    out_dir: str = "./tabpfn_cv_out",
    output_prefix: str = "manual",
    kf_num: int = 5,
    log_modes: List[bool] = [False],
    device: str = "auto",
    seed: int = 42,
    enc: str = "utf-8-sig",
    split_desc: str = "pre_split",
    meta: Optional[Dict[str, Any]] = None,
    used_features: Optional[List[str]] = None,
    do_shap: bool = True,
    shap_max_samples: int = 200,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    task_out_dir = os.path.join(out_dir, output_prefix)
    os.makedirs(task_out_dir, exist_ok=True)

    device = device if device in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")

    for log_mode in log_modes:
        mode_tag = "log" if bool(log_mode) else "raw"
        cv_pack = run_kfold_cv(X_tr, y_tr, kf_num, device, bool(log_mode), seed)
        save_cv_metrics(task_out_dir, output_prefix, mode_tag, cv_pack, enc)

        cv_summary_meta = {
            "task": output_prefix,
            "kf_num": kf_num,
            "n_train": int(len(y_tr)),
            "n_features": int(X_tr.shape[1]),
            "device": device,
            **(meta or {}),
        }
        append_cv_summary(task_out_dir, output_prefix, mode_tag, cv_pack["summary"], cv_summary_meta, enc)

        s = cv_pack["summary"]
        print(
            f"[{output_prefix} | {mode_tag.upper()}] CV R²: {s['cv_r2_mean']:.4f} ± {s['cv_r2_std']:.4f} | "
            f"MAE: {s['cv_mae_mean']:.4f} ± {s['cv_mae_std']:.4f} | "
            f"MSE: {s['cv_mse_mean']:.4f} ± {s['cv_mse_std']:.4f}"
        )

        test_metrics, y_pred_test = fit_final_and_eval_holdout(
            X_tr, y_tr, X_te, y_te, device, bool(log_mode), seed,
            feature_names=used_features,
            do_shap=do_shap,
            shap_max_samples=shap_max_samples,
        )
        save_test_metrics(task_out_dir, output_prefix, mode_tag, test_metrics, split_desc, enc)
        save_test_preds(task_out_dir, output_prefix, mode_tag, y_te, y_pred_test, enc)

        scatter_path = os.path.join(
            task_out_dir,
            f"{output_prefix}_{mode_tag}_{split_desc}_scatter.png",
        )
        plot_pred_vs_true_scatter(
            y_true=y_te,
            y_pred=y_pred_test,
            task_label=output_prefix,
            mode_tag=mode_tag,
            split_desc=split_desc,
            kf_num=kf_num,
            cv_summary=cv_pack["summary"],
            test_metrics=test_metrics,
            out_path=scatter_path,
        )

        print(
            f"[{output_prefix} | {mode_tag.upper()} | {split_desc}] "
            f"TEST R²: {test_metrics['r2']:.4f} | MAE: {test_metrics['mae']:.4f} | MSE: {test_metrics['mse']:.4f}"
        )

    if used_features:
        save_used_features(task_out_dir, output_prefix, used_features)


# ================== YAML 驱动的单任务执行 ==================

def run_single_tabpfn_task(
    cfg: Dict[str, Any], task_name: str, task_cfg: Dict[str, Any],
    sep: str, enc: str, device: str, out_dir: str, seed: int
) -> None:
    print(f"[TabPFN] Running task: {task_name}")

    target = task_cfg["target"]
    feature_cols = task_cfg.get("feature_cols")
    exclude_cols = task_cfg.get("exclude_cols")
    kf_num = int(task_cfg.get("kf_num", 5))
    log_modes = task_cfg.get("log_modes", [False])
    output_prefix = task_cfg.get("output_prefix", task_name)
    test_split = task_cfg.get("test_split", None)

    # SHAP 相关配置
    shap_cfg = task_cfg.get("shap", {})
    do_shap = bool(shap_cfg.get("enable", True))   # 默认开启
    shap_max_samples = int(shap_cfg.get("max_samples", 200))
    shap_on = str(shap_cfg.get("shap_on", "external")).lower()  # external / holdout

    # 分箱重采样配置
    bin_cfg = task_cfg.get("binning", {})
    bin_enable = bool(bin_cfg.get("enable", False))
    bin_n_bins = int(bin_cfg.get("n_bins", 10))
    bin_mode = str(bin_cfg.get("mode", "undersample"))      # "undersample" 或 "balance"
    bin_strategy = str(bin_cfg.get("strategy", "quantile")) # "quantile" 或 "uniform"
    bin_max_per_bin = bin_cfg.get("max_per_bin", None)
    if bin_max_per_bin is not None:
        bin_max_per_bin = int(bin_max_per_bin)

    _ensure_dir(out_dir)
    task_out_dir = os.path.join(out_dir, output_prefix)
    _ensure_dir(task_out_dir)

    preprocess = task_cfg.get("preprocess", {})

    def _mk_pre_steps(cols):
        steps = []
        if preprocess.get("drop_constant", False):
            steps.append(step_drop_constant(cols or []))
        if preprocess.get("fillna_median", False):
            steps.append(step_fillna_numeric(cols or [], "median"))
        if preprocess.get("dropna_target", True):
            steps.append(step_dropna_in([target]))
        return steps

    # ======== 数据来源判定：三种入口 ========
    train_csv = task_cfg.get("train_csv")
    test_csv = task_cfg.get("test_csv")
    in_csv_list = task_cfg.get("in_csv_list")
    in_csv = task_cfg.get("in_csv")

    # ------------------------------------------------------------------
    # 情况 1：双 CSV（A=source, B=external）
    # 需求：A 内部 80/20 -> CV 在 A_fit(80%) -> 最终模型预测 A_holdout(20%) + B
    # ------------------------------------------------------------------
    if train_csv and test_csv:
        pack = build_xy_from_two_csvs(
            train_csv=_get(cfg, train_csv, ctx=f"{task_name}.train_csv"),
            test_csv=_get(cfg, test_csv,  ctx=f"{task_name}.test_csv"),
            target=target,
            sep=sep, enc=enc,
            feature_cols=feature_cols,
            exclude_cols=exclude_cols,
            align=preprocess.get("align", "intersection"),
            preprocess_steps_train=_mk_pre_steps(feature_cols),
            preprocess_steps_test=_mk_pre_steps(feature_cols),
        )

        X_src, y_src = pack.X_tr, pack.y_tr      # A（source，全量）
        X_ext, y_ext = pack.X_te, pack.y_te      # B（external，全量）
        used_feats = pack.used_features

        dual_cfg = task_cfg.get("dual_dataset", {})
        inner_test_split = float(dual_cfg.get("inner_test_split", 0.2))  # 默认 0.2（即 80/20）
        if not (0.0 < inner_test_split < 1.0):
            raise ValueError(f"{task_name}: dual_dataset.inner_test_split 必须在 (0,1)，当前={inner_test_split}")

        # A 内部切：fit(80%) + holdout(20%)
        idx_fit, idx_hold = split_indices(len(y_src), inner_test_split, seed)
        X_fit, y_fit, X_hold, y_hold = split_by_indices(X_src, y_src, idx_fit, idx_hold)

        # 分箱：只对 A_fit 生效（符合你的语义）
        if bin_enable:
            print(
                f"[{task_name}] 启用分箱训练（仅 A_fit=80%）：n_bins={bin_n_bins}, mode={bin_mode}, "
                f"strategy={bin_strategy}, max_per_bin={bin_max_per_bin}"
            )
            X_fit, y_fit = apply_binning_resample_on_target(
                X_fit,
                y_fit,
                n_bins=bin_n_bins,
                mode=bin_mode,
                max_per_bin=bin_max_per_bin,
                strategy=bin_strategy,
                seed=seed,
            )

        # 逐 mode 执行
        for log_mode in log_modes:
            mode_tag = "log" if bool(log_mode) else "raw"

            # CV：只在 A_fit 上做
            cv_pack = run_kfold_cv(X_fit, y_fit, kf_num, device, bool(log_mode), seed)
            save_cv_metrics(task_out_dir, output_prefix, mode_tag, cv_pack, enc)

            cv_summary_meta = {
                "task": task_name,
                "kf_num": kf_num,
                "n_train_fit": int(len(y_fit)),
                "n_train_total_src": int(len(y_src)),
                "n_features": int(X_fit.shape[1]),
                "device": device,
                "external_test_rows": int(len(y_ext)),
                "inner_test_split": float(inner_test_split),
            }
            append_cv_summary(task_out_dir, output_prefix, mode_tag, cv_pack["summary"], cv_summary_meta, enc)

            s = cv_pack["summary"]
            print(
                f"[{task_name} | {mode_tag.upper()}] "
                f"CV(on A_fit) R²: {s['cv_r2_mean']:.4f} ± {s['cv_r2_std']:.4f} | "
                f"MAE: {s['cv_mae_mean']:.4f} ± {s['cv_mae_std']:.4f} | "
                f"MSE: {s['cv_mse_mean']:.4f} ± {s['cv_mse_std']:.4f}"
            )

            # 最终拟合一次：A_fit -> 评估 A_holdout + external(B)
            m_hold, pred_hold, m_ext, pred_ext = fit_once_and_eval_two_sets(
                X_fit, y_fit,
                X_hold, y_hold,
                X_ext, y_ext,
                device=device,
                log_mode=bool(log_mode),
                seed=seed,
                do_shap=do_shap,
                shap_max_samples=shap_max_samples,
                feature_names=used_feats,
                shap_on=("holdout" if shap_on == "holdout" else "external"),
            )

            # 保存：A_holdout
            hold_tag = f"A_holdout_{inner_test_split:.2f}"
            save_test_metrics_tag(task_out_dir, output_prefix, mode_tag, m_hold, hold_tag, enc)
            save_test_preds_tag(task_out_dir, output_prefix, mode_tag, y_hold, pred_hold, hold_tag, enc)

            hold_scatter = os.path.join(task_out_dir, f"{output_prefix}_{mode_tag}_{hold_tag}_scatter.png")
            plot_pred_vs_true_scatter(
                y_true=y_hold,
                y_pred=pred_hold,
                task_label=task_name,
                mode_tag=mode_tag,
                split_desc=hold_tag,
                kf_num=kf_num,
                cv_summary=cv_pack["summary"],
                test_metrics=m_hold,
                out_path=hold_scatter,
            )

            # 保存：external(B)
            ext_tag = "external_B"
            save_test_metrics_tag(task_out_dir, output_prefix, mode_tag, m_ext, ext_tag, enc)
            save_test_preds_tag(task_out_dir, output_prefix, mode_tag, y_ext, pred_ext, ext_tag, enc)

            ext_scatter = os.path.join(task_out_dir, f"{output_prefix}_{mode_tag}_{ext_tag}_scatter.png")
            plot_pred_vs_true_scatter(
                y_true=y_ext,
                y_pred=pred_ext,
                task_label=task_name,
                mode_tag=mode_tag,
                split_desc=ext_tag,
                kf_num=kf_num,
                cv_summary=cv_pack["summary"],
                test_metrics=m_ext,
                out_path=ext_scatter,
            )

            print(
                f"[{task_name} | {mode_tag.upper()}] "
                f"A_holdout R²={m_hold['r2']:.4f} MAE={m_hold['mae']:.4f} MSE={m_hold['mse']:.4f}  ||  "
                f"external_B R²={m_ext['r2']:.4f} MAE={m_ext['mae']:.4f} MSE={m_ext['mse']:.4f}"
            )

        save_used_features(task_out_dir, output_prefix, used_feats)
        return  # ★ 双 CSV 已完成，避免继续走后续通用分支

    # ------------------------------------------------------------------
    # 情况 2：in_csv_list —— 多个 CSV 合并后一份再随机划分
    # ------------------------------------------------------------------
    if in_csv_list:
        csv_paths = [_get(cfg, p, ctx=f"{task_name}.in_csv_list[{i}]") for i, p in enumerate(in_csv_list)]
        df_list = [load_dataframe(p, sep=sep, enc=enc) for p in csv_paths]
        df_all = pd.concat(df_list, ignore_index=True)

        _ = inspect_df(df_all, target, feature_cols, exclude_cols)

        if preprocess:
            df_all, _logs = preprocess_df(df_all, _mk_pre_steps(feature_cols))

        X_all, y_all, used_feats = make_X_y(df_all, target, feature_cols, exclude_cols)

        if isinstance(test_split, (int, float)) and 0.0 < float(test_split) < 1.0:
            idx_tr, idx_te = split_indices(len(y_all), float(test_split), seed)
            X_tr, y_tr, X_te, y_te = split_by_indices(X_all, y_all, idx_tr, idx_te)
            split_desc = f"holdout_{test_split}"
        else:
            raise ValueError(f"{task_name}: 使用 in_csv_list 时必须提供有效的 test_split (0,1)，当前为 {test_split!r}。")

    # ------------------------------------------------------------------
    # 情况 3：单个 in_csv —— 普通单 CSV 随机划分
    # ------------------------------------------------------------------
    elif in_csv:
        X_tr, y_tr, X_te, y_te, used_feats, idx_tr, idx_te, _ = build_xy_from_csv(
            in_csv=_get(cfg, in_csv, ctx=f"{task_name}.in_csv"),
            target=target,
            sep=sep, enc=enc,
            feature_cols=feature_cols,
            exclude_cols=exclude_cols,
            test_split=test_split,
            seed=seed,
            preprocess_steps=_mk_pre_steps(feature_cols),
        )
        split_desc = f"holdout_{test_split}"

    else:
        raise ValueError(
            f"{task_name}: 未指定数据来源。请至少配置以下字段之一：\n"
            f"  - train_csv + test_csv（双 CSV）\n"
            f"  - in_csv_list（多 CSV 合并后随机划分）\n"
            f"  - in_csv（单 CSV 随机划分）"
        )

    # ======== 若启用分箱重采样，则仅对训练集进行重采样 ========
    if bin_enable:
        print(
            f"[{task_name}] 启用分箱训练：n_bins={bin_n_bins}, mode={bin_mode}, "
            f"strategy={bin_strategy}, max_per_bin={bin_max_per_bin}"
        )
        X_tr, y_tr = apply_binning_resample_on_target(
            X_tr,
            y_tr,
            n_bins=bin_n_bins,
            mode=bin_mode,
            max_per_bin=bin_max_per_bin,
            strategy=bin_strategy,
            seed=seed,
        )

    # ======== TabPFN 训练 + CV + 测试评估（含 SHAP） ========
    for log_mode in log_modes:
        mode_tag = "log" if bool(log_mode) else "raw"
        cv_pack = run_kfold_cv(X_tr, y_tr, kf_num, device, bool(log_mode), seed)
        save_cv_metrics(task_out_dir, output_prefix, mode_tag, cv_pack, enc)

        cv_summary_meta = {
            "task": task_name,
            "kf_num": kf_num,
            "n_train": int(len(y_tr)),
            "n_features": int(X_tr.shape[1]),
            "device": device,
        }
        append_cv_summary(task_out_dir, output_prefix, mode_tag, cv_pack["summary"], cv_summary_meta, enc)

        s = cv_pack["summary"]
        print(
            f"[{task_name} | {mode_tag.upper()}] "
            f"CV R²: {s['cv_r2_mean']:.4f} ± {s['cv_r2_std']:.4f} | "
            f"MAE: {s['cv_mae_mean']:.4f} ± {s['cv_mae_std']:.4f} | "
            f"MSE: {s['cv_mse_mean']:.4f} ± {s['cv_mse_std']:.4f}"
        )

        test_metrics, y_pred_test = fit_final_and_eval_holdout(
            X_tr, y_tr, X_te, y_te, device, bool(log_mode), seed,
            feature_names=used_feats,
            do_shap=do_shap,
            shap_max_samples=shap_max_samples,
        )
        save_test_metrics(task_out_dir, output_prefix, mode_tag, test_metrics, split_desc, enc)
        save_test_preds(task_out_dir, output_prefix, mode_tag, y_te, y_pred_test, enc)

        scatter_path = os.path.join(
            task_out_dir,
            f"{output_prefix}_{mode_tag}_{split_desc}_scatter.png",
        )
        plot_pred_vs_true_scatter(
            y_true=y_te,
            y_pred=y_pred_test,
            task_label=task_name,
            mode_tag=mode_tag,
            split_desc=split_desc,
            kf_num=kf_num,
            cv_summary=cv_pack["summary"],
            test_metrics=test_metrics,
            out_path=scatter_path,
        )

        print(
            f"[{task_name} | {mode_tag.upper()} | {split_desc}] "
            f"TEST R²: {test_metrics['r2']:.4f} | "
            f"TEST MAE: {test_metrics['mae']:.4f} | "
            f"TEST MSE: {test_metrics['mse']:.4f}"
        )

    save_used_features(task_out_dir, output_prefix, used_feats)


# ================== 入口调度 ==================

def run_tabpfn_tasks_from_cfg(cfg: Dict[str, Any], task_keys: Optional[List[str]] = None) -> None:
    """
    对 YAML 中 tabpfn_tasks 的任务逐个执行：训练集内 K 折 CV + 测试评估（单 CSV / 多 CSV 合并 / 双 CSV 外部测试）。
    """
    tasks_all = cfg.get("tabpfn_tasks", {})
    if not tasks_all:
        print("[TabPFN] 未找到 tabpfn_tasks 配置，跳过。")
        return

    sep, enc = _io_params(cfg)
    device = _pick_device(cfg)
    seed = int(cfg.get("params", {}).get("seed", 42))
    out_dir = cfg.get("paths", {}).get("tabpfn_out_dir", "./tabpfn_cv_out")

    if task_keys is None:
        task_keys = list(tasks_all.keys())

    for tkey in task_keys:
        task_cfg = tasks_all.get(tkey)
        if task_cfg is None:
            print(f"[TabPFN] 跳过任务 {tkey}：未配置")
            continue
        run_single_tabpfn_task(cfg, tkey, task_cfg, sep, enc, device, out_dir, seed)