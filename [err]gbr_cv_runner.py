# -*- coding: utf-8 -*-
# gbr_cv_runner.py
"""
基于 GradientBoostingRegressor 的回归流程：
- 支持单 CSV / 双 CSV / 多 CSV 合并（in_csv / train_csv+test_csv / in_csv_list）
- 训练集内 K 折交叉验证
- 留出测试集评估
- 支持 log_mode（内部 log(y + LOG_SHIFT)，评估时还原）
- 支持分箱重采样（binning）以平衡目标分布
- 支持贝叶斯优化超参数（skopt.BayesSearchCV）
- 支持 SHAP 特征重要性分析（每个任务/模式只在最终模型上做一次）
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

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

# 贝叶斯优化
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    _HAVE_SKOPT = True
except ImportError:
    BayesSearchCV = None  # type: ignore
    Real = Integer = None  # type: ignore
    _HAVE_SKOPT = False

# SHAP
try:
    import shap  # type: ignore
    _HAVE_SHAP = True
except ImportError:
    shap = None  # type: ignore
    _HAVE_SHAP = False

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


# ================== GBR 模型与分箱 ==================


def _build_gbr(task_cfg: Dict[str, Any], seed: int, gbr_params: Optional[Dict[str, Any]] = None) -> GradientBoostingRegressor:
    """
    根据 task_cfg['gbr_params'] 与可选的 gbr_params（BayesSearch 最优结果）构造 GradientBoostingRegressor。
    """
    base = task_cfg.get("gbr_params", {}) or {}
    params = dict(base)
    if gbr_params:
        params.update(gbr_params)

    # sklearn 的 GradientBoostingRegressor 不提供 random_state 参数名以外太多花样
    return GradientBoostingRegressor(
        loss=params.get("loss", "squared_error"),
        learning_rate=float(params.get("learning_rate", 0.05)),
        n_estimators=int(params.get("n_estimators", 800)),
        subsample=float(params.get("subsample", 1.0)),
        criterion=params.get("criterion", "friedman_mse"),
        min_samples_split=int(params.get("min_samples_split", 2)),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        min_weight_fraction_leaf=float(params.get("min_weight_fraction_leaf", 0.0)),
        max_depth=int(params.get("max_depth", 3)),
        min_impurity_decrease=float(params.get("min_impurity_decrease", 0.0)),
        max_features=params.get("max_features", None),
        random_state=seed,
    )


def apply_binning_resample(
    X: np.ndarray,
    y: pd.Series,
    bin_cfg: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, pd.Series]:
    """
    对 (X, y) 进行目标分箱 + 重采样。
    参见 rf_cv_runner.apply_binning_resample 的说明。
    """
    if not bin_cfg or not bool(bin_cfg.get("enable", False)):
        return X, y

    rng = np.random.default_rng(seed)
    n_bins = int(bin_cfg.get("n_bins", 8))
    mode = str(bin_cfg.get("mode", "undersample")).lower()
    strategy = str(bin_cfg.get("strategy", "uniform")).lower()
    max_per_bin = int(bin_cfg.get("max_per_bin", 150))

    y_arr = y.to_numpy(dtype=float)
    y_min, y_max = float(np.min(y_arr)), float(np.max(y_arr))
    if y_min == y_max:
        print("[Binning] y 全部相同，分箱无意义，直接跳过。")
        return X, y

    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(y_arr, qs)
    else:
        edges = np.linspace(y_min, y_max, n_bins + 1)

    edges = np.unique(edges)
    if len(edges) <= 2:
        print("[Binning] 分箱边界过少，跳过分箱。")
        return X, y

    n_bins_eff = len(edges) - 1
    bin_ids = np.digitize(y_arr, edges[1:-1], right=False)
    bin_to_indices = {b: np.where(bin_ids == b)[0] for b in range(n_bins_eff)}
    orig_counts = {int(b): int(len(idx)) for b, idx in bin_to_indices.items()}

    all_new_idx: List[int] = []

    if mode == "balance":
        max_count_all = max(len(v) for v in bin_to_indices.values())
        target = min(max_count_all, max_per_bin)
        for b, idx in bin_to_indices.items():
            if len(idx) == 0:
                continue
            if len(idx) >= target:
                chosen = rng.choice(idx, size=target, replace=False)
            else:
                chosen = rng.choice(idx, size=target, replace=True)
            all_new_idx.extend(chosen.tolist())
    else:
        for b, idx in bin_to_indices.items():
            if len(idx) == 0:
                continue
            if len(idx) > max_per_bin:
                chosen = rng.choice(idx, size=max_per_bin, replace=False)
            else:
                chosen = idx
            all_new_idx.extend(list(chosen))

    all_new_idx = np.array(all_new_idx, dtype=int)
    all_new_idx.sort()

    new_counts = {int(b): int(np.sum(bin_ids[all_new_idx] == b)) for b in range(n_bins_eff)}
    print(f"[Binning] 原始各 bin 样本数：{orig_counts}")
    print(f"[Binning] 重采样后各 bin 样本数：{new_counts}，总样本数由 {len(y_arr)} → {len(all_new_idx)}")

    X_new = X[all_new_idx]
    y_new = y.iloc[all_new_idx].reset_index(drop=True)
    return X_new, y_new


# ================== 贝叶斯调参（可选） ==================


def bayes_optimize_gbr(
    X_tr: np.ndarray,
    y_tr: pd.Series,
    log_mode: bool,
    seed: int,
    bayes_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    用 BayesSearchCV 优化 GradientBoostingRegressor 的部分超参数。
    """
    if not _HAVE_SKOPT:
        raise ImportError(
            "BayesSearchCV 不可用，请先安装 scikit-optimize：\n"
            "  pip install scikit-optimize"
        )

    if log_mode:
        y_min = float(y_tr.min())
        if y_min <= -LOG_SHIFT:
            raise ValueError(
                f"[GBR|Bayes] log_mode=True 时，训练集 y 的最小值为 {y_min:.4f}，"
                f"需要满足 y > {-LOG_SHIFT} 才能使用 log(y + {LOG_SHIFT})。"
            )

    n_iter = int(bayes_cfg.get("n_iter", 25))
    cv_inner = int(bayes_cfg.get("cv", 3))
    n_jobs = int(bayes_cfg.get("n_jobs", -1))

    # 这里只对少数关键超参数做搜索，避免搜索空间过大
    search_spaces = {
        "learning_rate": Real(1e-3, 0.5, prior="log-uniform"),
        "n_estimators": Integer(100, 2000),
        "max_depth": Integer(2, 8),
        "subsample": Real(0.5, 1.0, prior="uniform"),
    }

    y_arr = y_tr.to_numpy(dtype=float)
    if log_mode:
        y_arr = np.log(y_arr + LOG_SHIFT)

    inner_cv = KFold(n_splits=cv_inner, shuffle=True, random_state=seed)

    base_model = GradientBoostingRegressor(random_state=seed)

    opt = BayesSearchCV(
        estimator=base_model,
        search_spaces=search_spaces,
        n_iter=n_iter,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        random_state=seed,
        verbose=0,
        refit=True,
    )

    print(f"[GBR|Bayes] mode={'log' if log_mode else 'raw'} | n_iter={n_iter}, cv={cv_inner}")
    opt.fit(X_tr, y_arr)
    print(f"[GBR|Bayes] best_score={opt.best_score_:.6f}")
    print(f"[GBR|Bayes] best_params={opt.best_params_}")

    return dict(opt.best_params_)


# ================== 训练评估 ==================


def run_kfold_cv(
    X_tr: np.ndarray,
    y_tr: pd.Series,
    kf_num: int,
    log_mode: bool,
    seed: int,
    task_cfg: Dict[str, Any],
    gbr_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    训练集上 K 折交叉验证；不做标准化。
    """
    if log_mode:
        y_min = float(y_tr.min())
        if y_min <= -LOG_SHIFT:
            raise ValueError(
                f"log_mode=True 时，训练集 y 的最小值为 {y_min:.4f}，"
                f"需要满足 y > {-LOG_SHIFT} 才能使用 log(y + {LOG_SHIFT})。"
            )

    kf = KFold(n_splits=int(kf_num), shuffle=True, random_state=seed)
    fold_rows: List[Dict[str, float]] = []

    y_tr_np = y_tr.to_numpy(dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
        Xtr, Xval = X_tr[tr_idx], X_tr[val_idx]

        if log_mode:
            y_tr_fold = np.log(y_tr_np[tr_idx] + LOG_SHIFT)
            y_val_fold = np.log(y_tr_np[val_idx] + LOG_SHIFT)
        else:
            y_tr_fold = y_tr_np[tr_idx]
            y_val_fold = y_tr_np[val_idx]

        model = _build_gbr(task_cfg, seed, gbr_params)
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
        "cv_r2_mean": metrics_df["r2"].mean(),
        "cv_r2_std": metrics_df["r2"].std(ddof=1),
        "cv_mae_mean": metrics_df["mae"].mean(),
        "cv_mae_std": metrics_df["mae"].std(ddof=1),
        "cv_mse_mean": metrics_df["mse"].mean(),
        "cv_mse_std": metrics_df["mse"].std(ddof=1),
    }
    return {"fold_metrics": fold_rows, "summary": summary}


def fit_final_and_eval_holdout(
    X_tr: np.ndarray,
    y_tr: pd.Series,
    X_te: np.ndarray,
    y_te: pd.Series,
    log_mode: bool,
    seed: int,
    task_cfg: Dict[str, Any],
    gbr_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], np.ndarray, GradientBoostingRegressor]:
    """
    用全训练集拟合 GBR（不做标准化），并在测试集上评估。
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

    y_tr_np = y_tr.to_numpy(dtype=float)
    if log_mode:
        ytr = np.log(y_tr_np + LOG_SHIFT)
    else:
        ytr = y_tr_np

    model = _build_gbr(task_cfg, seed, gbr_params)
    model.fit(X_tr, ytr)

    y_pred = model.predict(X_te)
    if log_mode:
        y_pred = np.exp(y_pred) - LOG_SHIFT

    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    return {"r2": r2, "mae": mae, "mse": mse, "n_test": int(len(y_te))}, y_pred, model


# ================== 落盘 ==================


def save_cv_metrics(out_dir: str, prefix: str, mode: str, cv_metrics: Dict[str, Any], enc: str) -> str:
    path = os.path.join(out_dir, f"{prefix}_{mode}_cv_metrics.csv")
    pd.DataFrame(cv_metrics["fold_metrics"]).to_csv(path, index=False, encoding=enc)
    return path


def append_cv_summary(
    out_dir: str,
    prefix: str,
    mode: str,
    cv_summary: Dict[str, float],
    meta: Dict[str, Any],
    enc: str,
) -> str:
    row = {**{"mode": mode}, **meta, **cv_summary}
    path = os.path.join(out_dir, f"{prefix}_cv_summary.csv")
    header_needed = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, index=False, mode="a", header=header_needed, encoding=enc)
    return path


def save_test_metrics(
    out_dir: str,
    prefix: str,
    mode: str,
    test_metrics: Dict[str, float],
    split_desc: str,
    enc: str,
) -> str:
    row = {**test_metrics, "split": split_desc}
    path = os.path.join(out_dir, f"{prefix}_{mode}_test_metrics.csv")
    pd.DataFrame([row]).to_csv(path, index=False, encoding=enc)
    return path


def save_test_preds(
    out_dir: str,
    prefix: str,
    mode: str,
    y_te: pd.Series,
    y_pred: np.ndarray,
    enc: str,
) -> str:
    path = os.path.join(out_dir, f"{prefix}_{mode}_test_preds.csv")
    pd.DataFrame(
        {"y_true": y_te.to_numpy(dtype=float), "y_pred": y_pred.astype(float)}
    ).to_csv(path, index=False, encoding=enc)
    return path


def save_used_features(out_dir: str, prefix: str, used_features: List[str]) -> str:
    path = os.path.join(out_dir, f"{prefix}_used_features.txt")
    with open(path, "w", encoding="utf-8") as f:
        for c in used_features:
            f.write(f"{c}\n")
    return path


# ================== 直接吃已分好 X/Y 的入口（可选） ==================


def run_gbr_with_xy(
    X_tr: np.ndarray,
    y_tr: pd.Series,
    X_te: np.ndarray,
    y_te: pd.Series,
    *,
    out_dir: str = "./gbr_cv_out",
    output_prefix: str = "manual_gbr",
    kf_num: int = 5,
    log_modes: List[bool] = (False,),
    seed: int = 42,
    enc: str = "utf-8-sig",
    split_desc: str = "pre_split",
    meta: Optional[Dict[str, Any]] = None,
    used_features: Optional[List[str]] = None,
    bayes_opt: Optional[Dict[str, Any]] = None,
    task_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """
    直接给定 X/y 的简化入口；主要用于调试。
    不做 SHAP。
    """
    os.makedirs(out_dir, exist_ok=True)
    task_out_dir = os.path.join(out_dir, output_prefix)
    os.makedirs(task_out_dir, exist_ok=True)

    bayes_cfg = bayes_opt or {}
    bayes_enabled = bool(bayes_cfg.get("enabled", True))

    task_cfg = task_cfg or {}

    for log_mode in log_modes:
        mode_tag = "log" if bool(log_mode) else "raw"

        gbr_params: Dict[str, Any] = {}
        if bayes_enabled:
            gbr_params = bayes_optimize_gbr(X_tr, y_tr, bool(log_mode), seed, bayes_cfg)

        cv_pack = run_kfold_cv(X_tr, y_tr, kf_num, bool(log_mode), seed, task_cfg, gbr_params)
        save_cv_metrics(task_out_dir, output_prefix, mode_tag, cv_pack, enc)

        cv_summary_meta = {
            "task": output_prefix,
            "kf_num": kf_num,
            "n_train": int(len(y_tr)),
            "n_features": int(X_tr.shape[1]),
            "device": "cpu",
            **(meta or {}),
        }
        append_cv_summary(task_out_dir, output_prefix, mode_tag, cv_pack["summary"], cv_summary_meta, enc)

        s = cv_pack["summary"]
        print(
            f"[{output_prefix} | {mode_tag.upper()}] CV R²: {s['cv_r2_mean']:.4f} ± {s['cv_r2_std']:.4f} | "
            f"MAE: {s['cv_mae_mean']:.4f} ± {s['cv_mae_std']:.4f} | "
            f"MSE: {s['cv_mse_mean']:.4f} ± {s['cv_mse_std']:.4f}"
        )

        test_metrics, y_pred_test, _model = fit_final_and_eval_holdout(
            X_tr, y_tr, X_te, y_te, bool(log_mode), seed, task_cfg, gbr_params
        )
        save_test_metrics(task_out_dir, output_prefix, mode_tag, test_metrics, split_desc, enc)
        save_test_preds(task_out_dir, output_prefix, mode_tag, y_te, y_pred_test, enc)

        print(
            f"[{output_prefix} | {mode_tag.upper()} | {split_desc}] "
            f"TEST R²: {test_metrics['r2']:.4f} | MAE: {test_metrics['mae']:.4f} | MSE: {test_metrics['mse']:.4f}"
        )

    if used_features:
        save_used_features(task_out_dir, output_prefix, used_features)


# ================== YAML 驱动的单任务执行 ==================


def run_single_gbr_task(
    cfg: Dict[str, Any],
    task_name: str,
    task_cfg: Dict[str, Any],
    sep: str,
    enc: str,
    out_dir: str,
    seed: int,
) -> None:
    print(f"[GBR] Running task: {task_name}")

    target = task_cfg["target"]
    feature_cols = task_cfg.get("feature_cols")
    exclude_cols = task_cfg.get("exclude_cols")
    kf_num = int(task_cfg.get("kf_num", 5))
    log_modes = task_cfg.get("log_modes", [False])
    output_prefix = task_cfg.get("output_prefix", task_name)
    test_split = task_cfg.get("test_split", None)

    # 根目录 / 任务子目录
    _ensure_dir(out_dir)
    task_out_dir = os.path.join(out_dir, output_prefix)
    _ensure_dir(task_out_dir)

    # 预处理步骤开关
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

    bayes_cfg = task_cfg.get("bayes_opt", {})
    bayes_enabled = bool(bayes_cfg.get("enabled", True))

    # 分箱与 SHAP 配置
    bin_cfg = task_cfg.get("binning", {}) or {}
    shap_cfg = task_cfg.get("shap", {}) or {}
    do_shap = bool(shap_cfg.get("enable", False)) and _HAVE_SHAP
    max_shap_samples = int(shap_cfg.get("max_samples", 200))

    # ======== 数据来源判定：三种入口 ========
    train_csv = task_cfg.get("train_csv")
    test_csv = task_cfg.get("test_csv")
    in_csv_list = task_cfg.get("in_csv_list")
    in_csv = task_cfg.get("in_csv")

    # 情况 1：显式指定 train_csv + test_csv（双 CSV）
    if train_csv and test_csv:
        pack = build_xy_from_two_csvs(
            train_csv=_get(cfg, train_csv, ctx=f"{task_name}.train_csv"),
            test_csv=_get(cfg, test_csv, ctx=f"{task_name}.test_csv"),
            target=target,
            sep=sep,
            enc=enc,
            feature_cols=feature_cols,
            exclude_cols=exclude_cols,
            align=preprocess.get("align", "intersection"),
            preprocess_steps_train=_mk_pre_steps(feature_cols),
            preprocess_steps_test=_mk_pre_steps(feature_cols),
        )
        X_tr, y_tr, X_te, y_te = pack.X_tr, pack.y_tr, pack.X_te, pack.y_te
        used_feats = pack.used_features
        split_desc = "explicit_train_test"

    # 情况 2：in_csv_list —— 多个 CSV 合并后一份再随机划分
    elif in_csv_list:
        csv_paths = [
            _get(cfg, p, ctx=f"{task_name}.in_csv_list[{i}]")
            for i, p in enumerate(in_csv_list)
        ]
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
            raise ValueError(
                f"{task_name}: 使用 in_csv_list 时必须提供有效的 test_split (0,1)，当前为 {test_split!r}。"
            )

    # 情况 3：单个 in_csv —— 普通单 CSV 随机划分
    elif in_csv:
        X_tr, y_tr, X_te, y_te, used_feats, idx_tr, idx_te, _ = build_xy_from_csv(
            in_csv=_get(cfg, in_csv, ctx=f"{task_name}.in_csv"),
            target=target,
            sep=sep,
            enc=enc,
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
            f"  - train_csv + test_csv（双 CSV 显式划分）\n"
            f"  - in_csv_list（多 CSV 合并后随机划分）\n"
            f"  - in_csv（单 CSV 随机划分）"
        )

    # 先对训练集做（可选）分箱重采样
    X_tr_used, y_tr_used = apply_binning_resample(X_tr, y_tr, bin_cfg, seed)

    # ======== GBR + CV + 测试评估 + SHAP ========
    for log_mode in log_modes:
        mode_tag = "log" if bool(log_mode) else "raw"

        gbr_params: Dict[str, Any] = {}
        if bayes_enabled:
            gbr_params = bayes_optimize_gbr(X_tr_used, y_tr_used, bool(log_mode), seed, bayes_cfg)

        cv_pack = run_kfold_cv(X_tr_used, y_tr_used, kf_num, bool(log_mode), seed, task_cfg, gbr_params)
        save_cv_metrics(task_out_dir, output_prefix, mode_tag, cv_pack, enc)

        cv_summary_meta = {
            "task": task_name,
            "kf_num": kf_num,
            "n_train": int(len(y_tr_used)),
            "n_features": int(X_tr_used.shape[1]),
            "device": "cpu",
        }
        append_cv_summary(task_out_dir, output_prefix, mode_tag, cv_pack["summary"], cv_summary_meta, enc)

        s = cv_pack["summary"]
        print(
            f"[{task_name} | {mode_tag.upper()}] "
            f"CV R²: {s['cv_r2_mean']:.4f} ± {s['cv_r2_std']:.4f} | "
            f"MAE: {s['cv_mae_mean']:.4f} ± {s['cv_mae_std']:.4f} | "
            f"MSE: {s['cv_mse_mean']:.4f} ± {s['cv_mse_std']:.4f}"
        )

        test_metrics, y_pred_test, model_final = fit_final_and_eval_holdout(
            X_tr_used, y_tr_used, X_te, y_te, bool(log_mode), seed, task_cfg, gbr_params
        )

        shap_summary: Dict[str, float] = {}
        if do_shap:
            if not _HAVE_SHAP:
                print(f"[{task_name} | {mode_tag.upper()}] shap 未安装，跳过 SHAP 分析。")
            else:
                try:
                    n_te = X_te.shape[0]
                    if n_te <= max_shap_samples:
                        idx_shap = np.arange(n_te)
                    else:
                        rng = np.random.default_rng(seed)
                        idx_shap = rng.choice(n_te, size=max_shap_samples, replace=False)
                    X_shap = X_te[idx_shap]

                    explainer = shap.TreeExplainer(model_final)
                    shap_values = explainer.shap_values(X_shap)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    shap_values = np.array(shap_values, dtype=float)

                    mean_abs = np.mean(np.abs(shap_values), axis=0)
                    shap_summary = {
                        f"shap_abs_mean_{feat}": float(mean_abs[i])
                        for i, feat in enumerate(used_feats)
                    }
                    shap_summary["shap_n_samples"] = int(X_shap.shape[0])

                    shap_path = os.path.join(task_out_dir, f"{output_prefix}_{mode_tag}_shap_values.csv")
                    pd.DataFrame(shap_values, columns=used_feats).to_csv(shap_path, index=False, encoding=enc)

                    shap_summary_path = os.path.join(task_out_dir, f"{output_prefix}_{mode_tag}_shap_summary.csv")
                    pd.DataFrame([shap_summary]).to_csv(shap_summary_path, index=False, encoding=enc)

                    print(f"[{task_name} | {mode_tag.upper()}] 已保存 SHAP 结果：{shap_path}")
                except Exception as e:  # noqa: BLE001
                    print(f"[{task_name} | {mode_tag.upper()}] 计算 SHAP 时出错：{e}")

        test_metrics_out = {**test_metrics, **shap_summary}
        save_test_metrics(task_out_dir, output_prefix, mode_tag, test_metrics_out, split_desc, enc)
        save_test_preds(task_out_dir, output_prefix, mode_tag, y_te, y_pred_test, enc)

        print(
            f"[{task_name} | {mode_tag.upper()} | {split_desc}] "
            f"TEST R²: {test_metrics['r2']:.4f} | "
            f"TEST MAE: {test_metrics['mae']:.4f} | "
            f"TEST MSE: {test_metrics['mse']:.4f}"
        )

    save_used_features(task_out_dir, output_prefix, used_feats)


# ================== 入口调度 ==================


def run_gbr_tasks_from_cfg(cfg: Dict[str, Any], task_keys: Optional[List[str]] = None) -> None:
    """
    对 YAML 中 gbr_tasks 的任务逐个执行。
    """
    tasks_all = cfg.get("gbr_tasks", {})
    if not tasks_all:
        print("[GBR] 未找到 gbr_tasks 配置，跳过。")
        return

    sep, enc = _io_params(cfg)
    seed = int(cfg.get("params", {}).get("seed", 42))
    out_dir = cfg.get("paths", {}).get("gbr_out_dir", "./gbr_cv_out")

    if task_keys is None:
        task_keys = list(tasks_all.keys())

    for tkey in task_keys:
        task_cfg = tasks_all.get(tkey)
        if task_cfg is None:
            print(f"[GBR] 跳过任务 {tkey}：未配置")
            continue
        run_single_gbr_task(cfg, tkey, task_cfg, sep, enc, out_dir, seed)
