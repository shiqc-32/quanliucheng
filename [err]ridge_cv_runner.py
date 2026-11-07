# -*- coding: utf-8 -*-
# ridge_cv_runner.py
"""
基于 Ridge 回归的流程：
- 支持单 CSV / 双 CSV / 多 CSV 合并
- 训练集内 K 折交叉验证
- 留出测试集评估
- 支持 log_mode（log(y + LOG_SHIFT)，评估时还原）
- 用 BayesSearchCV 优化 alpha 等超参数
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

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

try:
    from skopt import BayesSearchCV
    from skopt.space import Real
    _HAVE_SKOPT = True
except ImportError:
    _HAVE_SKOPT = False

LOG_SHIFT = 1.0


# ================== 工具与通用 ==================


def _get(cfg: Dict[str, Any], dotted: Any, default=None, *, ctx: str = ""):
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
            raise ValueError(f"_get 解析引用字符串失败：{dotted}（应为 @root.key）")
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


# ================== 贝叶斯调参 ==================


def bayes_optimize_ridge(
    X_tr: np.ndarray,
    y_tr: pd.Series,
    log_mode: bool,
    seed: int,
    bayes_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    BayesSearchCV + Pipeline(StandardScaler + Ridge) 优化 alpha。
    返回可直接传 Ridge(**params) 的 dict。
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
                f"[Ridge|Bayes] log_mode=True 时，训练集 y 的最小值为 {y_min:.4f}，"
                f"需要满足 y > {-LOG_SHIFT} 才能使用 log(y + {LOG_SHIFT})。"
            )

    n_iter = int(bayes_cfg.get("n_iter", 25))
    cv_inner = int(bayes_cfg.get("cv", 3))
    n_jobs = int(bayes_cfg.get("n_jobs", -1))

    search_spaces = {
        "ridge__alpha": Real(1e-3, 1e3, prior="log-uniform"),
    }

    y_arr = y_tr.to_numpy(dtype=float)
    if log_mode:
        y_arr = np.log(y_arr + LOG_SHIFT)

    inner_cv = KFold(n_splits=cv_inner, shuffle=True, random_state=seed)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(random_state=seed)),
    ])

    opt = BayesSearchCV(
        estimator=pipe,
        search_spaces=search_spaces,
        n_iter=n_iter,
        cv=inner_cv,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        random_state=seed,
        verbose=0,
        refit=True,
    )

    print(f"[Ridge|Bayes] mode={'log' if log_mode else 'raw'} | n_iter={n_iter}, cv={cv_inner}")
    opt.fit(X_tr, y_arr)
    print(f"[Ridge|Bayes] best_score={opt.best_score_:.6f}")
    print(f"[Ridge|Bayes] best_params={opt.best_params_}")

    best = opt.best_params_
    ridge_params: Dict[str, Any] = {}
    for k, v in best.items():
        if k.startswith("ridge__"):
            ridge_params[k.split("__", 1)[1]] = v
    return ridge_params


# ================== 训练评估 ==================


def run_kfold_cv(
    X_tr: np.ndarray,
    y_tr: pd.Series,
    kf_num: int,
    log_mode: bool,
    seed: int,
    ridge_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
        scaler = StandardScaler().fit(X_tr[tr_idx])
        Xtr = scaler.transform(X_tr[tr_idx])
        Xval = scaler.transform(X_tr[val_idx])

        if log_mode:
            y_tr_fold = np.log(y_tr_np[tr_idx] + LOG_SHIFT)
            y_val_fold = np.log(y_tr_np[val_idx] + LOG_SHIFT)
        else:
            y_tr_fold = y_tr_np[tr_idx]
            y_val_fold = y_tr_np[val_idx]

        model = Ridge(random_state=seed, **(ridge_params or {}))
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
    ridge_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
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

    model = Ridge(random_state=seed, **(ridge_params or {}))
    model.fit(Xtr_full, ytr)

    y_pred = model.predict(Xte_full)
    if log_mode:
        y_pred = np.exp(y_pred) - LOG_SHIFT

    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    return {"r2": r2, "mae": mae, "mse": mse, "n_test": int(len(y_te))}, y_pred


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


# ================== 直接吃已分好 X/Y 的入口 ==================


def run_ridge_with_xy(
    X_tr: np.ndarray,
    y_tr: pd.Series,
    X_te: np.ndarray,
    y_te: pd.Series,
    *,
    out_dir: str = "./ridge_cv_out",
    output_prefix: str = "manual_ridge",
    kf_num: int = 5,
    log_modes: List[bool] = (False,),
    seed: int = 42,
    enc: str = "utf-8-sig",
    split_desc: str = "pre_split",
    meta: Optional[Dict[str, Any]] = None,
    used_features: Optional[List[str]] = None,
    bayes_opt: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    task_out_dir = os.path.join(out_dir, output_prefix)
    os.makedirs(task_out_dir, exist_ok=True)

    bayes_cfg = bayes_opt or {}
    bayes_enabled = bool(bayes_cfg.get("enabled", True))

    for log_mode in log_modes:
        mode_tag = "log" if bool(log_mode) else "raw"

        ridge_params: Dict[str, Any] = {}
        if bayes_enabled:
            ridge_params = bayes_optimize_ridge(X_tr, y_tr, bool(log_mode), seed, bayes_cfg)

        cv_pack = run_kfold_cv(X_tr, y_tr, kf_num, bool(log_mode), seed, ridge_params)
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

        test_metrics, y_pred_test = fit_final_and_eval_holdout(
            X_tr, y_tr, X_te, y_te, bool(log_mode), seed, ridge_params
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


def run_single_ridge_task(
    cfg: Dict[str, Any],
    task_name: str,
    task_cfg: Dict[str, Any],
    sep: str,
    enc: str,
    out_dir: str,
    seed: int,
) -> None:
    print(f"[Ridge] Running task: {task_name}")

    target = task_cfg["target"]
    feature_cols = task_cfg.get("feature_cols")
    exclude_cols = task_cfg.get("exclude_cols")
    kf_num = int(task_cfg.get("kf_num", 5))
    log_modes = task_cfg.get("log_modes", [False])
    output_prefix = task_cfg.get("output_prefix", task_name)
    test_split = task_cfg.get("test_split", None)

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

    bayes_cfg = task_cfg.get("bayes_opt", {})
    bayes_enabled = bool(bayes_cfg.get("enabled", True))

    train_csv = task_cfg.get("train_csv")
    test_csv = task_cfg.get("test_csv")
    in_csv_list = task_cfg.get("in_csv_list")
    in_csv = task_cfg.get("in_csv")

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
            f"  - train_csv + test_csv\n"
            f"  - in_csv_list\n"
            f"  - in_csv"
        )

    for log_mode in log_modes:
        mode_tag = "log" if bool(log_mode) else "raw"

        ridge_params: Dict[str, Any] = {}
        if bayes_enabled:
            ridge_params = bayes_optimize_ridge(X_tr, y_tr, bool(log_mode), seed, bayes_cfg)

        cv_pack = run_kfold_cv(X_tr, y_tr, kf_num, bool(log_mode), seed, ridge_params)
        save_cv_metrics(task_out_dir, output_prefix, mode_tag, cv_pack, enc)

        cv_summary_meta = {
            "task": task_name,
            "kf_num": kf_num,
            "n_train": int(len(y_tr)),
            "n_features": int(X_tr.shape[1]),
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

        test_metrics, y_pred_test = fit_final_and_eval_holdout(
            X_tr, y_tr, X_te, y_te, bool(log_mode), seed, ridge_params
        )
        save_test_metrics(task_out_dir, output_prefix, mode_tag, test_metrics, split_desc, enc)
        save_test_preds(task_out_dir, output_prefix, mode_tag, y_te, y_pred_test, enc)

        print(
            f"[{task_name} | {mode_tag.upper()} | {split_desc}] "
            f"TEST R²: {test_metrics['r2']:.4f} | "
            f"TEST MAE: {test_metrics['mae']:.4f} | "
            f"TEST MSE: {test_metrics['mse']:.4f}"
        )

    save_used_features(task_out_dir, output_prefix, used_feats)


# ================== 入口调度 ==================


def run_ridge_tasks_from_cfg(cfg: Dict[str, Any], task_keys: Optional[List[str]] = None) -> None:
    """
    对 YAML 中 ridge_tasks 的任务逐个执行。
    """
    tasks_all = cfg.get("ridge_tasks", {})
    if not tasks_all:
        print("[Ridge] 未找到 ridge_tasks 配置，跳过。")
        return

    sep, enc = _io_params(cfg)
    seed = int(cfg.get("params", {}).get("seed", 42))
    out_dir = cfg.get("paths", {}).get("ridge_out_dir", "./ridge_cv_out")

    if task_keys is None:
        task_keys = list(tasks_all.keys())

    for tkey in task_keys:
        task_cfg = tasks_all.get(tkey)
        if task_cfg is None:
            print(f"[Ridge] 跳过任务 {tkey}：未配置")
            continue
        run_single_ridge_task(cfg, tkey, task_cfg, sep, enc, out_dir, seed)
