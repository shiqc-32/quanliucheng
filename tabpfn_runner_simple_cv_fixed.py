# -*- coding: utf-8 -*-
# tabpfn_runner_simple.py
#
# 功能说明（单 CSV 简化版）：
# 1) 仅支持单 CSV 输入
# 2) 内部随机划分训练/测试集
# 3) 在训练集上做 K 折交叉验证
# 4) 最终模型在测试集上评估
# 5) 可选目标分箱重采样（只作用于训练集）
# 6) 可选 SHAP 解释，并支持服务器上设置并行核数
# 7) 可选外部 CSV 预测：例如用训练数据训练模型，预测 BJ_model_input.csv，
#    并将 z0_TabPFN / d_TabPFN 写回指定 CSV
#
# 说明：
# - SHAP 推荐 backend="threading"
# - 若 backend="loky"，可能因模型无法跨进程序列化而失败
# - log_mode=True 时，训练使用 log(y + LOG_SHIFT)，预测后再反变换

import os
import re
import warnings
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tabpfn import TabPFNRegressor
from joblib import Parallel, delayed

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    shap = None
    _HAS_SHAP = False


LOG_SHIFT = 1.0


# =========================================================
# 通用工具
# =========================================================
def _get(cfg: Dict[str, Any], dotted: Any, default=None, *, ctx: str = ""):
    """
    支持 @paths.xxx 这种引用写法。
    """
    if isinstance(dotted, str) and dotted.startswith("@"):
        m = re.match(r"@([^\.]+)\.(.+)", dotted)
        if not m:
            raise ValueError(f"_get 解析引用失败：{dotted}（格式应为 @root.key）")

        root, key = m.group(1), m.group(2)
        sub = cfg.get(root, {})
        if key not in sub:
            raise ValueError(
                f"_get 解析 {ctx or dotted} 失败：cfg['{root}'] 中不存在键 '{key}'。\n"
                f"已存在键：{list(sub.keys())}"
            )
        return sub[key]

    if dotted is None:
        if default is None:
            raise ValueError(f"_get 收到 None 且未提供 default（ctx={ctx}）")
        return default

    return dotted


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.abspath(path), exist_ok=True)


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


def _load_dataframe(path: str, sep: str = ",", enc: str = "utf-8-sig") -> pd.DataFrame:
    """
    读取 CSV，带编码兜底。
    """
    tried = []
    for one_enc in [enc, "utf-8-sig", "utf-8", "gb18030"]:
        if one_enc in tried:
            continue
        tried.append(one_enc)
        try:
            return pd.read_csv(path, sep=sep, encoding=one_enc)
        except Exception:
            continue

    raise ValueError(f"读取 CSV 失败：{path}，尝试过编码 {tried}")


# =========================================================
# 数据构建与预处理
# =========================================================
def build_xy_from_single_csv(
    cfg: Dict[str, Any],
    *,
    in_csv: str,
    target: str,
    feature_cols: Optional[List[str]],
    exclude_cols: Optional[List[str]],
    test_split: float,
    seed: int,
    sep: str,
    enc: str,
    preprocess_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series, List[str], pd.DataFrame]:
    """
    从单个 CSV 构造 X/y，并划分训练/测试集。
    返回：
        X_tr, y_tr, X_te, y_te, used_features, df_clean
    """
    preprocess_cfg = preprocess_cfg or {}
    in_csv = _get(cfg, in_csv, ctx="in_csv")
    df = _load_dataframe(in_csv, sep=sep, enc=enc)

    if target not in df.columns:
        raise ValueError(f"目标列不存在：{target}")

    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")

    # 特征列选择
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"以下 feature_cols 不在 CSV 中：{missing}")
        used_features = list(feature_cols)
    else:
        exclude_cols = exclude_cols or []
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        used_features = [c for c in numeric_cols if c != target and c not in exclude_cols]

    if not used_features:
        raise ValueError("没有可用特征列，请检查 feature_cols / exclude_cols 配置。")

    # 特征统一转数值
    for c in used_features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1) 先丢掉目标缺失
    if preprocess_cfg.get("dropna_target", True):
        before = len(df)
        df = df.dropna(subset=[target]).reset_index(drop=True)
        print(f"[Preprocess] dropna_target: {before} -> {len(df)}")

    # 2) 对特征做中位数填补
    if preprocess_cfg.get("fillna_median", True):
        med = df[used_features].median(numeric_only=True)
        df[used_features] = df[used_features].fillna(med)
        print("[Preprocess] fillna_median: 已对特征做中位数填补")

    # 3) 删除常量列
    if preprocess_cfg.get("drop_constant", True):
        keep_features = [c for c in used_features if df[c].nunique(dropna=False) > 1]
        drop_features = [c for c in used_features if c not in keep_features]
        if drop_features:
            print(f"[Preprocess] drop_constant: 删除常量列 {drop_features}")
        used_features = keep_features

    if not used_features:
        raise ValueError("预处理后无可用特征列。")

    # 再保险：只保留特征和目标都非空的样本
    mask = df[used_features].notna().all(axis=1) & df[target].notna()
    df = df.loc[mask].reset_index(drop=True)

    if len(df) < 5:
        raise ValueError(f"清洗后有效样本过少：{len(df)}")

    X_all = df[used_features].to_numpy(dtype=float)
    y_all = df[target].astype(float).reset_index(drop=True)

    if not (0.0 < float(test_split) < 1.0):
        raise ValueError(f"test_split 必须在 (0,1)，当前为 {test_split}")

    idx_all = np.arange(len(df))
    idx_tr, idx_te = train_test_split(
        idx_all,
        test_size=float(test_split),
        random_state=seed,
        shuffle=True,
    )

    X_tr = X_all[idx_tr]
    X_te = X_all[idx_te]
    y_tr = y_all.iloc[idx_tr].reset_index(drop=True)
    y_te = y_all.iloc[idx_te].reset_index(drop=True)

    print(
        f"[Data] 总样本数={len(df)} | 训练集={len(y_tr)} | 测试集={len(y_te)} | "
        f"特征数={len(used_features)}"
    )

    return X_tr, y_tr, X_te, y_te, used_features, df


# =========================================================
# 目标分箱重采样（只作用于训练集）
# =========================================================
def apply_binning_resample_on_target(
    X_tr: np.ndarray,
    y_tr: pd.Series,
    *,
    n_bins: int = 10,
    mode: str = "undersample",
    max_per_bin: Optional[int] = None,
    strategy: str = "quantile",
    seed: int = 42,
) -> Tuple[np.ndarray, pd.Series]:
    y = y_tr.to_numpy(dtype=float)

    if len(y) < n_bins or np.allclose(y.min(), y.max()):
        print(f"[Binning] 样本不足或目标变化太小，跳过分箱重采样。n={len(y)}")
        return X_tr, y_tr

    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(y, qs)
    elif strategy == "uniform":
        edges = np.linspace(y.min(), y.max(), n_bins + 1)
    else:
        raise ValueError(f"[Binning] 未知 strategy={strategy}")

    if np.allclose(edges[0], edges[-1]):
        print("[Binning] 分箱边界 collapse，跳过重采样。")
        return X_tr, y_tr

    bin_idx = np.digitize(y, edges[1:-1], right=False)
    rng = np.random.default_rng(seed)

    unique_bins, counts = np.unique(bin_idx, return_counts=True)
    count_map = dict(zip(unique_bins, counts))
    print(f"[Binning] 原始各 bin 样本数：{count_map}")

    if max_per_bin is None:
        max_per_bin = int(np.median(list(count_map.values())))
        max_per_bin = max(max_per_bin, 1)

    selected_indices: List[np.ndarray] = []

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
        for b in unique_bins:
            idxs = np.where(bin_idx == b)[0]
            if len(idxs) >= target_n:
                choose = rng.choice(idxs, size=target_n, replace=False)
            else:
                choose = rng.choice(idxs, size=target_n, replace=True)
            selected_indices.append(choose)
    else:
        raise ValueError(f"[Binning] 未知 mode={mode}")

    selected_indices = np.concatenate(selected_indices)
    rng.shuffle(selected_indices)

    X_new = X_tr[selected_indices]
    y_new = y_tr.iloc[selected_indices].reset_index(drop=True)

    y_new_np = y_new.to_numpy(dtype=float)
    bin_idx_new = np.digitize(y_new_np, edges[1:-1], right=False)
    unique_bins_new, counts_new = np.unique(bin_idx_new, return_counts=True)
    count_map_new = dict(zip(unique_bins_new, counts_new))

    print(f"[Binning] 重采样后各 bin 样本数：{count_map_new}，总样本数由 {len(y_tr)} -> {len(y_new)}")
    return X_new, y_new



# =========================================================
# Kanda (2013) soft-prior loss: trainable OOF calibration layer
# TabPFN weights remain frozen; gradients update calibrator only.
# =========================================================

def _kanda_values(X, feature_names, target, method="M1"):
    names = list(feature_names or [])
    required = ["lp", "lf", "have", "hmax", "hstd"]
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(f"[Kanda] 缺少原始形态学特征: {missing}; 当前特征: {names}")
    data = {c: np.asarray(X[:, names.index(c)], float) for c in required}
    lp, lf, have, hmax, std = [data[k] for k in required]
    mask = (np.isfinite(lp) & np.isfinite(lf) & np.isfinite(have) &
            np.isfinite(hmax) & np.isfinite(std) & (have > 0) &
            (hmax > 0) & (std >= 0) & (lp > 0) & (lp <= 1) &
            (lf > 0))
    meth = str(method).upper()
    if meth not in {"M1", "M2"}:
        raise ValueError("[Kanda] method 只支持 M1/M2")
    vals = np.full(len(X), np.nan, float)
    if target.lower().startswith("d"):
        x = np.divide(std + have, hmax, out=np.full_like(hmax, np.nan), where=hmax>0)
        mask &= (x >= 0) & (x <= 1)
        a,b,c = (1.29,0.36,-0.17) if meth=="M1" else (0.86,0.28,-0.18)
        vals = hmax * (c*x*x + (a*np.power(np.maximum(lp,0),b)-c)*x)
    elif "z0" in target.lower():
        a,b,c = (0.71,20.21,-0.77) if meth=="M1" else (0.93,8.93,4.68)
        rmac = 1.0 + np.power(4.43,-lp)*(lp-1.0)
        z = 1.0-rmac
        q = (0.5*1.0*1.2 / (0.4**2))*z*lf
        mask &= q>0
        with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
            zmac = have*z*np.exp(-np.power(q,-0.5))
            yy = lp*std/have
            vals = zmac*(b*yy*yy+c*yy+a)
    else:
        raise ValueError(f"[Kanda] 无法识别目标 {target}, 应为 d 或 z0")
    mask &= np.isfinite(vals)
    return vals, mask


class _KandaCalibrator:
    """Minimal learnable residual layer with differentiable data + Kanda loss."""
    def __init__(self, params, scale):
        self.params=np.asarray(params, float)
        self.scale=float(scale)

    def predict(self, base, phys, mask):
        base=np.asarray(base,float).ravel()
        phys=np.asarray(phys,float).ravel()
        valid=np.asarray(mask,bool)
        # Constant + base adjustment + physical residual only where formula is valid.
        delta=np.where(valid, phys-base, 0.0)
        return base + self.scale*self.params[0] + self.params[1]*base + self.params[2]*delta


def _fit_kanda_calibrator(X, y, feature_names, target, log_mode, device, seed, phys_cfg):
    """OOF TabPFN predictions prevent in-sample calibration leakage."""
    y=np.asarray(y,float).ravel()
    alpha=float(phys_cfg.get("alpha",0.05))
    if alpha<0:
        raise ValueError("[Kanda] alpha 不能为负")
    method=phys_cfg.get("method","M1")
    values,mask=_kanda_values(X,feature_names,target,method)
    if not mask.any():
        raise ValueError("[Kanda] 没有符合公式适用域的训练样本")
    nfold=min(int(phys_cfg.get("oof_folds",3)),len(y))
    if nfold<2:
        raise ValueError("[Kanda] oof_folds 必须 >=2")
    oof=np.empty(len(y),float)
    for ti,vi in KFold(n_splits=nfold,shuffle=True,random_state=seed).split(X):
        sc=StandardScaler().fit(X[ti])
        m=TabPFNRegressor(device=device,random_state=seed)
        fit_y=np.log(y[ti]+LOG_SHIFT) if log_mode else y[ti]
        m.fit(sc.transform(X[ti]),fit_y)
        pred=np.asarray(m.predict(sc.transform(X[vi])),float).ravel()
        oof[vi]=np.exp(pred)-LOG_SHIFT if log_mode else pred
    if not np.isfinite(oof).all():
        raise ValueError("[Kanda] OOF 预测包含非有限值")
    scale=float(phys_cfg.get("scale",np.std(y)))
    if scale<=0 or not np.isfinite(scale):
        scale=max(float(np.mean(np.abs(y))),1.0)
    # Parametric differentiable layer in original physical units.
    torch.manual_seed(seed)
    p=torch.nn.Parameter(torch.zeros(3,dtype=torch.float64))
    opt=torch.optim.Adam([p],lr=float(phys_cfg.get("lr",0.01)))
    tbase=torch.tensor(oof,dtype=torch.float64)
    ty=torch.tensor(y,dtype=torch.float64)
    tp=torch.tensor(np.where(mask,values,oof),dtype=torch.float64)
    tm=torch.tensor(mask,dtype=torch.bool)
    steps=int(phys_cfg.get("epochs",300))
    reg=float(phys_cfg.get("regularization",0.01))
    for _ in range(steps):
        opt.zero_grad()
        pred=tbase+scale*p[0]+p[1]*tbase+p[2]*torch.where(tm,tp-tbase,0.)
        ldata=torch.mean(((pred-ty)/scale)**2)
        lphys=torch.mean(((pred[tm]-tp[tm])/scale)**2)
        loss=ldata+alpha*lphys+reg*torch.sum(p*p)
        loss.backward()
        opt.step()
    print(f"[Kanda] method={method} alpha={alpha} valid={mask.sum()}/{len(mask)} "
          f"data_loss={ldata.item():.6g} physics_loss={lphys.item():.6g}")
    return _KandaCalibrator(p.detach().numpy(),scale)


def _apply_kanda_calibration(base, X_raw, feature_names, target, calibrator, phys_cfg):
    vals,mask=_kanda_values(X_raw,feature_names,target,phys_cfg.get("method","M1"))
    return calibrator.predict(base,vals,mask)

# =========================================================
# K 折交叉验证
# =========================================================
def run_kfold_cv(
    X_tr: np.ndarray,
    y_tr: pd.Series,
    kf_num: int,
    device: str,
    log_mode: bool,
    seed: int,
    feature_names=None, target=None, physics_cfg=None,
) -> Dict[str, Any]:
    if log_mode:
        y_min = float(y_tr.min())
        if y_min <= -LOG_SHIFT:
            raise ValueError(
                f"log_mode=True 时，要求 y > {-LOG_SHIFT}。当前训练集最小值={y_min:.6f}"
            )

    kf = KFold(n_splits=int(kf_num), shuffle=True, random_state=seed)
    y_tr_np = y_tr.to_numpy(dtype=float)
    fold_rows = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr), start=1):
        scaler = StandardScaler().fit(X_tr[tr_idx])
        Xtr = scaler.transform(X_tr[tr_idx])
        Xval = scaler.transform(X_tr[val_idx])

        if log_mode:
            y_fit = np.log(y_tr_np[tr_idx] + LOG_SHIFT)
            y_val = np.log(y_tr_np[val_idx] + LOG_SHIFT)
        else:
            y_fit = y_tr_np[tr_idx]
            y_val = y_tr_np[val_idx]

        model = TabPFNRegressor(device=device, random_state=seed)
        model.fit(Xtr, y_fit)
        y_pred = model.predict(Xval)

        if log_mode:
            y_pred = np.exp(y_pred) - LOG_SHIFT
            y_val_eval = np.exp(y_val) - LOG_SHIFT
        else:
            y_val_eval = y_val

        if physics_cfg and physics_cfg.get("enable",False):
            calibrator=_fit_kanda_calibrator(X_tr[tr_idx], y_tr_np[tr_idx],
                feature_names,target,log_mode,device,seed+fold,physics_cfg)
            y_pred=_apply_kanda_calibration(y_pred,X_tr[val_idx],feature_names,
                target,calibrator,physics_cfg)

        r2 = r2_score(y_val_eval, y_pred)
        mae = mean_absolute_error(y_val_eval, y_pred)
        mse = mean_squared_error(y_val_eval, y_pred)

        fold_rows.append({
            "fold": fold,
            "r2": float(r2),
            "mae": float(mae),
            "mse": float(mse),
        })

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


# =========================================================
# SHAP 并行
# =========================================================
def _split_chunks(X: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    chunk_size = max(int(chunk_size), 1)
    return [X[i:i + chunk_size] for i in range(0, len(X), chunk_size)]


def _normalize_shap_values(shap_values: Any) -> np.ndarray:
    arr = np.array(shap_values)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]

    raise ValueError(f"[SHAP] 无法识别的 shap_values 形状：{arr.shape}")


def _shap_chunk_worker(
    model,
    background: np.ndarray,
    X_chunk: np.ndarray,
    method: str,
    kernel_nsamples: Any,
    kernel_l1_reg: Any,
    perm_npermutations: int,
    seed: int,
) -> np.ndarray:
    """
    真正执行单个 chunk 的 SHAP 计算。
    """
    def f_pred(x):
        return model.predict(x)

    method = str(method).lower()

    if method == "kernel":
        explainer = shap.KernelExplainer(f_pred, background)
        shap_values = explainer.shap_values(
            X_chunk,
            nsamples=kernel_nsamples,
            l1_reg=kernel_l1_reg,
            silent=True,
        )
    elif method == "permutation":
        explainer = shap.PermutationExplainer(f_pred, background, seed=seed)
        shap_values = explainer.shap_values(
            X_chunk,
            npermutations=int(perm_npermutations),
            silent=True,
        )
    else:
        raise ValueError(f"[SHAP] method 仅支持 kernel / permutation，当前={method}")

    return _normalize_shap_values(shap_values)


def _shap_chunk_worker_with_log(
    idx: int,
    total_chunks: int,
    model,
    background: np.ndarray,
    X_chunk: np.ndarray,
    method: str,
    kernel_nsamples: Any,
    kernel_l1_reg: Any,
    perm_npermutations: int,
    seed: int,
) -> Tuple[int, np.ndarray]:
    """
    带日志的单个 chunk 计算。
    返回 (idx, shap_part)，便于最终按顺序还原。
    """
    shap_part = _shap_chunk_worker(
        model=model,
        background=background,
        X_chunk=X_chunk,
        method=method,
        kernel_nsamples=kernel_nsamples,
        kernel_l1_reg=kernel_l1_reg,
        perm_npermutations=perm_npermutations,
        seed=seed,
    )
    print(f"[SHAP] 已完成 chunk {idx}/{total_chunks} | 当前块样本数={len(X_chunk)}")
    return idx, shap_part


def compute_mean_abs_shap_parallel(
    model,
    background: np.ndarray,
    X_shap: np.ndarray,
    *,
    method: str = "permutation",
    n_jobs: int = 1,
    backend: str = "threading",
    chunk_size: int = 8,
    verbose: int = 0,
    kernel_nsamples: Any = "auto",
    kernel_l1_reg: Any = "num_features(10)",
    perm_npermutations: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    按样本块并行计算 SHAP，返回 mean(|SHAP|)。
    每完成一个 chunk，会打印一条进度。
    推荐 backend="threading"。
    """
    if X_shap is None or len(X_shap) == 0:
        raise ValueError("[SHAP] X_shap 为空")

    method = str(method).lower()
    backend = str(backend).lower()
    n_jobs = max(int(n_jobs), 1)
    chunks = _split_chunks(X_shap, chunk_size)
    total_chunks = len(chunks)

    print(
        f"[SHAP] method={method} | backend={backend} | "
        f"n_jobs={n_jobs} | chunk_size={chunk_size} | n_chunks={total_chunks}"
    )

    if n_jobs == 1 or total_chunks == 1:
        results = []
        for i, chunk in enumerate(chunks, start=1):
            idx, part = _shap_chunk_worker_with_log(
                idx=i,
                total_chunks=total_chunks,
                model=model,
                background=background,
                X_chunk=chunk,
                method=method,
                kernel_nsamples=kernel_nsamples,
                kernel_l1_reg=kernel_l1_reg,
                perm_npermutations=perm_npermutations,
                seed=seed,
            )
            results.append((idx, part))
    else:
        try:
            results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
                delayed(_shap_chunk_worker_with_log)(
                    idx=i,
                    total_chunks=total_chunks,
                    model=model,
                    background=background,
                    X_chunk=chunk,
                    method=method,
                    kernel_nsamples=kernel_nsamples,
                    kernel_l1_reg=kernel_l1_reg,
                    perm_npermutations=perm_npermutations,
                    seed=seed,
                )
                for i, chunk in enumerate(chunks, start=1)
            )
        except Exception as e:
            raise RuntimeError(
                f"[SHAP] 并行失败。当前 backend={backend}, n_jobs={n_jobs}。"
                f"建议优先使用 backend='threading' 或 n_jobs=1。原始错误：{e}"
            )

    results = sorted(results, key=lambda x: x[0])
    parts = [part for _, part in results]

    shap_matrix = np.concatenate(parts, axis=0)

    print(f"[SHAP] 全部完成 | total_chunks={total_chunks} | total_samples={len(X_shap)}")
    return shap_matrix


# =========================================================
# 最终拟合 + 测试集评估 + SHAP
# =========================================================
def fit_final_and_eval_holdout(
    X_tr: np.ndarray,
    y_tr: pd.Series,
    X_te: np.ndarray,
    y_te: pd.Series,
    *,
    device: str,
    log_mode: bool,
    seed: int,
    feature_names: Optional[List[str]] = None,
    shap_cfg: Optional[Dict[str, Any]] = None,
    target=None, physics_cfg=None,
) -> Tuple[Dict[str, float], np.ndarray, Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    返回：
        test_metrics, y_pred_test, shap_df
    """
    shap_cfg = shap_cfg or {}

    if log_mode:
        y_tr_min = float(y_tr.min())
        y_te_min = float(y_te.min())
        if y_tr_min <= -LOG_SHIFT or y_te_min <= -LOG_SHIFT:
            raise ValueError(
                f"log_mode=True 且使用 log(y + {LOG_SHIFT}) 时，要求 y > {-LOG_SHIFT}。"
                f"当前 y_train_min={y_tr_min:.6f}, y_test_min={y_te_min:.6f}"
            )

    scaler = StandardScaler().fit(X_tr)
    Xtr_full = scaler.transform(X_tr)
    Xte_full = scaler.transform(X_te)

    y_tr_np = y_tr.to_numpy(dtype=float)
    if log_mode:
        y_fit = np.log(y_tr_np + LOG_SHIFT)
    else:
        y_fit = y_tr_np

    model = TabPFNRegressor(device=device, random_state=seed)
    model.fit(Xtr_full, y_fit)

    y_pred = model.predict(Xte_full)
    if log_mode:
        y_pred = np.exp(y_pred) - LOG_SHIFT

    if physics_cfg and physics_cfg.get("enable",False):
        calibrator=_fit_kanda_calibrator(X_tr,y_tr_np,feature_names,target,
            log_mode,device,seed,physics_cfg)
        y_pred=_apply_kanda_calibration(y_pred,X_te,feature_names,target,
            calibrator,physics_cfg)

    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)

    test_metrics = {
        "r2": float(r2),
        "mae": float(mae),
        "mse": float(mse),
        "n_test": int(len(y_te)),
    }

    shap_df = None
    shap_plot_pack = None

    do_shap = bool(shap_cfg.get("enable", False))
    if do_shap and physics_cfg and physics_cfg.get("enable",False):
        warnings.warn("[SHAP] 当前解释的是原始 TabPFN，不包含 Kanda 校正层；请勿将其称为最终预测的 SHAP")
    if do_shap and not _HAS_SHAP:
        warnings.warn("未安装 shap，跳过 SHAP 计算。")
        do_shap = False

    if do_shap:
        try:
            background_size = int(shap_cfg.get("background_size", 80))
            max_samples = int(shap_cfg.get("max_samples", 200))
            method = str(shap_cfg.get("method", "permutation")).lower()
            n_jobs = int(shap_cfg.get("n_jobs", 1))
            backend = str(shap_cfg.get("backend", "threading")).lower()
            chunk_size = int(shap_cfg.get("chunk_size", 8))
            verbose = int(shap_cfg.get("verbose", 0))
            kernel_nsamples = shap_cfg.get("kernel_nsamples", "auto")
            kernel_l1_reg = shap_cfg.get("kernel_l1_reg", "num_features(10)")
            perm_npermutations = int(shap_cfg.get("perm_npermutations", 10))

            n_bg = min(background_size, Xtr_full.shape[0])
            bg_idx = np.random.choice(Xtr_full.shape[0], size=n_bg, replace=False)
            background = Xtr_full[bg_idx]

            n_te = Xte_full.shape[0]
            if n_te > max_samples:
                te_idx = np.random.choice(n_te, size=max_samples, replace=False)
                X_shap_scaled = Xte_full[te_idx]
                X_shap_raw = X_te[te_idx]
            else:
                X_shap_scaled = Xte_full
                X_shap_raw = X_te

            shap_matrix = compute_mean_abs_shap_parallel(
                model=model,
                background=background,
                X_shap=X_shap_scaled,
                method=method,
                n_jobs=n_jobs,
                backend=backend,
                chunk_size=chunk_size,
                verbose=verbose,
                kernel_nsamples=kernel_nsamples,
                kernel_l1_reg=kernel_l1_reg,
                perm_npermutations=perm_npermutations,
                seed=seed,
            )

            # 这里必须补上
            mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

            if feature_names is not None and len(feature_names) == len(mean_abs_shap):
                final_feature_names = list(feature_names)
            else:
                final_feature_names = [f"f{i}" for i in range(len(mean_abs_shap))]

            shap_df = pd.DataFrame({
                "feature": final_feature_names,
                "mean_abs_shap": mean_abs_shap,
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            shap_df["rank"] = np.arange(1, len(shap_df) + 1)

            shap_plot_pack = {
                "shap_matrix": shap_matrix,
                "X_shap_raw": X_shap_raw,
                "feature_names": final_feature_names,
            }

        except Exception as e:
            print(f"[WARN] SHAP 计算失败：{e}")
        
    return test_metrics, np.asarray(y_pred, dtype=float).ravel(), shap_df, shap_plot_pack


# =========================================================
# 可视化
# =========================================================
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
    y_true_np = y_true.to_numpy(dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float).ravel()

    if y_true_np.size == 0:
        print(f"[Plot] 空测试集，跳过绘图：{out_path}")
        return

    try:
        m, b = np.polyfit(y_true_np, y_pred_np, 1)
    except Exception:
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
        f"Model Stability ({kf_num}-Fold CV)  "
        f"Mean R²: {cv_r2_mean:.3f} | Std: {cv_r2_std:.3f} | CoV: {cov_r2:.2f}%"
    )
    fig.suptitle(f"{title1}\n{title2}", fontsize=12)
    ax.set_title(f"Evaluation on {split_desc}")

    box_text = (
        f"Split: {split_desc}\n"
        f"Test R²: {test_r2:.3f}\n"
        f"RMSE: {rmse:.3f}\n"
        f"MAE: {mae:.3f}"
    )
    ax.text(
        0.05, 0.95, box_text,
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


# =========================================================
# 落盘
# =========================================================
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


def save_test_metrics(out_dir: str, prefix: str, mode: str, test_metrics: Dict[str, float], split_desc: str, enc: str) -> str:
    path = os.path.join(out_dir, f"{prefix}_{mode}_test_metrics.csv")
    row = {**test_metrics, "split": split_desc}
    pd.DataFrame([row]).to_csv(path, index=False, encoding=enc)
    return path


def save_test_preds(out_dir: str, prefix: str, mode: str, y_te: pd.Series, y_pred: np.ndarray, enc: str) -> str:
    path = os.path.join(out_dir, f"{prefix}_{mode}_test_preds.csv")
    pd.DataFrame({
        "y_true": y_te.to_numpy(dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float).ravel(),
    }).to_csv(path, index=False, encoding=enc)
    return path


def save_shap_importance(out_dir: str, prefix: str, mode: str, shap_df: pd.DataFrame, enc: str) -> str:
    path = os.path.join(out_dir, f"{prefix}_{mode}_shap_importance.csv")
    shap_df.to_csv(path, index=False, encoding=enc)
    return path

def save_shap_beeswarm_plot(
    out_dir: str,
    prefix: str,
    mode: str,
    shap_matrix: np.ndarray,
    X_shap_raw: np.ndarray,
    feature_names: List[str],
    max_display: int = 15,
) -> str:
    """
    保存 SHAP summary beeswarm 图。
    """
    path = os.path.join(out_dir, f"{prefix}_{mode}_shap_beeswarm.png")

    explanation = shap.Explanation(
        values=shap_matrix,
        data=X_shap_raw,
        feature_names=feature_names,
    )

    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SHAP] beeswarm 图已保存：{path}")
    return path

def save_used_features(out_dir: str, prefix: str, used_features: List[str]) -> str:
    path = os.path.join(out_dir, f"{prefix}_used_features.txt")
    with open(path, "w", encoding="utf-8") as f:
        for c in used_features:
            f.write(f"{c}\n")
    return path



# =========================================================
# 外部 CSV 预测：用训练 CSV 拟合模型，预测 BJ_model_input.csv，并写回指定 CSV
# =========================================================
def _default_tabpfn_pred_col(target: str, task_name: str = "") -> str:
    """
    根据目标列名自动推断预测列名。
    - 目标列包含 z0 -> z0_TabPFN
    - 目标列以 d 开头或包含 d_ -> d_TabPFN
    """
    text = f"{target}_{task_name}".lower()
    if "z0" in text:
        return "z0_TabPFN"
    if re.search(r"(^|[_\W])d($|[_\W])", text) or text.startswith("d") or "d_" in text:
        return "d_TabPFN"
    return f"{target}_TabPFN"


def _resolve_predict_col(
    predict_cfg: Dict[str, Any],
    *,
    target: str,
    task_name: str,
    mode_tag: str,
    n_log_modes: int,
) -> str:
    """
    确定外部预测写入的列名。
    若 log_modes 同时包含 raw/log，默认自动添加 _raw / _log 后缀，避免覆盖。
    """
    pred_col = str(predict_cfg.get("pred_col") or _default_tabpfn_pred_col(target, task_name))
    add_mode_suffix = bool(predict_cfg.get("add_mode_suffix", n_log_modes > 1))
    if add_mode_suffix and n_log_modes > 1:
        pred_col = f"{pred_col}_{mode_tag}"
    return pred_col


def build_x_from_prediction_csv(
    cfg: Dict[str, Any],
    *,
    predict_csv: str,
    used_features: List[str],
    train_medians: pd.Series,
    sep: str,
    enc: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    读取待预测 CSV，并按照训练时实际使用的 used_features 构造 X_pred。

    注意：
    - 待预测 CSV 不需要目标列；
    - 即使 BJ 文件中包含 z0_LES / d_LES，这里也不会使用；
    - 缺失值使用训练数据的特征中位数填补，而不是使用待预测数据自身中位数。
    """
    predict_csv = _get(cfg, predict_csv, ctx="predict.in_csv")
    df_pred = _load_dataframe(predict_csv, sep=sep, enc=enc).copy()

    missing = [c for c in used_features if c not in df_pred.columns]
    if missing:
        raise ValueError(
            f"[Predict] 待预测 CSV 缺少训练所需特征列：{missing}\n"
            f"待预测文件：{predict_csv}\n"
            f"当前列名：{list(df_pred.columns)}"
        )

    for c in used_features:
        df_pred[c] = pd.to_numeric(df_pred[c], errors="coerce")

    # 使用训练集/训练CSV的中位数填补 BJ 待预测数据的特征缺失。
    fill_values = train_medians.reindex(used_features)
    if fill_values.isna().any():
        bad = fill_values[fill_values.isna()].index.tolist()
        raise ValueError(f"[Predict] 以下特征的训练中位数为空，无法用于预测缺失值填补：{bad}")

    df_pred[used_features] = df_pred[used_features].fillna(fill_values)

    if df_pred[used_features].isna().any().any():
        bad_cols = df_pred[used_features].columns[df_pred[used_features].isna().any()].tolist()
        raise ValueError(f"[Predict] 以下特征在填补后仍存在 NaN：{bad_cols}")

    X_pred = df_pred[used_features].to_numpy(dtype=float)
    return df_pred, X_pred


def fit_tabpfn_final_predictor(
    X_fit: np.ndarray,
    y_fit: pd.Series,
    *,
    device: str,
    log_mode: bool,
    seed: int,
) -> Tuple[TabPFNRegressor, StandardScaler]:
    """
    拟合最终用于外部数据预测的 TabPFN 模型。
    """
    y_fit_np = pd.Series(y_fit).astype(float).to_numpy(dtype=float)

    if log_mode:
        y_min = float(np.nanmin(y_fit_np))
        if y_min <= -LOG_SHIFT:
            raise ValueError(
                f"[Predict] log_mode=True 时要求 y > {-LOG_SHIFT}，当前训练目标最小值={y_min:.6f}"
            )
        y_model = np.log(y_fit_np + LOG_SHIFT)
    else:
        y_model = y_fit_np

    scaler = StandardScaler().fit(X_fit)
    X_fit_scaled = scaler.transform(X_fit)

    model = TabPFNRegressor(device=device, random_state=seed)
    model.fit(X_fit_scaled, y_model)
    return model, scaler


def _write_prediction_column_to_csv(
    cfg: Dict[str, Any],
    *,
    df_pred_input: pd.DataFrame,
    y_pred: np.ndarray,
    predict_cfg: Dict[str, Any],
    pred_col: str,
    sep: str,
    enc: str,
) -> str:
    """
    将预测结果写入指定 CSV。

    默认行为：
    - out_csv 未提供时，写回预测输入 CSV；
    - write_mode='add_column' 时，如果 out_csv 已存在，则在已有文件中添加/覆盖 pred_col；
    - 若提供 id_cols 且预测输入和输出文件都含有这些列，则按 id_cols 对齐写入；
    - 若预测输入缺少 id_cols、但 out_csv 行数与预测行数一致，则自动退回按行号写入。

    注意：预测输入 CSV 应该是 BJ_model_input.csv 这类待预测数据，而不是训练数据。
    """
    in_csv_raw = (
        predict_cfg.get("predict_csv")
        or predict_cfg.get("data_csv")
        or predict_cfg.get("in_csv")
        or predict_cfg.get("out_csv")
    )
    if not in_csv_raw:
        raise ValueError("[Predict] 必须提供 predict.in_csv / predict.predict_csv / predict.out_csv 之一。")

    in_csv = _get(cfg, in_csv_raw, ctx="predict.in_csv")
    out_csv_raw = predict_cfg.get("out_csv", in_csv_raw)
    out_csv = _get(cfg, out_csv_raw, ctx="predict.out_csv")
    out_csv = os.path.abspath(out_csv)

    write_mode = str(predict_cfg.get("write_mode", "add_column")).lower()
    keep_all_input_cols = bool(predict_cfg.get("keep_all_input_cols", True))
    id_cols = predict_cfg.get("id_cols", None)

    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_pred) != len(df_pred_input):
        raise ValueError(
            f"[Predict] 预测值长度与待预测数据行数不一致：len(y_pred)={len(y_pred)}, len(df)={len(df_pred_input)}"
        )

    pred_df = df_pred_input.copy()
    pred_df[pred_col] = y_pred

    if write_mode in {"overwrite", "replace"}:
        out_df = pred_df if keep_all_input_cols else pred_df[[pred_col]].copy()

    elif write_mode in {"add_column", "append_column", "update_column"}:
        if os.path.exists(out_csv):
            out_df = _load_dataframe(out_csv, sep=sep, enc=enc).copy()

            if id_cols:
                missing_in_pred = [c for c in id_cols if c not in pred_df.columns]
                missing_in_out = [c for c in id_cols if c not in out_df.columns]

                if missing_in_out:
                    raise ValueError(
                        f"[Predict] 使用 id_cols 对齐失败。\n"
                        f"id_cols={id_cols}\n"
                        f"输出文件缺少：{missing_in_out}"
                    )

                if missing_in_pred:
                    # 预测输入缺少 id_cols 时，不能按主键合并。
                    # 如果 out_csv 与预测输入行数一致，则退回按行号写入。
                    # 如果行数不同，通常说明 predict.in_csv 误指向了训练数据。
                    if len(out_df) != len(pred_df):
                        raise ValueError(
                            f"[Predict] 使用 id_cols 对齐失败，且不能退回按行写入。\n"
                            f"id_cols={id_cols}\n"
                            f"预测输入缺少：{missing_in_pred}\n"
                            f"输出文件缺少：{missing_in_out}\n"
                            f"out_csv 行数={len(out_df)}, predict 行数={len(pred_df)}。\n"
                            f"请确认 predict.in_csv 指向 BJ_model_input.csv，而不是训练数据。"
                        )
                    warnings.warn(
                        f"[Predict] 预测输入缺少 id_cols={missing_in_pred}，"
                        f"但 out_csv 行数与预测行数一致，已按行号写入 {pred_col}。"
                    )
                    out_df[pred_col] = y_pred
                else:
                    if pred_df.duplicated(subset=id_cols).any():
                        dup = pred_df.loc[pred_df.duplicated(subset=id_cols, keep=False), id_cols]
                        raise ValueError(f"[Predict] 待预测数据 id_cols 存在重复，无法安全合并：\n{dup.head()}")
                    if out_df.duplicated(subset=id_cols).any():
                        dup = out_df.loc[out_df.duplicated(subset=id_cols, keep=False), id_cols]
                        raise ValueError(f"[Predict] 输出文件 id_cols 存在重复，无法安全合并：\n{dup.head()}")

                    update_df = pred_df[list(id_cols) + [pred_col]].copy()
                    out_df = out_df.drop(columns=[pred_col], errors="ignore")
                    out_df = out_df.merge(update_df, on=list(id_cols), how="left")
            else:
                if len(out_df) != len(pred_df):
                    raise ValueError(
                        f"[Predict] out_csv 已存在，但行数与待预测数据不一致，且未提供 id_cols，无法按行写入。\n"
                        f"out_csv 行数={len(out_df)}, predict 行数={len(pred_df)}"
                    )
                out_df[pred_col] = y_pred
        else:
            if keep_all_input_cols:
                out_df = pred_df
            else:
                if id_cols:
                    missing_id = [c for c in id_cols if c not in pred_df.columns]
                    if missing_id:
                        raise ValueError(f"[Predict] keep_all_input_cols=False 但缺少 id_cols：{missing_id}")
                    out_df = pred_df[list(id_cols) + [pred_col]].copy()
                else:
                    out_df = pd.DataFrame({pred_col: y_pred})
    else:
        raise ValueError(f"[Predict] 未知 write_mode={write_mode}，支持 add_column / overwrite")

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        _ensure_dir(out_dir)
    out_df.to_csv(out_csv, index=False, encoding=enc)
    return out_csv


def run_external_prediction_for_task(
    cfg: Dict[str, Any],
    *,
    task_name: str,
    task_cfg: Dict[str, Any],
    used_features: List[str],
    df_clean: pd.DataFrame,
    X_tr: np.ndarray,
    y_tr: pd.Series,
    target: str,
    device: str,
    seed: int,
    sep: str,
    enc: str,
    log_mode: bool,
    mode_tag: str,
    n_log_modes: int,
    physics_cfg=None,
) -> Optional[str]:
    """
    若 task_cfg.predict.enable=True，则：
    1. 使用训练 CSV 拟合最终模型；
    2. 读取 BJ_model_input.csv；
    3. 生成预测值；
    4. 将预测列写回指定 CSV。

    注意：BJ 文件中即使含有 z0_LES / d_LES，也不会在该函数中用于评价或训练。
    """
    predict_cfg = task_cfg.get("predict", {}) or {}
    if not bool(predict_cfg.get("enable", False)):
        return None

    # 待预测数据路径。优先级：predict_csv / data_csv / in_csv / out_csv。
    # 为了适配“直接把结果写回 BJ_model_input.csv”的用法，若没有写 predict.in_csv，
    # 会默认使用 predict.out_csv 作为待预测输入文件。
    predict_in_csv = (
        predict_cfg.get("predict_csv")
        or predict_cfg.get("data_csv")
        or predict_cfg.get("in_csv")
        or predict_cfg.get("out_csv")
    )
    if not predict_in_csv:
        raise ValueError(
            f"[{task_name}] predict.enable=True 时必须提供 predict.in_csv / predict.predict_csv / predict.out_csv 之一。"
        )

    print(f"[{task_name} | Predict] predict input = {_get(cfg, predict_in_csv, ctx='predict input')}")
    print(f"[{task_name} | Predict] predict output = {_get(cfg, predict_cfg.get('out_csv', predict_in_csv), ctx='predict output')}")

    train_scope = str(predict_cfg.get("train_scope", "all_clean")).lower()

    if train_scope == "all_clean":
        # 使用清洗后的完整训练 CSV 拟合最终外部预测模型。
        if target not in df_clean.columns:
            raise ValueError(f"[{task_name}] df_clean 中缺少目标列：{target}")
        X_fit = df_clean[used_features].to_numpy(dtype=float)
        y_fit = df_clean[target].astype(float).reset_index(drop=True)
        train_medians = df_clean[used_features].median(numeric_only=True)
    elif train_scope in {"train_split", "holdout_train"}:
        # 使用 holdout 划分后的训练集拟合；如果启用了 binning，则这里使用 binning 后的 X_tr/y_tr。
        X_fit = X_tr
        y_fit = y_tr.reset_index(drop=True)
        train_medians = pd.Series(np.nanmedian(X_tr, axis=0), index=used_features)
    else:
        raise ValueError(
            f"[{task_name}] predict.train_scope={train_scope} 不支持。"
            f"可选：all_clean / train_split"
        )

    print(
        f"[{task_name} | {mode_tag.upper()} | Predict] "
        f"使用 train_scope={train_scope} 训练外部预测模型：n={len(y_fit)}, features={len(used_features)}"
    )

    model, scaler = fit_tabpfn_final_predictor(
        X_fit=X_fit,
        y_fit=y_fit,
        device=device,
        log_mode=bool(log_mode),
        seed=seed,
    )

    df_pred, X_pred = build_x_from_prediction_csv(
        cfg,
        predict_csv=predict_in_csv,
        used_features=used_features,
        train_medians=train_medians,
        sep=sep,
        enc=enc,
    )

    X_pred_scaled = scaler.transform(X_pred)
    y_pred = model.predict(X_pred_scaled)
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if log_mode:
        y_pred = np.exp(y_pred) - LOG_SHIFT

    if physics_cfg and physics_cfg.get("enable",False):
        calibrator=_fit_kanda_calibrator(X_fit,np.asarray(y_fit,float),
            used_features,target,log_mode,device,seed,physics_cfg)
        y_pred=_apply_kanda_calibration(y_pred,X_pred,used_features,target,
            calibrator,physics_cfg)

    pred_col = _resolve_predict_col(
        predict_cfg,
        target=target,
        task_name=task_name,
        mode_tag=mode_tag,
        n_log_modes=n_log_modes,
    )

    out_csv = _write_prediction_column_to_csv(
        cfg,
        df_pred_input=df_pred,
        y_pred=y_pred,
        predict_cfg=predict_cfg,
        pred_col=pred_col,
        sep=sep,
        enc=enc,
    )

    print(
        f"[{task_name} | {mode_tag.upper()} | Predict] "
        f"已写入预测列 {pred_col} -> {out_csv} | n_pred={len(y_pred)}"
    )
    return out_csv


# =========================================================
# 单任务执行
# =========================================================
def run_single_tabpfn_task(
    cfg: Dict[str, Any],
    task_name: str,
    task_cfg: Dict[str, Any],
    sep: str,
    enc: str,
    device: str,
    out_dir: str,
    seed: int,
) -> None:
    print(f"[TabPFN] Running task: {task_name}")

    target = task_cfg["target"]
    feature_cols = task_cfg.get("feature_cols")
    exclude_cols = task_cfg.get("exclude_cols")
    kf_num = int(task_cfg.get("kf_num", 5))
    log_modes = task_cfg.get("log_modes", [False])
    output_prefix = task_cfg.get("output_prefix", task_name)
    test_split = float(task_cfg.get("test_split", 0.2))
    in_csv = task_cfg.get("in_csv")

    if not in_csv:
        raise ValueError(f"{task_name}: 单 CSV 简化版必须提供 in_csv。")

    preprocess_cfg = task_cfg.get("preprocess", {})
    shap_cfg = task_cfg.get("shap", {})
    bin_cfg = task_cfg.get("binning", {})
    physics_cfg = task_cfg.get("physics_loss", cfg.get("physics_loss", {})) or {}

    bin_enable = bool(bin_cfg.get("enable", False))
    bin_n_bins = int(bin_cfg.get("n_bins", 10))
    bin_mode = str(bin_cfg.get("mode", "undersample"))
    bin_strategy = str(bin_cfg.get("strategy", "quantile"))
    bin_max_per_bin = bin_cfg.get("max_per_bin", None)
    if bin_max_per_bin is not None:
        bin_max_per_bin = int(bin_max_per_bin)

    _ensure_dir(out_dir)
    task_out_dir = os.path.join(out_dir, output_prefix)
    _ensure_dir(task_out_dir)

    X_tr, y_tr, X_te, y_te, used_features, _df_clean = build_xy_from_single_csv(
        cfg,
        in_csv=in_csv,
        target=target,
        feature_cols=feature_cols,
        exclude_cols=exclude_cols,
        test_split=test_split,
        seed=seed,
        sep=sep,
        enc=enc,
        preprocess_cfg=preprocess_cfg,
    )
    split_desc = f"holdout_{test_split}"

    # 若启用分箱重采样，则只对训练集执行
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

    for log_mode in log_modes:
        mode_tag = "log" if bool(log_mode) else "raw"

        # 1) 训练集上做 K 折 CV
        cv_pack = run_kfold_cv(
            X_tr=X_tr,
            y_tr=y_tr,
            kf_num=kf_num,
            device=device,
            log_mode=bool(log_mode),
            seed=seed,
            feature_names=used_features,
            target=target,
            physics_cfg=physics_cfg,
        )
        save_cv_metrics(task_out_dir, output_prefix, mode_tag, cv_pack, enc)

        cv_summary_meta = {
            "task": task_name,
            "kf_num": kf_num,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
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

        # 2) 最终模型拟合并在测试集上评估
        test_metrics, y_pred_test, shap_df, shap_plot_pack = fit_final_and_eval_holdout(
            X_tr=X_tr,
            y_tr=y_tr,
            X_te=X_te,
            y_te=y_te,
            device=device,
            log_mode=bool(log_mode),
            seed=seed,
            feature_names=used_features,
            shap_cfg=shap_cfg,
            target=target,
            physics_cfg=physics_cfg,
        )

        save_test_metrics(task_out_dir, output_prefix, mode_tag, test_metrics, split_desc, enc)
        save_test_preds(task_out_dir, output_prefix, mode_tag, y_te, y_pred_test, enc)

        if shap_df is not None:
            save_shap_importance(task_out_dir, output_prefix, mode_tag, shap_df, enc)

        save_beeswarm = bool(shap_cfg.get("save_beeswarm", True))
        beeswarm_max_display = int(shap_cfg.get("beeswarm_max_display", 15))

        if save_beeswarm and shap_plot_pack is not None:
            save_shap_beeswarm_plot(
                out_dir=task_out_dir,
                prefix=output_prefix,
                mode=mode_tag,
                shap_matrix=shap_plot_pack["shap_matrix"],
                X_shap_raw=shap_plot_pack["X_shap_raw"],
                feature_names=shap_plot_pack["feature_names"],
                max_display=beeswarm_max_display,
            )

        scatter_path = os.path.join(task_out_dir, f"{output_prefix}_{mode_tag}_{split_desc}_scatter.png")
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

        # 3) 可选：使用训练 CSV 拟合最终模型，并对外部 CSV（例如 BJ_model_input.csv）预测，
        #    预测结果写入 predict.out_csv 或直接写回 predict.in_csv。
        run_external_prediction_for_task(
            cfg=cfg,
            task_name=task_name,
            task_cfg=task_cfg,
            used_features=used_features,
            df_clean=_df_clean,
            X_tr=X_tr,
            y_tr=y_tr,
            target=target,
            device=device,
            seed=seed,
            sep=sep,
            enc=enc,
            log_mode=bool(log_mode),
            mode_tag=mode_tag,
            n_log_modes=len(log_modes),
            physics_cfg=physics_cfg,
        )

    save_used_features(task_out_dir, output_prefix, used_features)


# =========================================================
# 总入口
# =========================================================
def run_tabpfn_tasks_from_cfg(cfg: Dict[str, Any], task_keys: Optional[List[str]] = None) -> None:
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

        run_single_tabpfn_task(
            cfg=cfg,
            task_name=tkey,
            task_cfg=task_cfg,
            sep=sep,
            enc=enc,
            device=device,
            out_dir=out_dir,
            seed=seed,
        )