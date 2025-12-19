# -*- coding: utf-8 -*-
# xy_blocks.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============ 基础 I/O ============

def load_dataframe(path: str, sep: str = ",", enc: str = "utf-8-sig") -> pd.DataFrame:
    return pd.read_csv(path, encoding=enc, sep=sep)

# ============ fg 过滤工具 ============

def _normalize_fg_values(fg_filter: Any) -> List[int]:
    """
    将 fg_filter 规范成整数列表：
    - None: 抛给上层处理
    - 单个标量: 变成 [int(x)]
    - list/tuple/set: 全部转为 int
    """
    if fg_filter is None:
        return []
    # 字符串 "0" / "1" 这种也支持
    if isinstance(fg_filter, (int, float, str)):
        return [int(fg_filter)]
    if isinstance(fg_filter, (list, tuple, set)):
        return [int(v) for v in fg_filter]
    raise TypeError(f"fg_filter 类型不支持: {type(fg_filter)}; 期望 int / list / tuple / set / str")

def apply_fg_filter(
    df: pd.DataFrame,
    fg_filter: Any = None,
    *,
    ctx: str = ""
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    按 fg 列过滤数据：
      - fg_filter 为 None: 不做任何操作
      - fg_filter 为 0 / 1 / [0] / [1] / [0,1] 等：只保留 fg 在这些值中的行

    返回 (过滤后的 df, 日志字符串)
    """
    if fg_filter is None:
        return df, None

    if "fg" not in df.columns:
        raise ValueError(
            f"apply_fg_filter: 在 {ctx or 'DataFrame'} 中指定了 fg_filter={fg_filter!r}，"
            f"但数据里不存在 'fg' 列。"
        )

    keep_vals = _normalize_fg_values(fg_filter)
    before = len(df)
    out = df[df["fg"].isin(keep_vals)].copy()
    note = f"fg_filter[{ctx or 'df'}]: keep {keep_vals}, rows {before} -> {len(out)}"
    return out, note

# ============ 巡检与预处理（可插拔） ============

@dataclass
class InspectReport:
    n_rows: int
    n_cols: int
    target: str
    n_target_nan: int
    non_numeric_features: List[str]
    constant_features: List[str]
    nan_rate_by_col: Dict[str, float]
    suggestions: List[str] = field(default_factory=list)

def inspect_df(
    df: pd.DataFrame,
    target: str,
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    constant_tol: float = 1e-12,
) -> InspectReport:
    cols = list(df.columns)
    if feature_cols:
        feats = [c for c in feature_cols if c in cols]
    else:
        excl = set((exclude_cols or []) + [target])
        feats = [c for c in cols if c not in excl]

    non_numeric = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    consts = []
    for c in feats:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            if s.dropna().nunique() <= 1:
                consts.append(c)

    nan_rate = {c: float(df[c].isna().mean()) for c in [target] + feats if c in df.columns}
    n_target_nan = int(df[target].isna().sum()) if target in df.columns else df.shape[0]

    suggestions = []
    if n_target_nan > 0:
        suggestions.append(f"目标列 {target} 含 NaN，建议先处理（丢弃或插补）")
    if non_numeric:
        suggestions.append(f"存在非数值特征列：{non_numeric}，建议转数值或剔除")
    if consts:
        suggestions.append(f"存在常量列：{consts}，建议删除")
    if any(rate > 0 for rate in nan_rate.values()):
        suggestions.append("部分列存在缺失值，建议插补或删除缺失行")

    return InspectReport(
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        target=target,
        n_target_nan=n_target_nan,
        non_numeric_features=non_numeric,
        constant_features=consts,
        nan_rate_by_col=nan_rate,
        suggestions=suggestions,
    )

# 预处理函数类型：接收 df，返回（df, log）
PreprocFn = Callable[[pd.DataFrame], Tuple[pd.DataFrame, str]]

def preprocess_df(
    df: pd.DataFrame,
    steps: Optional[List[PreprocFn]] = None,
    *,
    inplace: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    可插拔预处理：传入若干函数，每个函数签名形如 f(df) -> (df2, note)。
    你可后续把自己的特殊处理函数 append 到 steps 里。
    """
    work = df if inplace else df.copy()
    logs: List[str] = []
    if not steps:
        return work, logs
    for fn in steps:
        work, note = fn(work)
        if note:
            logs.append(note)
    return work, logs

# 提供几个常用内置步骤（可选用）
def step_drop_constant(features: List[str]) -> PreprocFn:
    def _f(df: pd.DataFrame):
        drop_cols = [c for c in features if c in df.columns and df[c].dropna().nunique() <= 1]
        out = df.drop(columns=drop_cols, errors="ignore")
        return out, f"drop_constant: {drop_cols}" if drop_cols else "drop_constant: none"
    return _f

def step_fillna_numeric(features: List[str], strategy: str = "median") -> PreprocFn:
    def _f(df: pd.DataFrame):
        cols = [c for c in features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if strategy == "median":
            vals = {c: df[c].median() for c in cols}
        elif strategy == "mean":
            vals = {c: df[c].mean() for c in cols}
        else:
            raise ValueError("strategy 仅支持 median/mean")
        return df.fillna(value=vals), f"fillna_numeric({strategy}) on {len(cols)} cols"
    return _f

def step_dropna_in(cols: List[str]) -> PreprocFn:
    def _f(df: pd.DataFrame):
        before = len(df)
        out = df.dropna(subset=[c for c in cols if c in df.columns])
        return out, f"dropna rows: {before - len(out)}"
    return _f

# ============ 特征/目标构造与切分 ============

def make_X_y(
    df: pd.DataFrame,
    target: str,
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, pd.Series, List[str]]:
    if feature_cols:
        feats = [c for c in feature_cols if c in df.columns]
    else:
        excl = set((exclude_cols or []) + [target])
        feats = [c for c in df.columns if c not in excl]

    feats_num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    if not feats_num:
        raise ValueError("没有可用的数值特征列，请检查 feature_cols / exclude_cols 与数据。")

    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y, errors="coerce")
    if y.isna().any():
        raise ValueError(f"目标列 {target} 含 NaN 或不可转数字值，请先清洗。")

    X = df[feats_num].to_numpy(dtype=float)
    return X, y, feats_num

def split_indices(n_samples: int, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < float(test_ratio) < 1.0):
        raise ValueError("test_ratio 必须是 (0,1) 小数。")
    idx_all = np.arange(n_samples)
    idx_tr, idx_te = train_test_split(idx_all, test_size=float(test_ratio), random_state=seed, shuffle=True)
    return idx_tr, idx_te

def split_by_indices(
    X: np.ndarray, y: pd.Series, idx_tr: np.ndarray, idx_te: np.ndarray
) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]:
    return (
        X[idx_tr], y.iloc[idx_tr].reset_index(drop=True),
        X[idx_te], y.iloc[idx_te].reset_index(drop=True),
    )

# ============ 单 CSV 工作流：读 →（fg 过滤 / 巡检 / 预处理）→ 构造 X/y → 划分 ============

def build_xy_from_csv(
    in_csv: str,
    target: str,
    sep: str = ",",
    enc: str = "utf-8-sig",
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    test_split: Optional[float] = None,
    seed: int = 42,
    preprocess_steps: Optional[List[PreprocFn]] = None,
    fg_filter: Any = None,  # ★ 新增：按 fg 列过滤
) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series, List[str], np.ndarray, np.ndarray, pd.DataFrame]:
    df_all = load_dataframe(in_csv, sep, enc)

    # ★ 先按 fg 过滤（虚拟城市 vs 真实城市）
    df_all, fg_note = apply_fg_filter(df_all, fg_filter, ctx="single_csv")
    # 这里 fg_note 你要打印也行，现在先不打印

    # 读完先巡检（不急切切分）
    report = inspect_df(df_all, target, feature_cols, exclude_cols)
    if report.suggestions:
        # 这里不强制处理，只是留出可扩展入口
        pass

    # 可选预处理（外部决定 steps；也可为空）
    if preprocess_steps:
        df_all, _logs = preprocess_df(df_all, preprocess_steps)

    X_all, y_all, used_feats = make_X_y(df_all, target, feature_cols, exclude_cols)

    if isinstance(test_split, (int, float)) and 0.0 < float(test_split) < 1.0:
        idx_tr, idx_te = split_indices(len(y_all), float(test_split), seed)
        X_tr, y_tr, X_te, y_te = split_by_indices(X_all, y_all, idx_tr, idx_te)
    else:
        raise ValueError("请提供有效的 test_split (0,1) 小数。")

    return X_tr, y_tr, X_te, y_te, used_feats, idx_tr, idx_te, df_all

# ============ 双 CSV 工作流：训练/测试来自不同数据集 ============

def _align_feature_space(
    df_tr: pd.DataFrame,
    df_te: pd.DataFrame,
    target: str,
    feature_cols: Optional[List[str]],
    exclude_cols: Optional[List[str]],
    align: str = "intersection",  # "intersection" | "train_only" | "union"
) -> List[str]:
    """返回用于两侧的一致特征列清单。默认取交集最稳妥。"""
    if feature_cols:
        feats_tr = [c for c in feature_cols if c in df_tr.columns]
        feats_te = [c for c in feature_cols if c in df_te.columns]
    else:
        excl = set((exclude_cols or []) + [target])
        feats_tr = [c for c in df_tr.columns if c not in excl]
        feats_te = [c for c in df_te.columns if c not in excl]

    if align == "intersection":
        feats = list(sorted(set(feats_tr).intersection(feats_te)))
    elif align == "train_only":
        feats = feats_tr
    elif align == "union":
        # 注意：union 可能引入对方不存在的列（需要填充 NaN 或 0），一般不建议
        feats = list(sorted(set(feats_tr).union(feats_te)))
    else:
        raise ValueError("align 仅支持 intersection/train_only/union")

    # 仅保留数值列
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(df_tr[c]) and pd.api.types.is_numeric_dtype(df_te[c])]
    if not feats:
        raise ValueError("对齐后无共同的数值特征列，请检查列定义或选择 align='train_only' 并确保测试集包含这些列。")
    return feats

@dataclass
class XYPack:
    X_tr: np.ndarray
    y_tr: pd.Series
    X_te: np.ndarray
    y_te: pd.Series
    used_features: List[str]
    idx_tr: np.ndarray
    idx_te: np.ndarray
    df_tr: pd.DataFrame
    df_te: pd.DataFrame
    notes: Dict[str, Any] = field(default_factory=dict)

def build_xy_from_two_csvs(
    train_csv: str,
    test_csv: str,
    target: str,
    *,
    sep: str = ",",
    enc: str = "utf-8-sig",
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    align: str = "intersection",
    preprocess_steps_train: Optional[List[PreprocFn]] = None,
    preprocess_steps_test: Optional[List[PreprocFn]] = None,
    fg_filter_train: Any = None,   # ★ 新增：训练集按 fg 过滤
    fg_filter_test: Any = None,    # ★ 新增：测试集按 fg 过滤
) -> XYPack:
    """
    训练/测试各自来源不同 CSV；读完先 fg 过滤、巡检与可选预处理，再统一列空间后构造 X/y。
    """
    df_tr = load_dataframe(train_csv, sep, enc)
    df_te = load_dataframe(test_csv,  sep, enc)

    # ★ 先按 fg 列过滤
    df_tr, note_tr = apply_fg_filter(df_tr, fg_filter_train, ctx="train")
    df_te, note_te = apply_fg_filter(df_te, fg_filter_test,  ctx="test")

    # 巡检（不强制）
    rep_tr = inspect_df(df_tr, target, feature_cols, exclude_cols)
    rep_te = inspect_df(df_te, target, feature_cols, exclude_cols)

    # 可选预处理
    if preprocess_steps_train:
        df_tr, _ = preprocess_df(df_tr, preprocess_steps_train)
    if preprocess_steps_test:
        df_te, _ = preprocess_df(df_te, preprocess_steps_test)

    # 对齐特征空间
    feats = _align_feature_space(df_tr, df_te, target, feature_cols, exclude_cols, align=align)

    # 构造 X/y
    X_tr = df_tr[feats].to_numpy(dtype=float)
    y_tr = df_tr[target]
    if not pd.api.types.is_numeric_dtype(y_tr):
        y_tr = pd.to_numeric(y_tr, errors="coerce")
    if y_tr.isna().any():
        raise ValueError("训练集目标列含 NaN 或不可转为数值。")

    X_te = df_te[feats].to_numpy(dtype=float)
    y_te = df_te[target]
    if not pd.api.types.is_numeric_dtype(y_te):
        y_te = pd.to_numeric(y_te, errors="coerce")
    if y_te.isna().any():
        raise ValueError("测试集目标列含 NaN 或不可转为数值。")

    idx_tr = np.arange(len(y_tr))
    idx_te = np.arange(len(y_te))
    notes = {
        "align": align,
        "train_rows": int(len(y_tr)),
        "test_rows": int(len(y_te)),
        "inspect_train_suggestions": rep_tr.suggestions,
        "inspect_test_suggestions": rep_te.suggestions,
        "fg_filter_train": fg_filter_train,
        "fg_filter_test": fg_filter_test,
        "fg_note_train": note_tr,
        "fg_note_test": note_te,
    }
    return XYPack(X_tr, y_tr.reset_index(drop=True), X_te, y_te.reset_index(drop=True),
                  feats, idx_tr, idx_te, df_tr, df_te, notes)
