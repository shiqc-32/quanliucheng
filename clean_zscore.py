# -*- coding: utf-8 -*-
# clean_zscore.py
import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

def _ensure_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

def clean_with_zscore_from_cfg(
    cfg: Dict[str, Any],
    in_csv_path: Optional[str] = None,
    out_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    对指定 CSV 做 Z-score 清洗，并把“是否清洗 & 清洗指标”写入输出 CSV。

    额外约束（新增）：
      - 若行中 z0 < 0 或 d < 0（若这些列存在），该行整行删除，不参与 Z-score 判定。

    读取的配置（YAML）：
      params.encoding       : 统一输出编码（默认 utf-8-sig）
      params.sep            : 分隔符（默认 ,）
      clean.columns         : 需要计算 Z-score 的列名列表（必填）
      clean.z_thresh        : 阈值（默认 3.0）
      clean.mode            : "drop" / "flag"
      clean.keep_na         : True/False
      clean.input_encoding  : 可选，读取输入 CSV 的编码

    调用方可以通过 in_csv_path / out_csv_path 覆盖输入输出路径。
    若未传，则按 cfg.paths.model_input_csv → cfg.paths.urban_zclean_output。
    """
    paths   = cfg["paths"]
    params  = cfg.get("params", {})
    clean   = cfg.get("clean", {})

    # 输入/输出路径（可被函数参数覆盖）
    in_csv   = in_csv_path  or paths["model_input_csv"]
    out_csv  = out_csv_path or paths["urban_zclean_output"]

    sep         = params.get("sep", ",")
    encoding_out= params.get("encoding", "utf-8-sig")
    input_enc   = clean.get("input_encoding", None)

    columns   = clean.get("columns")               # 必填：数值列名
    z_thresh  = float(clean.get("z_thresh", 3.0))
    mode      = str(clean.get("mode", "drop")).lower()
    keep_na   = bool(clean.get("keep_na", True))

    # 1) 读数据（原始 DataFrame）
    df_raw = pd.read_csv(in_csv, sep=sep, encoding=input_enc)
    total_raw = len(df_raw)

    # 1.5) 额外约束：若存在 z0 或 d 列，则删除其中 <0 的行（整行删）
    df = df_raw.copy()
    neg_cols = [c for c in ["z0", "d"] if c in df.columns]
    n_neg_removed = 0
    if neg_cols:
        neg_mask = pd.Series(False, index=df.index)
        for c in neg_cols:
            neg_mask |= (df[c] < 0)
        n_neg_removed = int(neg_mask.sum())
        if n_neg_removed > 0:
            df = df.loc[~neg_mask].copy()
            print(f"[z-clean] 额外约束：删除 {n_neg_removed} 行（列 {neg_cols} 中存在小于 0 的值）")

    # 2) 校验参与列（基于已删除负值后的 df）
    if not columns or not isinstance(columns, list):
        raise ValueError("clean.columns 必须是非空列表（请在 YAML 中配置需要计算 Z-score 的列）。")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"clean.columns 中存在缺失列：{missing}；输入表的列为：{list(df.columns)}")

    num_cols: List[str] = []
    non_numeric: List[str] = []
    for c in columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            non_numeric.append(c)
    if non_numeric:
        raise TypeError(f"以下列不是数值类型，无法计算 Z-score：{non_numeric}")
    if not num_cols:
        raise ValueError("clean.columns 中没有可用于 Z-score 的数值列。")

    # 3) 计算 Z 分数：Z = (x - μ) / σ；σ=0 → 该列Z置0
    mu = df[num_cols].mean()
    sd = df[num_cols].std(ddof=0).replace(0, np.nan)
    z_df = (df[num_cols] - mu) / sd
    z_df = z_df.fillna(0.0)

    # 4) 行级异常判定
    abs_z = z_df.abs()
    row_is_outlier = (abs_z > z_thresh).any(axis=1)
    if not keep_na:
        na_mask = df[num_cols].isna().any(axis=1)
        row_is_outlier = row_is_outlier | na_mask

    # 5) 指标列
    z_max_vals = abs_z.max(axis=1)
    z_max_cols = abs_z.idxmax(axis=1)

    # 6) 组装输出
    out = df.copy()
    out["_zscore_outlier"] = row_is_outlier.astype(int)
    out["_z_max"]          = z_max_vals.fillna(0.0).astype(float)
    out["_z_max_col"]      = z_max_cols.fillna("")
    out["_z_clean_applied"]= 1
    out["_z_mode"]         = mode
    out["_z_thresh"]       = z_thresh
    out["_z_columns"]      = ",".join(num_cols)
    out["_z_keep_na"]      = 1 if keep_na else 0

    if mode == "drop":
        out = out.loc[out["_zscore_outlier"] == 0].copy()

    # 7) 写出
    _ensure_dir(out_csv)
    out.to_csv(out_csv, index=False, encoding=encoding_out, sep=sep)

    # 简报（基于原始行数 + 负值删除 + Z-score 删除）
    outliers = int(row_is_outlier.sum()) if mode == "drop" else int(row_is_outlier.sum())
    kept = int(len(out))
    removed_total = total_raw - kept if mode == "drop" else n_neg_removed  # flag 模式只算负值删除

    print("[z-clean] 完成：")
    print("  输入：", in_csv)
    print("  输出：", out_csv)
    print("  模式：", mode, f"(阈值={z_thresh}, 列={num_cols}, keep_na={keep_na})")
    print(f"  总行（原始）    : {total_raw}")
    print(f"  负值删除（z0/d): {n_neg_removed}")
    print(f"  Z-score 异常行 : {outliers}")
    print(f"  最终保留       : {kept}")
    print(f"  总删除         : {removed_total}")

    return {
        "input": in_csv,
        "output": out_csv,
        "mode": mode,
        "z_thresh": z_thresh,
        "columns_used": num_cols,
        "rows_total": total_raw,
        "rows_neg_removed": n_neg_removed,
        "rows_outlier": outliers,
        "rows_kept": kept,
        "rows_removed": removed_total,
    }
