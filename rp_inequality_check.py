# -*- coding: utf-8 -*-
# rp_inequality_check.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd


# ----------------------------
# 0) 解析引用：@paths.xxx 或 paths.xxx
# ----------------------------

def _get_by_dotted_path(root: Dict[str, Any], dotted: str) -> Any:
    cur: Any = root
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"无法解析引用路径：'{dotted}'，在 '{key}' 处断开。请检查 YAML 是否存在该键。")
        cur = cur[key]
    return cur


def _resolve_ref(root_cfg: Dict[str, Any], value: Any) -> Any:
    """
    支持两种引用写法：
      1) '@paths.urban_model_input'
      2) 'paths.urban_model_input'  （兼容漏写 @ 的情况）
    """
    if not isinstance(value, str):
        return value

    s = value.strip()

    # 标准写法：@xxx.yyy
    if s.startswith("@"):
        dotted = s[1:].strip()
        if dotted == "":
            raise ValueError("检测到非法引用 '@'（后面没有路径）")
        return _get_by_dotted_path(root_cfg, dotted)

    # 兼容写法：paths.xxx / columns.xxx / ...（仅当字符串像“纯引用路径”时才解析）
    if re.match(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$", s) and ("/" not in s) and ("\\" not in s):
        if s.split(".", 1)[0] in root_cfg:  # 顶层存在同名集合才解析
            return _get_by_dotted_path(root_cfg, s)

    return value


def _resolve_obj(root_cfg: Dict[str, Any], obj: Any, max_depth: int = 20) -> Any:
    if max_depth <= 0:
        raise RecursionError("引用解析超过最大深度，可能存在循环引用。")

    if isinstance(obj, dict):
        return {k: _resolve_obj(root_cfg, v, max_depth=max_depth) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_obj(root_cfg, v, max_depth=max_depth) for v in obj]
    if isinstance(obj, str):
        resolved = _resolve_ref(root_cfg, obj)
        if isinstance(resolved, str) and resolved != obj:
            return _resolve_obj(root_cfg, resolved, max_depth=max_depth - 1)
        return resolved
    return obj


# ----------------------------
# 1) 工具
# ----------------------------

def _require_cols(df: pd.DataFrame, cols: List[str], tag: str):
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"[{tag}] 缺少列：{c}")


def _find_rp_cols(df: pd.DataFrame, rp_prefix: str) -> List[str]:
    """
    找到 rp_0, rp_1, ... 这类列，并按编号排序。
    仅匹配：^rp_\d+$
    """
    pat = re.compile(rf"^{re.escape(rp_prefix)}\d+$")
    cols = [c for c in df.columns if isinstance(c, str) and pat.match(c)]
    cols.sort(key=lambda x: int(x.split(rp_prefix, 1)[1]))
    return cols


# ----------------------------
# 2) 主逻辑
# ----------------------------

def run_rp_inequality_pipeline(cfg: Dict[str, Any]) -> str:
    """
    读取 cfg['rp_ineq_check']，检查 lp 与 max(rp_0..rp_n) 是否相等；
    若 |lp - rp_max| <= diff_tol 也视为通过；
    不通过的行输出到 rp_inequality.csv（路径由 YAML 指定）。

    输出列包含：
      - （可选）ID（由 params.id_col 指定且输入存在）
      - case_name（由 params.case_name_col 指定且输入存在；默认 'case_name'）
      - row_index, lp, rp_max_col, rp_max, diff_abs, rp_max_value
    """
    if "rp_ineq_check" not in cfg:
        raise KeyError("cfg 缺少顶层集合：rp_ineq_check")

    # 先对 rp_ineq_check 这段做引用解析（支持 @paths.xxx / paths.xxx）
    sec_raw = cfg["rp_ineq_check"]
    if not isinstance(sec_raw, dict):
        raise TypeError("cfg['rp_ineq_check'] 必须是 dict")
    sec = _resolve_obj(cfg, sec_raw)

    paths = sec.get("paths", {})
    params = sec.get("params", {})

    in_csv = paths.get("in_csv")
    out_csv = paths.get("out_csv")
    if not in_csv or not out_csv:
        raise KeyError("[rp_ineq_check] 必须配置 paths.in_csv 和 paths.out_csv")

    sep = params.get("sep", ",")
    encoding = params.get("encoding", "utf-8-sig")
    lp_col = params.get("lp_col", "lp")
    rp_prefix = params.get("rp_prefix", "rp_")

    # 可选列名（如果输入里存在就带到输出）
    id_col = params.get("id_col", None)
    case_name_col = params.get("case_name_col", "case_name")  # 新增：输出 case_name

    # 浮点误差容忍（isclose）
    atol = float(params.get("atol", 1e-12))
    rtol = float(params.get("rtol", 1e-9))

    # 业务容忍阈值：差值在 0.02 以内也算通过
    diff_tol = float(params.get("diff_tol", 0.02))

    in_csv_p = Path(str(in_csv)).expanduser().resolve()
    out_csv_p = Path(str(out_csv)).expanduser().resolve()

    if not in_csv_p.exists():
        raise FileNotFoundError(f"[rp_ineq_check] 输入文件不存在：{in_csv_p}")

    df = pd.read_csv(in_csv_p, sep=sep, encoding=encoding)

    _require_cols(df, [lp_col], "rp_ineq_check/input")

    rp_cols = _find_rp_cols(df, rp_prefix)
    if not rp_cols:
        raise ValueError(f"[rp_ineq_check] 未找到任何 rp 列（前缀={rp_prefix}，期望 rp_0/rp_1/...）")

    # 数值化：不可转的设 NaN
    lp = pd.to_numeric(df[lp_col], errors="coerce").astype(float)
    rp_mat = df[rp_cols].apply(pd.to_numeric, errors="coerce").astype(float)

    # 逐行最大值及其列名
    rp_max = rp_mat.max(axis=1, skipna=True)
    rp_argmax_col = rp_mat.idxmax(axis=1, skipna=True)

    # 判定通过：isclose 或 绝对差值<=diff_tol
    diff = np.abs(lp.values - rp_max.values)
    ok_close = np.isclose(lp.values, rp_max.values, rtol=rtol, atol=atol, equal_nan=False)
    ok_diff = diff <= diff_tol
    ok = ok_close | ok_diff
    bad_mask = ~ok  # 不通过

    # 输出（不通过行）
    out = pd.DataFrame()

    # （可选）ID
    if id_col is not None and id_col in df.columns:
        out["ID"] = df[id_col]

    # 新增：case_name
    if case_name_col is not None and case_name_col in df.columns:
        out["case_name"] = df[case_name_col]
    else:
        # 不强制报错：如果你希望必须有 case_name，可改为 raise
        out["case_name"] = pd.NA

    out["row_index"] = df.index.astype(int)
    out["lp"] = lp
    out["rp_max_col"] = rp_argmax_col
    out["rp_max"] = rp_max
    out["diff_abs"] = diff  # 方便核对

    # rp_max_value：来自 rp_max_col 的值（便于核对）
    col_to_idx = {c: i for i, c in enumerate(rp_cols)}
    max_col_idx = rp_argmax_col.map(col_to_idx).fillna(-1).astype(int).values
    rp_values = rp_mat.to_numpy()
    rp_max_val = np.array([rp_values[i, j] if j >= 0 else np.nan for i, j in enumerate(max_col_idx)], dtype=float)
    out["rp_max_value"] = rp_max_val

    out_bad = out.loc[bad_mask].copy()

    out_csv_p.parent.mkdir(parents=True, exist_ok=True)
    out_bad.to_csv(out_csv_p, index=False, encoding=encoding)

    total = len(df)
    bad_n = int(bad_mask.sum())
    print(f"[rp_ineq_check] 输入行数: {total}")
    print(f"[rp_ineq_check] 通过条件: isclose(rtol={rtol}, atol={atol}) 或 |lp-rp_max|<= {diff_tol}")
    print(f"[rp_ineq_check] 不通过行数: {bad_n}")
    print(f"[rp_ineq_check] 输出: {out_csv_p}")

    return str(out_csv_p)
