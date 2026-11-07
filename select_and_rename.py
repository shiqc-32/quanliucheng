# -*- coding: utf-8 -*-
# select_and_rename.py
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# ────────────────────────── 工具 ──────────────────────────
def _ensure_parent_dir(path: str) -> None:
    """若输出文件所在目录不存在，则自动创建目录。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def _io_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """仅保留必要的通用参数：分隔符与编码。"""
    params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
    return {
        "sep": params.get("sep", ","),
        "encoding": params.get("encoding", "utf-8-sig"),
    }

def _get_task(cfg: Dict[str, Any], task_key: str) -> Dict[str, Any]:
    tasks = cfg.get("select_rename_tasks", {})
    task = tasks.get(task_key)
    if task is None:
        raise KeyError(f"[select_and_rename] 未找到任务 '{task_key}' 的配置。")
    if not task.get("columns"):
        raise ValueError(f"[select_and_rename] 任务 '{task_key}' 缺少 columns 定义。")
    if not isinstance(task.get("rename_map", {}), dict):
        raise ValueError(f"[select_and_rename] 任务 '{task_key}' 的 rename_map 必须是字典。")
    return task

# ─────────────────────── 数据规范函数 ───────────────────────
def _standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据规范（Data Standardization）：
    1) 若存在 hstd/have/hmax/lp：计算
         hd  = (hstd + have) / hmax
         hz0 = lp * hstd / have
       并将非有限值(±inf, nan)统一设为 NA。
    2) 针对 fg：
       - 若存在 fg：
           * 数字型/可转数字：==1 → 1，否则 0
           * 其他：字符串包含 'realistic' → 1，否则 0
       - 若不存在 fg 且存在 config：由 config 是否包含 'realistic' 生成 fg
    """
    out = df.copy()

    # —— 1) hd / hz0 —— #
    need = {"hstd", "have", "hmax", "lp"}
    if need.issubset(out.columns):
        # 强制数值化，非数值置 NaN
        hstd = pd.to_numeric(out["hstd"], errors="coerce").astype(float)
        have = pd.to_numeric(out["have"], errors="coerce").astype(float)
        hmax = pd.to_numeric(out["hmax"], errors="coerce").astype(float)
        lp   = pd.to_numeric(out["lp"],   errors="coerce").astype(float)

        hd  = (hstd + have) / hmax
        hz0 = (lp * hstd) / have

        hd  = hd.where(np.isfinite(hd),  pd.NA)
        hz0 = hz0.where(np.isfinite(hz0), pd.NA)

        out.loc[:, "hd"]  = hd
        out.loc[:, "hz0"] = hz0

    # —— 2) fg 规范 —— #
    if "fg" in out.columns:
        fg_num = pd.to_numeric(out["fg"], errors="coerce")
        if fg_num.notna().any():
            out.loc[:, "fg"] = (fg_num == 1).astype(int)
        else:
            s = out["fg"].astype(str).str.lower()
            out.loc[:, "fg"] = s.str.contains("realistic", na=False).astype(int)
    elif "config" in out.columns:
        s = out["config"].astype(str).str.lower()
        out.loc[:, "fg"] = s.str.contains("realistic", na=False).astype(int)

    return out

# ─────────────────────── 对外主函数 ───────────────────────
def select_and_rename_from_cfg(
    cfg: Dict[str, Any],
    in_csv_path: str,
    out_csv_path: str,
    task_key: str,
) -> None:
    """
    读取 in_csv_path → 按任务 columns 选择 → 追加 rp_*/U_* 列 → rename_map 改名 →
    数据规范（_standardize_data）→ 新增 ID 列（与 case_name 一一对应）→ 写出 out_csv_path。
    """
    io = _io_params(cfg)
    task = _get_task(cfg, task_key)

    usecols: List[str] = list(task["columns"])
    rename_map: Dict[str, str] = dict(task.get("rename_map", {}))

    # 读：先把整张表读进来，方便额外抓 rp_*/U_* 列
    df_full = pd.read_csv(in_csv_path, encoding=io["encoding"], sep=io["sep"])

    # ---- 自动追加 rp_* 和 U_*（不改名） ----
    extra_rp_cols = [c for c in df_full.columns if isinstance(c, str) and c.startswith("rp_")]
    extra_U_cols  = [c for c in df_full.columns if isinstance(c, str) and c.startswith("U_")]

    # 合并：任务指定列 + rp_* + U_*，去重保持顺序
    selected_cols = []
    for c in usecols + extra_rp_cols + extra_U_cols:
        if c in df_full.columns and c not in selected_cols:
            selected_cols.append(c)

    # 选列
    df = df_full[selected_cols].copy()

    # 改名（注意：rp_* 和 U_* 不在 rename_map 里，所以不会被改名）
    df = df.rename(columns=rename_map)

    # 数据规范（改名之后执行，以 hstd/have/hmax/lp 等“新名”为准）
    df = _standardize_data(df)

    # ------------------ 新增：ID 列（与 case_name 一一对应） ------------------
    if "case_name" not in df.columns:
        raise ValueError("[select_and_rename] 需要生成 ID，但输出表中缺少 'case_name' 列。"
                         "请确认 task.columns 或 rename_map 中包含/生成了 case_name。")

    # 方式 A（推荐）：同一个 case_name 分配同一个 ID（真正“一一对应”）
    # 先按首次出现顺序拿到唯一 case_name 列表
    uniq_cases = pd.Series(df["case_name"]).astype(str)
    uniq_order = uniq_cases.drop_duplicates().tolist()
    id_map = {name: i + 1 for i, name in enumerate(uniq_order)}
    df.insert(0, "ID", uniq_cases.map(id_map).astype("Int64"))

    # 如果你其实想要“每行一个递增 ID”，改成下面这一行并删除上面方式 A：
    # df.insert(0, "ID", pd.Series(range(1, len(df) + 1), index=df.index))

    # -----------------------------------------------------------------------

    # 写出
    _ensure_parent_dir(out_csv_path)
    df.to_csv(out_csv_path, index=False, encoding=io["encoding"], sep=io["sep"])
