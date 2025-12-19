# -*- coding: utf-8 -*-
# case_name_mapping.py (simplified)
import os
import re
import pandas as pd
from typing import List, Tuple, Dict, Optional

# ---------- 工具 ----------

def list_subdirs(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s0 = str(s).lower().strip()
    s0 = s0.replace("-", " ").replace("_", " ")
    s0 = re.sub(r"[^a-z0-9\s]", " ", s0)          # 仅保留字母数字和空格
    s0 = re.sub(r"\b0+(\d)\b", r"\1", s0)         # 去前导零
    s0 = re.sub(r"\s+", " ", s0).strip()          # 合并多空格
    return s0

def prefix_suffix_match(src: str, candidates: List[str], pre: int = 3, suf: int = 4) -> Tuple[Optional[str], float]:
    """
    仅使用兜底规则：normalize 后，若前 pre 个字符相同且后 suf 个字符相同，则视为匹配。
    仅当字符串长度 >= pre + suf 时才判定。
    命中返回 (原始候选名, 1.0)；未命中返回 (None, 0.0)。
    """
    src_n = normalize_name(src)
    if len(src_n) < (pre + suf):
        return None, 0.0

    for cand_raw in candidates:
        cand_n = normalize_name(cand_raw)
        if len(cand_n) < (pre + suf):
            continue
        if src_n[:pre] == cand_n[:pre] and src_n[-suf:] == cand_n[-suf:]:
            return cand_raw, 1.0
    return None, 0.0

# ---------- 主函数（仅兜底规则；名称在 flow 与 meta 完全一致，二者择一） ----------

def build_case_name_mapping(
    flow_csv_path: str,
    metadata_csv_path: str,
    flow_case_col: str,
    meta_case_col: str,
    meta_config_col: str,            # e.g., "Config"（idealized / realistic）
    idealized_root: str,
    realistic_root: str,
    cutoff: float = 0.72,            # 保留参数以兼容旧接口，但不再使用
    flow_encoding: Optional[str] = None,
    meta_encoding: Optional[str] = None,
    sep: str = ","
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    简化版逻辑：
      - 假定 flowdata 与 metadata 的案例名一致，因此仅以 metadata 的案例名作为“基准名”；
      - 由 metadata.Config 选择 idealized 或 realistic 目录；
      - 在对应目录仅用“前三字符 + 后四字符”的兜底规则匹配文件夹；
      - 不做 difflib 模糊匹配，不做一一对应后处理。

    返回：
      df_map：包含 flow_name、meta_name、Config、folder_root、folder_name、folder_score 等列；
      stats：简单统计信息。
    """

    # 读入 CSV
    df_flow = pd.read_csv(flow_csv_path, encoding=flow_encoding, sep=sep)
    df_meta = pd.read_csv(metadata_csv_path, encoding=meta_encoding, sep=sep)

    # 准备目录候选
    idealized_names = list_subdirs(idealized_root)
    realistic_names = list_subdirs(realistic_root)

    # 将 flow 的名称做成集合，用于存在性标记（名称应与 meta 完全一致）
    flow_name_set = set(df_flow[flow_case_col].astype(str).tolist())

    rows = []
    for _, row in df_meta.iterrows():
        base_name = str(row[meta_case_col])
        config_raw = str(row[meta_config_col]).strip().lower() if pd.notna(row[meta_config_col]) else ""

        # 依据 Config 选择目录与候选
        if "ideal" in config_raw:
            folder_root_used = idealized_root
            folder_candidates = idealized_names
        elif "real" in config_raw:
            folder_root_used = realistic_root
            folder_candidates = realistic_names
        else:
            folder_root_used = None
            folder_candidates = []

        # 仅用兜底规则匹配
        if folder_candidates:
            folder_match, folder_score = prefix_suffix_match(base_name, folder_candidates, pre=3, suf=4)
        else:
            folder_match, folder_score = (None, 0.0)

        rows.append({
            "flow_name": base_name if base_name in flow_name_set else None,  # 二者一致，这里给出在 flow 中是否存在
            "meta_name": base_name,
            "meta_score": 1.0,                       # 名称一致，视作 1.0
            "Config": config_raw,
            "folder_root": folder_root_used,
            "folder_name": folder_match,
            "folder_score": float(folder_score),     # 1.0（命中）或 0.0（未命中）
        })

    df_map = pd.DataFrame(rows)

    # 统计
    stats = {
        "meta_total": len(df_map),
        "in_flow": int(df_map["flow_name"].notna().sum()),
        "folder_matched": int(df_map["folder_name"].notna().sum()),
        "folder_unmatched": int(df_map["folder_name"].isna().sum()),
        "idealized_rows": int(df_map["Config"].fillna("").str.contains("ideal", case=False).sum()),
        "realistic_rows": int(df_map["Config"].fillna("").str.contains("real", case=False).sum()),
    }

    return df_map, stats
