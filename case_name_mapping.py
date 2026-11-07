# -*- coding: utf-8 -*-
# case_name_mapping.py

import os
import re
import json
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any


# =========================================================
# 基础工具
# =========================================================

def safe_str(x: Any) -> str:
    """安全转字符串"""
    if pd.isna(x):
        return ""
    return str(x).strip()


def ensure_parent_dir(path: str):
    """确保输出文件的父目录存在"""
    if path:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def list_subdirs(root: str) -> List[str]:
    """列出 root 下所有一级子文件夹名称"""
    if not root or not os.path.isdir(root):
        return []
    return sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])


def canonical_exact_name(s: str) -> str:
    """
    type3 用的轻量标准化：
    - 去首尾空格
    - 合并多余空格
    - 大小写不敏感
    注意：不删除数字，不改连字符，不改下划线
    """
    s = safe_str(s)
    s = re.sub(r"\s+", " ", s)
    return s.upper()


# =========================================================
# CSV 读取：编码自动兜底
# =========================================================

def _dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x is None:
            continue
        x = str(x).strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def read_csv_with_fallback(
    csv_path: str,
    preferred_encoding: Optional[str] = None,
    sep: str = ",",
    label: str = "csv"
) -> pd.DataFrame:
    """
    按多种编码尝试读取 CSV。
    优先使用 yaml 指定编码；若失败，再自动尝试常见编码。
    """
    encodings_to_try = _dedup_keep_order([
        preferred_encoding,
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
        "latin1",
    ])

    last_err = None
    for enc in encodings_to_try:
        try:
            print(f"[Info] 尝试读取 {label}: path={csv_path}, encoding={enc}, sep={repr(sep)}")
            df = pd.read_csv(csv_path, encoding=enc, sep=sep)
            print(f"[Info] 成功读取 {label}: encoding={enc}, shape={df.shape}")
            return df
        except UnicodeDecodeError as e:
            print(f"[Warn] {label} 使用编码 {enc} 读取失败：{e}")
            last_err = e
            continue
        except Exception as e:
            # 其他异常也打印出来，避免静默
            print(f"[Warn] {label} 使用编码 {enc} 读取失败：{repr(e)}")
            last_err = e
            continue

    raise RuntimeError(
        f"无法读取 {label}: {csv_path}\n"
        f"已尝试编码: {encodings_to_try}\n"
        f"最后一次错误: {repr(last_err)}"
    )


# =========================================================
# 三类案例名规则解析
# =========================================================

TYPE1_RE = re.compile(
    r"^([A-Z]{2})(\d+)-([A-Z]+)-(\d+)_d(\d{2,3})$",
    re.IGNORECASE
)
# 例：VA53-CL-56_d00
# prefix=VA, num1=53, mid=CL, num2=56, wind=00

TYPE2_RE = re.compile(
    r"^([A-Z]{2})-([A-Z]+)-([UV])(\d+)_d(\d{2,3})$",
    re.IGNORECASE
)
# 例：AU-Mel-U1_d00
# region=AU, city=Mel, uv=U, pos_num=1, wind=00


def parse_case_name(name: str) -> Dict[str, Any]:
    """
    将案例名解析为结构化字段。
    识别三类：
    1) VA53-CL-56_d00
    2) AU-Mel-U1_d00 / AU-Ade-V1_d00
    3) AU-Preston / SG-TelokKurau06 等，其余都归入 type3
    """
    raw = safe_str(name)

    # ---------- type1 ----------
    m1 = TYPE1_RE.match(raw)
    if m1:
        return {
            "raw": raw,
            "type": "type1",
            "prefix": m1.group(1).upper(),
            "num1": m1.group(2),
            "mid": m1.group(3).upper(),
            "num2": m1.group(4),
            "wind": m1.group(5),
        }

    # ---------- type2 ----------
    m2 = TYPE2_RE.match(raw)
    if m2:
        return {
            "raw": raw,
            "type": "type2",
            "region": m2.group(1).upper(),
            "city": m2.group(2).upper(),
            "uv": m2.group(3).upper(),
            "pos_num": m2.group(4),
            "wind": m2.group(5),
        }

    # ---------- type3 ----------
    return {
        "raw": raw,
        "type": "type3",
        "exact_name": canonical_exact_name(raw),
    }


def generate_match_keys(parsed: Dict[str, Any]) -> List[Tuple]:
    """
    为一个案例生成匹配键（按优先级排序）

    type1:
      strict : prefix + num1 + mid + num2 + wind
      relaxed: prefix + num1 + num2 + wind

    type2:
      strict : region + city + uv + pos_num + wind
      relaxed: region + pos_num + wind
      注意 relaxed 有意忽略 city 和 U/V，用于支持
      AU-Mel-U1_d00 -> AU-Ade-V1_d00 这类匹配

    type3:
      exact
    """
    t = parsed.get("type", "")

    if t == "type1":
        return [
            ("type1_strict",
             parsed["prefix"], parsed["num1"], parsed["mid"], parsed["num2"], parsed["wind"]),
            ("type1_relaxed",
             parsed["prefix"], parsed["num1"], parsed["num2"], parsed["wind"]),
        ]

    if t == "type2":
        return [
            ("type2_strict",
             parsed["region"], parsed["city"], parsed["uv"], parsed["pos_num"], parsed["wind"]),
            ("type2_relaxed",
             parsed["region"], parsed["pos_num"], parsed["wind"]),
        ]

    if t == "type3":
        return [
            ("type3_exact", parsed["exact_name"])
        ]

    return [
        ("raw_exact", canonical_exact_name(parsed.get("raw", "")))
    ]


# =========================================================
# 候选文件夹索引
# =========================================================

def build_folder_index(folder_names: List[str]) -> Dict[Tuple, List[str]]:
    """
    为候选文件夹建立索引：
        key -> [folder_name1, folder_name2, ...]
    """
    index = defaultdict(list)

    for folder in folder_names:
        parsed = parse_case_name(folder)
        keys = generate_match_keys(parsed)
        for k in keys:
            index[k].append(folder)

    # 去重 + 排序
    out = {}
    for k, v in index.items():
        out[k] = sorted(set(v))
    return out


# =========================================================
# 单个案例匹配
# =========================================================

def choose_unique_candidate(base_name: str, candidates: List[str]) -> Tuple[Optional[str], str]:
    """
    当一个 key 命中多个文件夹时，进一步尝试收敛：
    1) 若存在与 base_name 大小写无关的精确同名，则选它
    2) 否则返回歧义
    """
    if not candidates:
        return None, "no_candidate"

    if len(candidates) == 1:
        return candidates[0], "unique"

    base_exact = canonical_exact_name(base_name)
    exact_hits = [c for c in candidates if canonical_exact_name(c) == base_exact]

    if len(exact_hits) == 1:
        return exact_hits[0], "exact_among_multiple"

    return None, "ambiguous"


def match_case_to_folder(
    base_name: str,
    folder_index: Dict[Tuple, List[str]]
) -> Tuple[Optional[str], float, str, Optional[str], str]:
    """
    将单个案例名匹配到候选文件夹中

    返回：
      folder_match : 匹配到的文件夹名
      folder_score : 匹配分数（兼容旧接口）
      case_type    : type1/type2/type3
      match_rule   : 命中的规则名
      match_note   : 备注信息
    """
    parsed = parse_case_name(base_name)
    case_type = parsed.get("type", "unknown")
    keys = generate_match_keys(parsed)

    for key in keys:
        candidates = folder_index.get(key, [])
        if not candidates:
            continue

        chosen, status = choose_unique_candidate(base_name, candidates)
        rule_name = key[0]

        if chosen is not None:
            if rule_name.endswith("strict"):
                score = 1.0
            elif rule_name.endswith("relaxed"):
                score = 0.95
            elif rule_name.endswith("exact"):
                score = 1.0
            else:
                score = 0.90

            return chosen, score, case_type, rule_name, status

        # 有候选但歧义，则不再继续乱匹配
        return None, 0.0, case_type, rule_name, f"ambiguous: {' | '.join(candidates)}"

    return None, 0.0, case_type, None, "no_match"


# =========================================================
# 核心映射函数
# =========================================================

def build_case_name_mapping(
    flow_csv_path: str,
    metadata_csv_path: str,
    flow_case_col: str,
    meta_case_col: str,
    meta_config_col: str,
    idealized_root: str,
    realistic_root: str,
    cutoff: float = 0.72,            # 兼容保留，不实际使用
    flow_encoding: Optional[str] = None,
    meta_encoding: Optional[str] = None,
    sep: str = ","
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    读取 flowdata + metadata，并依据 metadata.Config 指向对应目录，
    使用三类规则进行文件夹匹配。
    """

    # ---------- 读表（带编码兜底） ----------
    df_flow = read_csv_with_fallback(
        flow_csv_path,
        preferred_encoding=flow_encoding,
        sep=sep,
        label="flow_csv"
    )
    df_meta = read_csv_with_fallback(
        metadata_csv_path,
        preferred_encoding=meta_encoding,
        sep=sep,
        label="metadata_csv"
    )

    # ---------- 列存在性检查 ----------
    if flow_case_col not in df_flow.columns:
        raise KeyError(
            f"flow CSV 中找不到列: {flow_case_col}\n"
            f"实际列名: {list(df_flow.columns)}"
        )
    if meta_case_col not in df_meta.columns:
        raise KeyError(
            f"metadata CSV 中找不到列: {meta_case_col}\n"
            f"实际列名: {list(df_meta.columns)}"
        )
    if meta_config_col not in df_meta.columns:
        raise KeyError(
            f"metadata CSV 中找不到列: {meta_config_col}\n"
            f"实际列名: {list(df_meta.columns)}"
        )

    # ---------- 目录候选 ----------
    idealized_names = list_subdirs(idealized_root)
    realistic_names = list_subdirs(realistic_root)

    idealized_index = build_folder_index(idealized_names)
    realistic_index = build_folder_index(realistic_names)

    # ---------- flow 名称集合 ----------
    flow_name_map = {}
    for x in df_flow[flow_case_col].astype(str).tolist():
        flow_name_map[safe_str(x)] = safe_str(x)

    rows = []

    for _, row in df_meta.iterrows():
        base_name = safe_str(row.get(meta_case_col, ""))
        config_raw = safe_str(row.get(meta_config_col, "")).lower()

        # 根据 Config 选根目录
        if "ideal" in config_raw:
            folder_root_used = idealized_root
            folder_index_used = idealized_index
        elif "real" in config_raw:
            folder_root_used = realistic_root
            folder_index_used = realistic_index
        else:
            folder_root_used = None
            folder_index_used = {}

        # 匹配文件夹
        if folder_root_used is not None:
            folder_match, folder_score, case_type, match_rule, match_note = match_case_to_folder(
                base_name, folder_index_used
            )
        else:
            folder_match, folder_score, case_type, match_rule, match_note = (
                None,
                0.0,
                parse_case_name(base_name)["type"],
                None,
                "unknown_config"
            )

        # 检查 flow 中是否存在同名案例
        flow_hit = flow_name_map.get(base_name, None)

        rows.append({
            "flow_name": flow_hit,
            "meta_name": base_name,
            "meta_score": 1.0 if flow_hit is not None else 0.0,
            "Config": config_raw,
            "case_type": case_type,
            "folder_root": folder_root_used,
            "folder_name": folder_match,
            "folder_score": float(folder_score),
            "match_rule": match_rule,
            "match_note": match_note,
        })

    df_map = pd.DataFrame(rows)

    stats = {
        "meta_total": int(len(df_map)),
        "in_flow": int(df_map["flow_name"].notna().sum()),
        "folder_matched": int(df_map["folder_name"].notna().sum()),
        "folder_unmatched": int(df_map["folder_name"].isna().sum()),
        "idealized_rows": int(df_map["Config"].fillna("").str.contains("ideal", case=False).sum()),
        "realistic_rows": int(df_map["Config"].fillna("").str.contains("real", case=False).sum()),
        "type1_rows": int((df_map["case_type"] == "type1").sum()),
        "type2_rows": int((df_map["case_type"] == "type2").sum()),
        "type3_rows": int((df_map["case_type"] == "type3").sum()),
        "ambiguous_rows": int(df_map["match_note"].fillna("").str.startswith("ambiguous").sum()),
        "unknown_config_rows": int((df_map["match_note"] == "unknown_config").sum()),
        "no_match_rows": int((df_map["match_note"] == "no_match").sum()),
    }

    return df_map, stats


# =========================================================
# cfg 解析辅助函数
# =========================================================

def _deep_get(d: dict, dotted_key: str, default=None):
    """
    支持用 'paths.xxx.yyy' 形式从嵌套 dict 取值
    """
    cur = d
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _resolve_ref(cfg: dict, value, max_depth: int = 8):
    """
    解析类似 '@paths.flow_csv_path' 的引用
    若不是引用，原样返回
    """
    cur = value
    for _ in range(max_depth):
        if isinstance(cur, str) and cur.startswith("@"):
            ref_key = cur[1:]
            nxt = _deep_get(cfg, ref_key, default=None)
            if nxt is None:
                return cur
            cur = nxt
        else:
            return cur
    return cur


def _recursive_find_first(obj, candidate_keys):
    """
    在任意嵌套 dict 中递归搜索候选 key，返回第一个找到的值
    """
    if isinstance(obj, dict):
        # 当前层
        for k in candidate_keys:
            if k in obj:
                return obj[k]
        # 子层
        for _, v in obj.items():
            found = _recursive_find_first(v, candidate_keys)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = _recursive_find_first(item, candidate_keys)
            if found is not None:
                return found

    return None


def _get_cfg_block(cfg: dict, block_name: str) -> dict:
    """
    优先读取根级 block；
    若不存在，再尝试 cfg['case_name_mapping'][block_name]
    """
    if isinstance(cfg.get(block_name), dict):
        return cfg[block_name]

    if isinstance(cfg.get("case_name_mapping"), dict):
        sub = cfg["case_name_mapping"].get(block_name)
        if isinstance(sub, dict):
            return sub

    return {}


def _get_value_from_cfg(cfg: dict, candidate_keys, default=None):
    """
    从 cfg 中尽可能兼容地取值：
    1) 根级 paths / case_name_mapping.paths
    2) 根级直接字段 / case_name_mapping 直接字段
    3) 全 cfg 递归搜索
    4) 自动解析 @paths.xxx
    """
    # ---------- 1) paths ----------
    for block in [
        cfg.get("paths", {}),
        cfg.get("case_name_mapping", {}).get("paths", {}) if isinstance(cfg.get("case_name_mapping"), dict) else {}
    ]:
        if isinstance(block, dict):
            for k in candidate_keys:
                if k in block:
                    return _resolve_ref(cfg, block[k])

    # ---------- 2) 直接字段 ----------
    for scope in [
        cfg,
        cfg.get("case_name_mapping", {}) if isinstance(cfg.get("case_name_mapping"), dict) else {}
    ]:
        if isinstance(scope, dict):
            for k in candidate_keys:
                if k in scope:
                    return _resolve_ref(cfg, scope[k])

    # ---------- 3) 递归搜索 ----------
    found = _recursive_find_first(cfg, candidate_keys)
    if found is not None:
        return _resolve_ref(cfg, found)

    return default


# =========================================================
# 运行入口：main 中直接 run_case_name_mapping(cfg)
# =========================================================

def run_case_name_mapping(cfg: Dict[str, Any]):
    """
    不要求修改 yaml 结构。
    main 中直接：

        run_case_name_mapping(cfg)
        print("[Done] 案例名称映射完成。")

    即可。

    兼容以下配置方式：
      1) 根级：cfg['columns'], cfg['params'], cfg['paths']
      2) 嵌套：cfg['case_name_mapping']['columns'/'params'/'paths']

    若 yaml 中未显式设置输出路径：
      - 默认输出到 metadata_csv_path 同目录下：
          case_name_mapping.csv
          case_name_mapping_stats.json
    """

    # ---------- 读取 columns / params ----------
    columns = _get_cfg_block(cfg, "columns")
    params = _get_cfg_block(cfg, "params")

    flow_case_col = columns.get("flow_case_col", "NameE")
    meta_case_col = columns.get("meta_case_col", "NameE")
    meta_config_col = columns.get("meta_config_col", "Config")

    cutoff = params.get("cutoff", 0.72)
    flow_encoding = params.get("flow_encoding", None)
    meta_encoding = params.get("meta_encoding", None)
    map_encoding = params.get("map_encoding", "utf-8-sig")
    sep = params.get("sep", ",")

    # ---------- 自动寻找路径 ----------
    flow_csv_path = _get_value_from_cfg(
        cfg,
        ["flow_csv_path", "flow_csv", "flowdata_csv", "flowdata_path", "flow_path"]
    )

    metadata_csv_path = _get_value_from_cfg(
        cfg,
        ["metadata_csv_path", "metadata_csv", "meta_csv_path", "metadata_path", "meta_path"]
    )

    idealized_root = _get_value_from_cfg(
        cfg,
        ["idealized_root", "ideal_root"]
    )

    realistic_root = _get_value_from_cfg(
        cfg,
        ["realistic_root", "real_root"]
    )

    output_map_csv = _get_value_from_cfg(
        cfg,
        ["case_name_mapping_csv", "output_map_csv", "map_csv", "case_map_csv"],
        default=None
    )

    output_stats_json = _get_value_from_cfg(
        cfg,
        ["case_name_mapping_stats_json", "output_stats_json", "map_stats_json"],
        default=None
    )

    # ---------- 基本检查 ----------
    if not flow_csv_path:
        raise ValueError(
            "未能从 cfg 中找到 flow CSV 路径。请检查 yaml 中是否存在 "
            "'flow_csv_path' / 'flow_csv' / 'flowdata_csv' / 'flow_path' 之类字段。"
        )

    if not metadata_csv_path:
        raise ValueError(
            "未能从 cfg 中找到 metadata CSV 路径。请检查 yaml 中是否存在 "
            "'metadata_csv_path' / 'metadata_csv' / 'meta_csv_path' / 'metadata_path' 之类字段。"
        )

    if not idealized_root:
        raise ValueError(
            "未能从 cfg 中找到 idealized_root。请检查 yaml 中是否存在 "
            "'idealized_root' 或 'ideal_root'。"
        )

    if not realistic_root:
        raise ValueError(
            "未能从 cfg 中找到 realistic_root。请检查 yaml 中是否存在 "
            "'realistic_root' 或 'real_root'。"
        )

    # ---------- 如果 yaml 没配输出路径，则自动默认 ----------
    if not output_map_csv:
        meta_dir = os.path.dirname(metadata_csv_path) or "."
        output_map_csv = os.path.join(meta_dir, "case_name_mapping.csv")

    if not output_stats_json:
        out_dir = os.path.dirname(output_map_csv) or "."
        output_stats_json = os.path.join(out_dir, "case_name_mapping_stats.json")

    print("[Info] 开始构建案例名称映射...")
    print(f"[Info] flow_csv_path      = {flow_csv_path}")
    print(f"[Info] metadata_csv_path  = {metadata_csv_path}")
    print(f"[Info] idealized_root     = {idealized_root}")
    print(f"[Info] realistic_root     = {realistic_root}")
    print(f"[Info] output_map_csv     = {output_map_csv}")
    print(f"[Info] output_stats_json  = {output_stats_json}")

    # ---------- 执行核心匹配 ----------
    df_map, stats = build_case_name_mapping(
        flow_csv_path=flow_csv_path,
        metadata_csv_path=metadata_csv_path,
        flow_case_col=flow_case_col,
        meta_case_col=meta_case_col,
        meta_config_col=meta_config_col,
        idealized_root=idealized_root,
        realistic_root=realistic_root,
        cutoff=cutoff,
        flow_encoding=flow_encoding,
        meta_encoding=meta_encoding,
        sep=sep
    )

    # ---------- 保存结果 ----------
    ensure_parent_dir(output_map_csv)
    ensure_parent_dir(output_stats_json)

    df_map.to_csv(output_map_csv, index=False, encoding=map_encoding)

    with open(output_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[Info] 映射表已保存: {output_map_csv}")
    print(f"[Info] 统计信息已保存: {output_stats_json}")
    print(f"[Info] meta_total        = {stats['meta_total']}")
    print(f"[Info] folder_matched    = {stats['folder_matched']}")
    print(f"[Info] folder_unmatched  = {stats['folder_unmatched']}")
    print(f"[Info] ambiguous_rows    = {stats['ambiguous_rows']}")
    print(f"[Info] no_match_rows     = {stats['no_match_rows']}")


if __name__ == "__main__":
    pass