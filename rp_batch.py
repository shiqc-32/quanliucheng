# -*- coding: utf-8 -*-
# rp_batch.py
from __future__ import annotations

from pathlib import Path
import math
import re
import copy
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd


# ============================================================
# 0) 解析 YAML 中的 @paths.xxx 引用（严格模式：缺键直接报错）
# ============================================================

def _get_by_dotted_path(root: Dict[str, Any], dotted: str) -> Any:
    """
    从 root dict 中用 'a.b.c' 这种 dotted path 取值。
    """
    cur: Any = root
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"无法解析引用路径：'{dotted}'，在 '{key}' 处断开。请检查 YAML 是否存在该键。")
        cur = cur[key]
    return cur


def _resolve_at_ref(root_cfg: Dict[str, Any], value: str) -> Any:
    """
    解析形如 '@paths.ideal_root' 的引用（严格模式）。
    """
    if not (isinstance(value, str) and value.startswith("@")):
        return value
    dotted = value[1:].strip()  # 去掉 '@'
    if dotted == "":
        raise ValueError("检测到非法引用 '@'（后面没有路径）")
    return _get_by_dotted_path(root_cfg, dotted)


def _resolve_obj(root_cfg: Dict[str, Any], obj: Any, max_depth: int = 20) -> Any:
    """
    递归解析 dict/list/str 中的 '@xxx.yyy' 引用。
    - 只支持“整个字符串就是引用”的情况。
    """
    if max_depth <= 0:
        raise RecursionError("引用解析超过最大深度，可能存在循环引用（A 引用 B，B 又引用 A）")

    if isinstance(obj, dict):
        return {k: _resolve_obj(root_cfg, v, max_depth=max_depth) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_obj(root_cfg, v, max_depth=max_depth) for v in obj]
    if isinstance(obj, str) and obj.startswith("@"):
        resolved = _resolve_at_ref(root_cfg, obj)
        return _resolve_obj(root_cfg, resolved, max_depth=max_depth - 1)
    return obj


# ============================================================
# 0.1 兼容：rp_profile 可扁平写路径键（不再强制 rp_profile.paths）
# ============================================================

def _normalize_rp_cfg_paths(rp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容两种 YAML 写法：

    写法 A（扁平）：
      rp_profile:
        root_dir: ...
        out_csv: ...
        meta_csv: ...
        cnm_path: ...
        out_inequality_csv: ...

    写法 B（嵌套）：
      rp_profile:
        paths:
          root_dir: ...
          out_csv: ...
          ...

    本函数会把写法 A 的路径键搬运/合并到 rp_cfg["paths"]，并返回 rp_cfg。
    """
    if "paths" not in rp_cfg or not isinstance(rp_cfg.get("paths"), dict):
        rp_cfg["paths"] = {}

    flat_keys = ["root_dir", "out_csv", "meta_csv", "cnm_path", "out_inequality_csv"]
    for k in flat_keys:
        if k in rp_cfg and k not in rp_cfg["paths"]:
            rp_cfg["paths"][k] = rp_cfg[k]
    return rp_cfg


# ============================================================
# 0.2 CSV 读取（多编码兜底）
# ============================================================

def _read_csv_flexible(path: Path, sep: str = ",", encoding: Optional[str] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV 不存在：{path}")

    encs: List[str] = []
    if encoding:
        encs.append(encoding)
    encs += ["utf-8-sig", "utf-8", "gbk"]

    last_err = None
    for enc in encs:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"读取 CSV 失败：{path}；最后一次错误：{last_err}")


def _require_cols(df: pd.DataFrame, cols: List[str], tag: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{tag}] 缺少列：{miss}")


# ============================================================
# 1) topo 读取与定位
# ============================================================

def read_topo_matrix(topo_path: Path, dtype=float) -> np.ndarray:
    """
    读取 topo 为二维矩阵（纯数字文本，空格/Tab 分隔）。
    topo 的每个元素表示该格点建筑高度（m），0 表示无建筑。
    """
    arr = np.loadtxt(str(topo_path), dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"topo 读取结果不是二维矩阵：{topo_path} (ndim={arr.ndim})")
    return arr


def find_topo_file(case_dir: Path, topo_glob: str) -> Optional[Path]:
    """
    在案例子目录中按 glob 找 topo 文件。
    约束：每个子目录只会有一个 topo；为稳妥仍取第一个匹配到的文件。
    同时优先选择“文件名或 stem 以 _topo 结尾”的文件。
    """
    hits = sorted([p for p in case_dir.glob(topo_glob) if p.is_file()])
    if not hits:
        return None

    def is_suffix_topo(p: Path) -> bool:
        return p.name.endswith("_topo") or p.stem.endswith("_topo")

    preferred = [p for p in hits if is_suffix_topo(p)]
    chosen = preferred[0] if preferred else hits[0]

    if len(hits) > 1:
        print(f"[WARN] {case_dir.name}: 匹配到 {len(hits)} 个 topo 文件，已选择：{chosen.name}")

    return chosen


# ============================================================
# 2) 核心计算：逐层 rp = A_build / A_total
# ============================================================

def compute_rp_profile_total_denom(
    H: np.ndarray,
    dz: float = 1.0,
    dx: float = 1.0,
    dy: float = 1.0,
    nodata: Optional[float] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    逐层计算 rp(z)=A_build(z)/A_total，层为 [z, z+dz).
    判据：对层底 z0，若格点高度 H > z0，则该格点在该层存在建筑实体。

    返回：rp 数组，长度 = ceil(Hmax/dz)（至少 1 层）
    """
    if dz <= 0:
        raise ValueError("dz 必须 > 0")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx, dy 必须 > 0")

    if nodata is None:
        valid = np.isfinite(H)
    else:
        valid = np.isfinite(H) & (H != nodata)

    H_valid = H[valid]
    if H_valid.size == 0:
        raise ValueError("有效格点数量为 0（请检查 nodata 或文件内容）")

    total_cells = int(valid.sum())
    cell_area = dx * dy
    A_total = total_cells * cell_area

    Hmax = float(np.nanmax(H_valid))
    nlayer = max(int(math.ceil(Hmax / dz)), 1)

    rp = np.empty(nlayer, dtype=float)
    for k in range(nlayer):
        z0 = k * dz
        built_cells = int((valid & (H > z0)).sum())
        A_build = built_cells * cell_area
        rp[k] = A_build / (A_total + eps)

    return rp


# ============================================================
# 3) 批处理：遍历案例、与 meta lp 对比、输出两张表
# ============================================================

def run_rp_profile_from_cfg(rp_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    rp_cfg 为“已经解析完成”的 cfg['rp_profile']。
    - 遍历 root_dir 下每个子文件夹计算 rp_0..rp_n, rp_ave, rp_std
    - 读入 cnm_path (folder_name->meta_name) 与 meta_csv (meta_key_col->lp)
      比较 rp_max 与 lp_base：
        * 不相等时 rel_flag：rp_max>lp_base 为 1，rp_max<lp_base 为 0
    - 输出：
        * out_csv：全量汇总
        * out_inequality_csv（可选）：仅不相等行；若未配置则 out_csv 同目录/rp_inequality.csv
    """
    rp_cfg = _normalize_rp_cfg_paths(rp_cfg)

    # 必填路径（支持扁平写法）
    for k in ["root_dir", "out_csv", "meta_csv", "cnm_path"]:
        if "paths" not in rp_cfg or k not in rp_cfg["paths"]:
            raise KeyError(f"rp_profile 缺少必填参数：{k}（可直接写在 rp_profile 下或写在 rp_profile.paths 下）")

    root_dir = Path(str(rp_cfg["paths"]["root_dir"])).expanduser().resolve()
    out_csv = Path(str(rp_cfg["paths"]["out_csv"])).expanduser().resolve()
    meta_csv = Path(str(rp_cfg["paths"]["meta_csv"])).expanduser().resolve()
    cnm_path = Path(str(rp_cfg["paths"]["cnm_path"])).expanduser().resolve()

    out_ineq = rp_cfg.get("paths", {}).get("out_inequality_csv", None)
    out_ineq_csv = Path(str(out_ineq)).expanduser().resolve() if out_ineq else (out_csv.parent / "rp_inequality.csv")

    topo_glob = rp_cfg.get("topo", {}).get("topo_glob", "*_topo*")
    nodata = rp_cfg.get("topo", {}).get("nodata", None)
    dx = float(rp_cfg.get("topo", {}).get("dx", 1.0))
    dy = float(rp_cfg.get("topo", {}).get("dy", 1.0))
    dz = float(rp_cfg.get("layering", {}).get("dz", 1.0))

    std_ddof = int(rp_cfg.get("stats", {}).get("std_ddof", 0))
    float_fmt = rp_cfg.get("output", {}).get("float_format", None)

    # 读 meta/cnm 的参数
    params = rp_cfg.get("params", {})
    sep = params.get("sep", ",")
    meta_enc = params.get("meta_encoding", None)
    map_enc = params.get("map_encoding", None)
    meta_key_col = params.get("meta_key_col", "NameE")  # meta_csv 中用于 join 的列
    meta_lp_col = params.get("meta_lp_col", "lp")       # meta_csv 中 lp 列名
    diff_tol = float(params.get("diff_tol", 0.0))       # |rp_max-lp_base|<=diff_tol 视为相等

    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir 不存在：{root_dir}")

    # 读入 meta 与 cnm
    df_meta = _read_csv_flexible(meta_csv, sep=sep, encoding=meta_enc)
    df_cnm = _read_csv_flexible(cnm_path, sep=sep, encoding=map_enc)
    _require_cols(df_meta, [meta_key_col, meta_lp_col], "meta_csv")
    _require_cols(df_cnm, ["folder_name", "meta_name"], "cnm")

    # folder_name -> meta_name（重复取第一）
    folder_to_meta = (
        df_cnm.dropna(subset=["folder_name", "meta_name"])
              .drop_duplicates(subset=["folder_name"], keep="first")
              .set_index("folder_name")["meta_name"]
              .to_dict()
    )
    # meta_key -> lp（重复取第一）
    meta_lp_map = (
        df_meta.dropna(subset=[meta_key_col])
               .drop_duplicates(subset=[meta_key_col], keep="first")
               .set_index(meta_key_col)[meta_lp_col]
               .to_dict()
    )

    case_dirs = [p for p in sorted(root_dir.iterdir()) if p.is_dir()]
    if not case_dirs:
        raise ValueError(f"root_dir 下未发现任何子文件夹：{root_dir}")

    records: List[Dict[str, Any]] = []
    max_layers = 0

    for case_dir in case_dirs:
        folder_name = case_dir.name
        topo_path = find_topo_file(case_dir, topo_glob)

        if topo_path is None:
            records.append({
                "ID": folder_name,
                "folder_name": folder_name,
                "error": f"no topo matched: {topo_glob}",
            })
            continue

        try:
            H = read_topo_matrix(topo_path, dtype=float)
            rp = compute_rp_profile_total_denom(H, dz=dz, dx=dx, dy=dy, nodata=nodata)

            max_layers = max(max_layers, len(rp))

            rp_ave = float(np.mean(rp)) if len(rp) > 0 else np.nan
            rp_std = float(np.std(rp, ddof=std_ddof)) if len(rp) > 0 else np.nan
            rp_max = float(np.nanmax(rp)) if len(rp) > 0 else np.nan

            # folder_name -> meta_name -> lp_base
            meta_name = folder_to_meta.get(folder_name, None)
            lp_base = meta_lp_map.get(meta_name, np.nan) if meta_name is not None else np.nan
            lp_base_num = pd.to_numeric(pd.Series([lp_base]), errors="coerce").iloc[0]
            lp_base_num = float(lp_base_num) if pd.notna(lp_base_num) else np.nan

            # compare
            diff_abs = np.nan
            is_equal = False
            rel_flag = pd.NA  # 1: rp_max>lp_base, 0: rp_max<lp_base

            if np.isfinite(rp_max) and np.isfinite(lp_base_num):
                diff_abs = float(abs(rp_max - lp_base_num))
                is_equal = diff_abs <= diff_tol
                if not is_equal:
                    rel_flag = 1 if rp_max > lp_base_num else 0

            records.append({
                "ID": folder_name,
                "folder_name": folder_name,
                "meta_name": meta_name,
                "lp_base": lp_base_num,
                "rp_max": rp_max,
                "diff_abs": diff_abs,
                "rel_flag": rel_flag,
                "is_equal": bool(is_equal) if (np.isfinite(rp_max) and np.isfinite(lp_base_num)) else False,
                "topo_file": topo_path.name,
                "n_layers": int(len(rp)),
                "rp_ave": rp_ave,
                "rp_std": rp_std,
                "_rp_array": rp,
            })

        except Exception as e:
            records.append({
                "ID": folder_name,
                "folder_name": folder_name,
                "topo_file": topo_path.name,
                "error": str(e),
            })

    # 展开 rp_0...rp_{max_layers-1}
    rows: List[Dict[str, Any]] = []
    for rec in records:
        rp_arr = rec.pop("_rp_array", None)
        row = dict(rec)

        for k in range(max_layers):
            row[f"rp_{k}"] = float(rp_arr[k]) if (rp_arr is not None and k < len(rp_arr)) else np.nan

        row.setdefault("n_layers", 0)
        row.setdefault("rp_ave", np.nan)
        row.setdefault("rp_std", np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)

    # 列排序：ID 固定第一；其后关键信息；最后 rp_* 按编号
    rp_cols = sorted(
        [c for c in df.columns if re.match(r"^rp_\d+$", str(c))],
        key=lambda x: int(x.split("_")[1])
    )
    base_cols = [
        "ID",
        "folder_name",
        "meta_name",
        "lp_base",
        "rp_max",
        "diff_abs",
        "rel_flag",
        "is_equal",
        "topo_file",
        "n_layers",
        "rp_ave",
        "rp_std",
        "error",
    ]
    base_cols = [c for c in base_cols if c in df.columns]
    other_cols = [c for c in df.columns if (c not in base_cols and c not in rp_cols)]
    ordered_cols = ["ID"] + [c for c in base_cols if c != "ID"] + other_cols + rp_cols
    df = df.reindex(columns=ordered_cols)

    # 主输出
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format=float_fmt)
    print(f"[OK] rp 汇总表已输出：{out_csv}")

    # 不相等输出（只输出“双方有效且 diff_abs > diff_tol”的行）
    mismatch_mask = df["diff_abs"].notna() & (df["diff_abs"] > diff_tol)
    df_ineq = df.loc[mismatch_mask].copy()

    out_ineq_csv.parent.mkdir(parents=True, exist_ok=True)
    df_ineq.to_csv(out_ineq_csv, index=False, float_format=float_fmt)
    print(f"[OK] rp 不相等表已输出：{out_ineq_csv} (rows={len(df_ineq)})")

    return df


# ============================================================
# 4) 入口：主程序直接 run_rp_profile_pipeline(cfg)
# ============================================================

def run_rp_profile_pipeline(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    主程序传入 yaml.safe_load 得到的完整 cfg(dict)。
    你可以在 YAML 中写：
      rp_profile:
        root_dir: "@paths.ideal_root"
        out_csv: "@paths.ideal_rp_csv"
        meta_csv: "@paths.meta_csv"
        cnm_path: "@paths.cnm_path"
        out_inequality_csv: "@paths.rp_inequality_csv"
    无需再写 rp_profile.paths 子层。
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg 必须是 dict（yaml.safe_load 的返回值）")
    if "rp_profile" not in cfg:
        raise KeyError("cfg 缺少顶层集合：rp_profile")

    rp_cfg_raw = cfg["rp_profile"]
    if not isinstance(rp_cfg_raw, dict):
        raise TypeError("cfg['rp_profile'] 必须是 dict")

    rp_cfg = copy.deepcopy(rp_cfg_raw)

    # 先解析 '@paths.xxx' 引用（严格模式：缺键直接报错）
    rp_cfg = _resolve_obj(cfg, rp_cfg)

    return run_rp_profile_from_cfg(rp_cfg)
