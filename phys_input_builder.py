# -*- coding: utf-8 -*-
"""
phys_input_builder.py

形成物理方案输入数据的脚本：

输入：一个总的 CSV（包含 fg, lf, lp, have, hmax, hstd, d, z0, rp_0... 等）
输出：
  1) r_f_total.csv:
       - 只有一列 r_f
       - 数据来自筛选/清洗后的 lf 列（与 Table1 / rp_profiles 行数严格一致）

  2) rp_profiles.csv:
       - 多列 rp_0000, rp_0001, rp_0002, ...
       - 数据来自筛选/清洗后的 rp_0, rp_1, rp_2, ...

  3) Table1-Kanda_R2.csv:
       - 表头：
         ID,FG,lambda_f,lambda_p,Have(m),Hmax(m),Hstd(m),
         d_LES(m),z0_LES(m),d_mac(m),z0_mac(m),
         d_kanda(m),z0_kanda(m),
         d_LES_o_Have,z0_LES_o_Have,d_mac_o_Have,z0_mac_o_Have,
         d_kanda_o_Have,z0_kanda_o_Have
       - d_mac/z0_mac/d_kanda/z0_kanda 使用 Mac/Kanda 公式计算
       - *_o_Have 系列为对应方案的 d/Have、z0/Have（LES、Mac、Kanda 三套）

核心增强（稳妥方案）：
- 所有输入列的“取值范围筛选”由 YAML 指定：phys_input.filters.ranges
- 支持严格/非严格、NaN 处理策略、闭区间/开区间
- 为避免行数错位：在输出 3 份 CSV 前会做“核心列有效性检查”，保证三份输出行对齐
"""

import os
import re
from typing import Dict, Any, Optional, Set, Tuple, List

import numpy as np
import pandas as pd

###################### 公共小工具 ######################


def _resolve_path(cfg: Dict[str, Any], val: Any) -> Any:
    """解析 '@paths.xxx' 写法。"""
    if isinstance(val, str) and val.startswith("@paths."):
        key = val.split(".", 1)[1]
        if "paths" not in cfg or key not in cfg["paths"]:
            raise KeyError(f"[phys_input] cfg.paths 中不存在键: '{key}'")
        return cfg["paths"][key]
    return val


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _safe_name(s: str) -> str:
    s = str(s).strip()
    return re.sub(r"[^\w\-.]+", "_", s)


def _colname(cols_cfg: Dict[str, Any], key_or_name: str) -> str:
    """
    filters.ranges 里的 key 建议写 columns 的“别名”(如 lp/lf/x/y)，也可直接写 CSV 的列名。
    这里统一解析成真实列名。
    """
    if not isinstance(key_or_name, str):
        return str(key_or_name)
    return cols_cfg.get(key_or_name, key_or_name)


def _parse_num_series(df: pd.DataFrame, col: str) -> pd.Series:
    """将列强制转数值；无法转的为 NaN。"""
    return pd.to_numeric(df[col], errors="coerce")


def _apply_range_filters(
    df: pd.DataFrame,
    filters_cfg: Dict[str, Any],
    cols_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    按 YAML phys_input.filters.ranges 对 df 做数值化 + 区间筛选。

    filters_cfg 支持：
      enable: true/false
      strict: true/false                # 列不存在时：报错 or 警告跳过
      missing_policy: drop/keep         # NaN 行：丢弃 or 保留
      inclusive: true/false             # true: >= <= ; false: > <
      ranges:
        lp: {min: 0.0, max: 1.0}
        lf: {max: 1.0}
        x:  {min: 0.0, max: 1.0}
        ...
    """
    if not filters_cfg or filters_cfg.get("enable", True) is False:
        return df

    strict = bool(filters_cfg.get("strict", False))
    missing_policy = str(filters_cfg.get("missing_policy", "drop")).strip().lower()
    inclusive = bool(filters_cfg.get("inclusive", True))
    ranges = filters_cfg.get("ranges", {}) or {}

    if not isinstance(ranges, dict) or len(ranges) == 0:
        print("[phys_input] filters.enable=True 但 filters.ranges 为空，跳过范围过滤")
        return df

    def _cmp_ge(a, b):  # noqa
        return a >= b if inclusive else a > b

    def _cmp_le(a, b):  # noqa
        return a <= b if inclusive else a < b

    out = df.copy()

    for key, spec in ranges.items():
        if not isinstance(spec, dict):
            spec = {}

        col = _colname(cols_cfg, key)
        if col not in out.columns:
            msg = f"[phys_input] filters.ranges 指定列 '{key}' -> '{col}' 不存在于数据中"
            if strict:
                raise ValueError(msg)
            print(msg + "，已跳过")
            continue

        vmin = spec.get("min", None)
        vmax = spec.get("max", None)

        s = _parse_num_series(out, col)
        before = len(out)

        mask = np.ones(before, dtype=bool)

        if missing_policy == "drop":
            mask &= s.notna().to_numpy()

        if vmin is not None:
            mask &= _cmp_ge(s.to_numpy(), float(vmin))
        if vmax is not None:
            mask &= _cmp_le(s.to_numpy(), float(vmax))

        out = out.loc[mask].copy()
        after = len(out)

        # 写回数值列，避免后面重复转换
        out[col] = _parse_num_series(out, col)

        print(
            f"[phys_input] 范围过滤: {col} "
            f"{'>=' if inclusive else '>'}{vmin if vmin is not None else '-inf'} , "
            f"{'<=' if inclusive else '<'}{vmax if vmax is not None else '+inf'} "
            f" => 保留 {after}/{before} (删除 {before - after})"
        )

        if after == 0:
            return out

    return out


###################### Macdonald + Kanda 公式 ######################

A_MAC = 4.43
BETA_MAC = 1.0
C_LB_MAC = 1.2
KAPPA = 0.4

# Kanda(2013) 的经验参数（保持你之前用的）
KANDA_D_PARAMS = {
    "M1": dict(a0=1.29, b0=0.36, c0=-0.17),
    "M2": dict(a0=0.86, b0=0.28, c0=-0.18),
}
KANDA_Z0_PARAMS = {
    "M1": dict(a1=0.71, b1=20.21, c1=-0.77),
    "M2": dict(a1=0.93, b1=8.93, c1=4.68),
}


def compute_macdonald(
    lp,
    lf,
    have,
    A=A_MAC,
    beta=BETA_MAC,
    C_lb=C_LB_MAC,
    kappa=KAPPA,
):
    """
    Macdonald(1998) 方案计算 d 和 z0。

    返回：d, z0, d_over_H, z0_over_H
    """
    lp = np.asarray(lp, dtype=float)
    lf = np.asarray(lf, dtype=float)
    have = np.asarray(have, dtype=float)

    # d / H = 1 + A^{-lambda_p} (lambda_p - 1)
    d_over_H = 1.0 + np.power(A, -lp) * (lp - 1.0)
    d = d_over_H * have

    # z0 / H = (1 - d/H) * exp( -[0.5 * beta * C_lb/kappa^2 * (1 - d/H) * lambda_f]^{-1/2} )
    term = 0.5 * beta * C_lb / (kappa**2) * (1.0 - d_over_H) * lf
    term = np.where(term <= 0.0, np.nan, term)

    exponent = -np.power(term, -0.5)
    z0_over_H = (1.0 - d_over_H) * np.exp(exponent)
    z0 = z0_over_H * have

    return d, z0, d_over_H, z0_over_H


def compute_kanda_d(lp, have, hstd, hmax, method: str = "M1"):
    """Kanda(2013) 方案计算位移高度 d。返回：d, d_over_Hmax, X"""
    if method not in KANDA_D_PARAMS:
        raise ValueError(f"[phys_input] 未知 Kanda d method: {method}")
    params = KANDA_D_PARAMS[method]
    a0, b0, c0 = params["a0"], params["b0"], params["c0"]

    lp = np.asarray(lp, dtype=float)
    have = np.asarray(have, dtype=float)
    hstd = np.asarray(hstd, dtype=float)
    hmax = np.asarray(hmax, dtype=float)

    # X = (Hstd + Have) / Hmax, 并裁剪到 [0, 1]
    X = np.where(hmax > 0.0, (hstd + have) / hmax, np.nan)
    X = np.clip(X, 0.0, 1.0)

    # d/Hmax 多项式
    d_over_Hmax = c0 * X**2 + (a0 * np.power(lp, b0) - c0) * X
    d = d_over_Hmax * hmax

    return d, d_over_Hmax, X


def compute_kanda_z0(lp, have, hstd, z0_mac, method: str = "M1"):
    """Kanda(2013) 方案计算 z0（相对于 Macdonald 的 z0_mac 做修正）。返回：z0_kanda, ratio, Y"""
    if method not in KANDA_Z0_PARAMS:
        raise ValueError(f"[phys_input] 未知 Kanda z0 method: {method}")
    params = KANDA_Z0_PARAMS[method]
    a1, b1, c1 = params["a1"], params["b1"], params["c1"]

    lp = np.asarray(lp, dtype=float)
    have = np.asarray(have, dtype=float)
    hstd = np.asarray(hstd, dtype=float)
    z0_mac = np.asarray(z0_mac, dtype=float)

    # Y = lambda_p * Hstd / Have
    Y = np.where(have > 0.0, lp * hstd / have, np.nan)
    ratio = b1 * Y**2 + c1 * Y + a1
    z0 = ratio * z0_mac

    return z0, ratio, Y


def apply_macdonald_kanda_to_df(df: pd.DataFrame, kanda_method: str = "M1") -> pd.DataFrame:
    """
    假定 df 中已经有如下列（列名固定）：
      lambda_p, lambda_f, Have(m), Hmax(m), Hstd(m), d_LES(m), z0_LES(m)

    在 df 中新增 / 覆盖：
      d_mac(m), z0_mac(m), d_kanda(m), z0_kanda(m)
      以及 LES / Mac / Kanda 三套无量纲列：
      d_LES_o_Have, z0_LES_o_Have,
      d_mac_o_Have, z0_mac_o_Have,
      d_kanda_o_Have, z0_kanda_o_Have
    """
    df_out = df.copy()

    lp = df_out["lambda_p"].values
    lf = df_out["lambda_f"].values
    have = df_out["Have(m)"].values
    hmax = df_out["Hmax(m)"].values
    hstd = df_out["Hstd(m)"].values

    # ---------- Macdonald ----------
    d_mac, z0_mac, d_mac_over_H, z0_mac_over_H = compute_macdonald(lp, lf, have)
    df_out["d_mac(m)"] = d_mac
    df_out["z0_mac(m)"] = z0_mac
    df_out["d_mac_o_Have"] = d_mac_over_H
    df_out["z0_mac_o_Have"] = z0_mac_over_H

    # ---------- Kanda ----------
    d_kanda, _, _ = compute_kanda_d(lp, have, hstd, hmax, method=kanda_method)
    z0_kanda, _, _ = compute_kanda_z0(lp, have, hstd, z0_mac, method=kanda_method)

    df_out["d_kanda(m)"] = d_kanda
    df_out["z0_kanda(m)"] = z0_kanda

    have_safe = np.where(have > 0.0, have, np.nan)

    # ---------- LES 无量纲 ----------
    if "d_LES(m)" in df_out.columns:
        df_out["d_LES_o_Have"] = df_out["d_LES(m)"].values / have_safe
    else:
        df_out["d_LES_o_Have"] = np.nan

    if "z0_LES(m)" in df_out.columns:
        df_out["z0_LES_o_Have"] = df_out["z0_LES(m)"].values / have_safe
    else:
        df_out["z0_LES_o_Have"] = np.nan

    # ---------- Kanda 无量纲 ----------
    df_out["d_kanda_o_Have"] = d_kanda / have_safe
    df_out["z0_kanda_o_Have"] = z0_kanda / have_safe

    return df_out


###################### 主流程：从 cfg 形成 3 份 CSV ######################


def run_phys_input_from_cfg(cfg: Dict[str, Any]) -> None:
    """
    从 cfg['phys_input'] 读取配置，生成：
      1) r_f_total.csv
      2) rp_profiles.csv
      3) Table1-Kanda_R2.csv
    """
    sec = cfg.get("phys_input")
    if not sec or not sec.get("enable", False):
        print("[phys_input] 未启用，跳过")
        return

    # 路径解析
    in_csv = _resolve_path(cfg, sec["in_csv"])
    out_r_f_csv = _resolve_path(cfg, sec["out_r_f_csv"])
    out_rp_profiles_csv = _resolve_path(cfg, sec["out_rp_profiles_csv"])
    out_table1_csv = _resolve_path(cfg, sec["out_table1_csv"])

    kanda_method = sec.get("kanda_method", "M1")
    fg_mode = str(sec.get("fg_mode", "mixed")).strip().lower()  # ideal / real / mixed

    # 是否自动 strip 列名空格（稳妥：默认开）
    strip_columns = bool(sec.get("strip_columns", True))

    # 兼容旧字段：x_min/x_max/y_min/y_max（如果你已完全用 filters.ranges，可忽略）
    x_min = sec.get("x_min")
    x_max = sec.get("x_max")
    y_min = sec.get("y_min")
    y_max = sec.get("y_max")

    cols_cfg = sec.get("columns", {}) or {}
    col_fg = cols_cfg.get("fg", "fg")
    col_lf = cols_cfg.get("lf", "lf")
    col_lp = cols_cfg.get("lp", "lp")
    col_have = cols_cfg.get("have", "have")
    col_hmax = cols_cfg.get("hmax", "hmax")
    col_hstd = cols_cfg.get("hstd", "hstd")
    col_dles = cols_cfg.get("d_LES", "d")
    col_z0les = cols_cfg.get("z0_LES", "z0")
    col_x = cols_cfg.get("x", "X")
    col_y = cols_cfg.get("y", "Y")
    rp_prefix = cols_cfg.get("rp_prefix", "rp_")

    print(f"[phys_input] 读取源数据: {os.path.abspath(in_csv)}")
    df_raw = pd.read_csv(in_csv)

    if strip_columns:
        df_raw = df_raw.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    # 给原始数据加 ID（按当前顺序）
    df_raw = df_raw.reset_index(drop=True)
    df_raw["ID"] = np.arange(1, len(df_raw) + 1, dtype=int)

    # ==== 统一范围过滤（由 YAML 指定）====
    df = df_raw.copy()
    filters_cfg = sec.get("filters", {}) or {}
    df = _apply_range_filters(df, filters_cfg, cols_cfg)

    print(f"[phys_input] 范围过滤后样本数: {len(df)}")
    if len(df) == 0:
        print("[phys_input] 范围过滤后无样本，直接结束")
        return

    # ==== FG 选择（仍保留 fg_mode 逻辑）====
    if col_fg in df.columns:
        fg_num = pd.to_numeric(df[col_fg], errors="coerce")
        if fg_mode == "ideal":
            df = df.loc[fg_num == 0].copy()
        elif fg_mode == "real":
            df = df.loc[fg_num == 1].copy()
        elif fg_mode == "mixed":
            pass
        else:
            raise ValueError(f"[phys_input] 未知 fg_mode: {fg_mode}（应为 ideal/real/mixed）")
    else:
        print(f"[phys_input] 警告：找不到 FG 列 '{col_fg}'，跳过 FG 筛选")

    # ==== 兼容旧的 X/Y 范围筛选（如果你已用 filters.ranges 管 x/y，可在 YAML 里不写 x_min 等）====
    for name, col_name, vmin, vmax in [
        ("X", col_x, x_min, x_max),
        ("Y", col_y, y_min, y_max),
    ]:
        if (vmin is None) and (vmax is None):
            continue
        if col_name in df.columns:
            s = pd.to_numeric(df[col_name], errors="coerce")
            before = len(df)
            mask = np.ones(before, dtype=bool)
            mask &= s.notna().to_numpy()
            if vmin is not None:
                mask &= (s.to_numpy() >= float(vmin))
            if vmax is not None:
                mask &= (s.to_numpy() <= float(vmax))
            df = df.loc[mask].copy()
            after = len(df)
            print(f"[phys_input] 旧式范围过滤: {name}({col_name}) => 保留 {after}/{before} (删除 {before-after})")
        else:
            print(f"[phys_input] 警告：找不到 {name} 列 '{col_name}'，跳过 {name} 筛选")

    df = df.reset_index(drop=True)
    print(f"[phys_input] 最终筛选后样本数: {len(df)}")
    if len(df) == 0:
        print("[phys_input] 筛选后无样本，直接结束")
        return

    # ==== 核心列有效性检查（保证三份输出行对齐）====
    core_cols = [col_lf, col_lp, col_have, col_hmax, col_hstd]
    missing_core = [c for c in core_cols if c not in df.columns]
    if missing_core:
        raise ValueError(f"[phys_input] 缺少核心列（用于 Mac/Kanda 计算）: {missing_core}")

    s_lf = _parse_num_series(df, col_lf)
    s_lp = _parse_num_series(df, col_lp)
    s_have = _parse_num_series(df, col_have)
    s_hmax = _parse_num_series(df, col_hmax)
    s_hstd = _parse_num_series(df, col_hstd)

    before_core = len(df)
    core_mask = (
        s_lf.notna()
        & s_lp.notna()
        & s_have.notna()
        & s_hmax.notna()
        & s_hstd.notna()
    )
    df = df.loc[core_mask.to_numpy()].copy().reset_index(drop=True)
    after_core = len(df)
    print(f"[phys_input] 核心列有效性：保留 {after_core}/{before_core}（删除 {before_core-after_core}）")
    if len(df) == 0:
        print("[phys_input] 核心列有效性筛选后无样本，结束")
        return

    # 写回数值列（后续直接用）
    df[col_lf] = _parse_num_series(df, col_lf)
    df[col_lp] = _parse_num_series(df, col_lp)
    df[col_have] = _parse_num_series(df, col_have)
    df[col_hmax] = _parse_num_series(df, col_hmax)
    df[col_hstd] = _parse_num_series(df, col_hstd)

    # ==== 1) r_f_total.csv ====
    df_r_f = pd.DataFrame({"r_f": df[col_lf].astype(float)})
    _ensure_dir_for_file(out_r_f_csv)
    df_r_f.to_csv(out_r_f_csv, index=False)
    print(f"[phys_input] 写出 r_f_total: {out_r_f_csv}")

    # ==== 2) rp_profiles.csv ====
    rp_cols: List[Tuple[int, str]] = []
    for c in df.columns:
        if isinstance(c, str) and c.startswith(rp_prefix):
            m = re.match(rf"^{re.escape(rp_prefix)}(\d+)$", c)
            if m:
                rp_cols.append((int(m.group(1)), c))

    if not rp_cols:
        print(f"[phys_input] 警告：未找到以 '{rp_prefix}' 开头且带数字后缀的 rp 列，rp_profiles 不生成")
    else:
        rp_cols_sorted = [c for _, c in sorted(rp_cols, key=lambda x: x[0])]
        df_rp = df[rp_cols_sorted].copy()

        # 重命名为 rp_0000, rp_0001, ...
        new_cols = {}
        for c in rp_cols_sorted:
            num = int(re.match(rf"^{re.escape(rp_prefix)}(\d+)$", c).group(1))
            new_cols[c] = f"rp_{num:04d}"
        df_rp.rename(columns=new_cols, inplace=True)

        _ensure_dir_for_file(out_rp_profiles_csv)
        df_rp.to_csv(out_rp_profiles_csv, index=False)
        print(f"[phys_input] 写出 rp_profiles: {out_rp_profiles_csv}")

    # ==== 3) Table1-Kanda_R2.csv ====
    def _safe_num(colname: str):
        if colname in df.columns:
            return pd.to_numeric(df[colname], errors="coerce")
        return np.nan

    table_df = pd.DataFrame(
        {
            "ID": df["ID"].astype(int),
            "FG": _safe_num(col_fg) if col_fg in df.columns else 0,
            "lambda_f": df[col_lf].astype(float),
            "lambda_p": df[col_lp].astype(float),
            "Have(m)": df[col_have].astype(float),
            "Hmax(m)": df[col_hmax].astype(float),
            "Hstd(m)": df[col_hstd].astype(float),
            "d_LES(m)": _safe_num(col_dles),
            "z0_LES(m)": _safe_num(col_z0les),
        }
    )

    table_df = apply_macdonald_kanda_to_df(table_df, kanda_method=kanda_method)

    cols_order = [
        "ID",
        "FG",
        "lambda_f",
        "lambda_p",
        "Have(m)",
        "Hmax(m)",
        "Hstd(m)",
        "d_LES(m)",
        "z0_LES(m)",
        "d_mac(m)",
        "z0_mac(m)",
        "d_kanda(m)",
        "z0_kanda(m)",
        "d_LES_o_Have",
        "z0_LES_o_Have",
        "d_mac_o_Have",
        "z0_mac_o_Have",
        "d_kanda_o_Have",
        "z0_kanda_o_Have",
    ]
    table_df = table_df[cols_order]

    _ensure_dir_for_file(out_table1_csv)
    table_df.to_csv(out_table1_csv, index=False, encoding="utf-8-sig")
    print(f"[phys_input] 写出 Table1-Kanda_R2: {out_table1_csv}")