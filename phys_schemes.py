# -*- coding: utf-8 -*-
"""
物理方案计算脚本：Macdonald(1998) + Kanda(2013)

适配你的表头：
ID	FG	lambda_f	lambda_p	Have(m)	Hmax(m)	Hstd(m)	d_LES(m)	z0_LES(m)	d_mac(m)	z0_mac(m)	d_kanda(m)	z0_kanda(m)	...

读取列：
  lambda_p (第 4 列) -> λ_p
  lambda_f (第 3 列) -> λ_f
  Have(m)  (第 5 列) -> H_ave
  Hmax(m)  (第 6 列) -> H_max
  Hstd(m)  (第 7 列) -> σ_H

写入列（覆盖原有内容）：
  d_mac(m)      : Macdonald 计算得到的 d
  z0_mac(m)     : Macdonald 计算得到的 z0
  d_kanda(m)    : Kanda(2013) Method 1/2 计算得到的 d
  z0_kanda(m)   : Kanda(2013) Method 1/2 计算得到的 z0
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# ========== Macdonald(1998) 常数 ==========
A_MAC = 4.43
BETA_MAC = 1.0
C_LB_MAC = 1.2
KAPPA = 0.4

# ========== Kanda(2013) 常数（Method 1 / Method 2） ==========
KANDA_D_PARAMS = {
    "M1": dict(a0=1.29, b0=0.36, c0=-0.17),
    "M2": dict(a0=0.86, b0=0.28, c0=-0.18),
}
KANDA_Z0_PARAMS = {
    "M1": dict(a1=0.71, b1=20.21, c1=-0.77),
    "M2": dict(a1=0.93, b1=8.93, c1=4.68),
}


# ========== Macdonald 方案：d 和 z0 ==========
def compute_macdonald(
    lp, lf, have,
    A=A_MAC, beta=BETA_MAC, C_lb=C_LB_MAC, kappa=KAPPA,
):
    """
    Macdonald et al. (1998b) 式 (8)(9)：

      d / H_ave = 1 + A^{-λ_p} (λ_p - 1)          (8)

      z0(mac) / H_ave = (1 - d/H_ave)
                        * exp( -{0.5 β C_lb / κ^2 * (1 - d/H_ave) λ_f}^{-0.5} )   (9)
    """
    lp = np.asarray(lp, dtype=float)
    lf = np.asarray(lf, dtype=float)
    have = np.asarray(have, dtype=float)

    # 式 (8)
    d_over_H = 1.0 + np.power(A, -lp) * (lp - 1.0)
    d = d_over_H * have

    # 式 (9)
    term = 0.5 * beta * C_lb / (kappa ** 2) * (1.0 - d_over_H) * lf
    term = np.where(term <= 0.0, np.nan, term)   # 避免 term<=0 导致开方错误

    exponent = -np.power(term, -0.5)   # - term^{-1/2}
    z0_over_H = (1.0 - d_over_H) * np.exp(exponent)
    z0 = z0_over_H * have

    return d, z0, d_over_H, z0_over_H


# ========== Kanda 方案：d ==========
def compute_kanda_d(
    lp, have, hstd, hmax,
    method: str = "M1",
):
    """
    Kanda(2013)：

      d / H_max = c0 X^2 + (a0 λ_p^{b0} - c0) X
      X = (σ_H + H_ave) / H_max,   0 <= X <= 1
    """
    params = KANDA_D_PARAMS[method]
    a0, b0, c0 = params["a0"], params["b0"], params["c0"]

    lp = np.asarray(lp, dtype=float)
    have = np.asarray(have, dtype=float)
    hstd = np.asarray(hstd, dtype=float)
    hmax = np.asarray(hmax, dtype=float)

    # X = (σ_H + H_ave) / H_max
    X = np.where(hmax > 0.0, (hstd + have) / hmax, np.nan)
    X = np.clip(X, 0.0, 1.0)

    d_over_Hmax = c0 * X ** 2 + (a0 * np.power(lp, b0) - c0) * X
    d = d_over_Hmax * hmax

    return d, d_over_Hmax, X


# ========== Kanda 方案：z0 ==========
def compute_kanda_z0(
    lp, have, hstd, z0_mac,
    method: str = "M1",
):
    """
    Kanda(2013) 式 (12)：

      z0 / z0(mac) = b1 Y^2 + c1 Y + a1       (12a)
      Y = λ_p σ_H / H_ave                     (12b)
    """
    params = KANDA_Z0_PARAMS[method]
    a1, b1, c1 = params["a1"], params["b1"], params["c1"]

    lp = np.asarray(lp, dtype=float)
    have = np.asarray(have, dtype=float)
    hstd = np.asarray(hstd, dtype=float)
    z0_mac = np.asarray(z0_mac, dtype=float)

    Y = np.where(have > 0.0, lp * hstd / have, np.nan)
    ratio = b1 * Y ** 2 + c1 * Y + a1   # z0 / z0(mac)
    z0 = ratio * z0_mac

    return z0, ratio, Y


# ========== 在 DataFrame 上应用两个方案 ==========
def apply_macdonald_kanda_to_df(df: pd.DataFrame,
                                kanda_method: str = "M1") -> pd.DataFrame:
    """
    针对你的表头做的固定映射：

      λ_p   <- 'lambda_p'
      λ_f   <- 'lambda_f'
      H_ave <- 'Have(m)'
      H_max <- 'Hmax(m)'
      σ_H   <- 'Hstd(m)'

    结果写入（覆盖原有内容）：
      'd_mac(m)', 'z0_mac(m)', 'd_kanda(m)', 'z0_kanda(m)'
    """
    df_out = df.copy()

    lp   = df_out["lambda_p"].values
    lf   = df_out["lambda_f"].values
    have = df_out["Have(m)"].values
    hmax = df_out["Hmax(m)"].values
    hstd = df_out["Hstd(m)"].values

    # ---- Macdonald ----
    d_mac, z0_mac, dH_mac, z0H_mac = compute_macdonald(lp, lf, have)
    df_out["d_mac(m)"]  = d_mac
    df_out["z0_mac(m)"] = z0_mac

    # ---- Kanda (Method 1 / 2) ----
    d_kanda, dHmax_k, X = compute_kanda_d(lp, have, hstd, hmax, method=kanda_method)
    z0_kanda, ratio_k, Y = compute_kanda_z0(lp, have, hstd, z0_mac, method=kanda_method)

    df_out["d_kanda(m)"]  = d_kanda
    df_out["z0_kanda(m)"] = z0_kanda

    return df_out


# ========== 给 main 调用的入口：从 cfg 读取 ==========
def run_mac_kanda_from_cfg(cfg_section: dict):
    """
    cfg_section 预期包含：
      {
        "csv": "/path/to/Table1-Urban.csv",
        "kanda_method": "M1"  # 可选，缺省 M1
      }
    """
    csv_path = Path(cfg_section["csv"])
    kanda_method = cfg_section.get("kanda_method", "M1")

    df = pd.read_csv(csv_path)
    df_out = apply_macdonald_kanda_to_df(df, kanda_method=kanda_method)
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Macdonald + Kanda({kanda_method}) 结果已写回: {csv_path}")


# ========== 命令行入口（可选，方便单独跑） ==========
def main():
    parser = argparse.ArgumentParser(
        description="根据 Macdonald(1998) 和 Kanda(2013) 计算 d 与 z0，结果写回同一个 CSV（覆盖）。"
    )
    parser.add_argument("--csv", required=True, help="输入/输出 CSV 路径（原地覆盖）")
    parser.add_argument(
        "--kanda_method",
        default="M1",
        choices=["M1", "M2"],
        help="Kanda 回归常数方案（Method 1 或 Method 2）",
    )
    args = parser.parse_args()

    cfg_section = {
        "csv": args.csv,
        "kanda_method": args.kanda_method,
    }
    run_mac_kanda_from_cfg(cfg_section)


if __name__ == "__main__":
    main()
