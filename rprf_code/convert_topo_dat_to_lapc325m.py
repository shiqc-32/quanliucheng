# -*- coding: utf-8 -*-
"""
convert_topo_dat_to_lapc325m.py

功能：
把三列格式 topo_BJ_2km.dat：

    x(km)    y(km)    topo(m)

转换成 Fortran 程序可读取的矩阵格式 lapc325m.txt。

Fortran 读取方式对应：

    read(10,*)
    read(10,*)
    read(10,*) ((topo(i,j),i=1,nx),j=ny,1,-1)

因此输出文件前两行是说明行，
之后按照 y 从大到小、x 从小到大的顺序写出二维 topo 矩阵。
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


# ============================================================
# 用户可修改路径
# ============================================================

INPUT_DAT = Path(r"/stu02/shiqc24/d_z0_ML/quanliucheng/data/BJ_325_47VS120/topo_BJ_2km.dat")

OUTPUT_TXT = Path(r"/stu02/shiqc24/d_z0_ML/quanliucheng/rprf_code/output/lapc325m.txt")


# ============================================================
# 读取三列 topo_BJ_2km.dat
# ============================================================

def read_topo_dat(input_file) -> pd.DataFrame:
    """
    读取三列 topo dat 文件。

    兼容有表头或无表头的情况。
    每一行提取前 3 个数字，作为：
        x(km), y(km), topo(m)
    """
    input_file = Path(input_file)

    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_file}")

    float_re = re.compile(
        r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?"
    )

    encodings = ["utf-8-sig", "utf-8", "gbk", "latin1"]
    lines = None
    last_error = None

    for enc in encodings:
        try:
            with open(input_file, "r", encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError as e:
            last_error = e

    if lines is None:
        raise UnicodeDecodeError(
            "unknown",
            b"",
            0,
            1,
            f"无法识别文件编码，最后一次错误：{last_error}",
        )

    records = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # 跳过常见注释行
        if line.startswith(("#", "!", "%", "//")):
            continue

        nums = float_re.findall(line)

        if len(nums) < 3:
            continue

        try:
            x, y, topo = map(float, nums[:3])
        except Exception:
            continue

        records.append((x, y, topo))

    if not records:
        raise ValueError(
            f"没有从文件中解析到有效三列数据：{input_file}"
        )

    df = pd.DataFrame(records, columns=["x", "y", "topo"])
    df = df.dropna(subset=["x", "y", "topo"]).copy()

    return df


# ============================================================
# 转换为 Fortran 矩阵格式
# ============================================================

def convert_to_lapc325m(
    input_dat,
    output_txt,
    expected_nx=201,
    expected_ny=201,
    float_format="{:12.4f}",
):
    """
    把三列 x/y/topo 数据转成 Fortran 可读的 lapc325m.txt。

    参数：
    input_dat:
        输入三列 dat 文件。

    output_txt:
        输出矩阵 txt 文件。

    expected_nx, expected_ny:
        如果不为 None，则检查网格大小是否符合预期。
        你的数据一般是 201 × 201。

    float_format:
        输出数字格式。
    """
    input_dat = Path(input_dat)
    output_txt = Path(output_txt)

    df = read_topo_dat(input_dat)

    x_vals = np.sort(df["x"].unique())
    y_vals = np.sort(df["y"].unique())

    nx = len(x_vals)
    ny = len(y_vals)

    if expected_nx is not None and nx != expected_nx:
        raise ValueError(
            f"x 方向网格数不符合预期：实际 nx={nx}, 预期 nx={expected_nx}"
        )

    if expected_ny is not None and ny != expected_ny:
        raise ValueError(
            f"y 方向网格数不符合预期：实际 ny={ny}, 预期 ny={expected_ny}"
        )

    dx_km = float(np.median(np.diff(x_vals)))
    dy_km = float(np.median(np.diff(y_vals)))

    print(f"[INFO] 读取完成：")
    print(f"       nx = {nx}")
    print(f"       ny = {ny}")
    print(f"       x范围 = {x_vals.min()} ~ {x_vals.max()} km")
    print(f"       y范围 = {y_vals.min()} ~ {y_vals.max()} km")
    print(f"       dx = {dx_km * 1000:.3f} m")
    print(f"       dy = {dy_km * 1000:.3f} m")

    # 构造二维矩阵：
    # 行 index = y，从小到大
    # 列 columns = x，从小到大
    topo_pivot = df.pivot_table(
        index="y",
        columns="x",
        values="topo",
        aggfunc="mean",
    )

    topo_pivot = topo_pivot.reindex(index=y_vals, columns=x_vals)

    if topo_pivot.isna().any().any():
        nan_count = int(topo_pivot.isna().sum().sum())
        raise ValueError(
            f"网格中存在缺失值，共 {nan_count} 个。"
            f"请检查 x/y 是否构成完整规则网格。"
        )

    topo_grid = topo_pivot.to_numpy(dtype=float)

    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(output_txt, "w", encoding="utf-8") as f:
        # 前两行会被 Fortran 的 read(10,*) 跳过
        f.write("Converted from topo_BJ_2km.dat\n")
        f.write(f"nx={nx}, ny={ny}, x(km) ascending, y(km) descending\n")

        # Fortran 读取：
        # read(10,*) ((topo(i,j),i=1,nx),j=ny,1,-1)
        #
        # 所以这里必须按：
        # y 从大到小；
        # 每一行 x 从小到大。
        for y in y_vals[::-1]:
            row = topo_pivot.loc[y, x_vals].to_numpy(dtype=float)
            line = " ".join(float_format.format(v) for v in row)
            f.write(line + "\n")

    print(f"[DONE] 已输出 Fortran 矩阵文件：{output_txt}")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    convert_to_lapc325m(
        input_dat=INPUT_DAT,
        output_txt=OUTPUT_TXT,
        expected_nx=201,
        expected_ny=201,
    )