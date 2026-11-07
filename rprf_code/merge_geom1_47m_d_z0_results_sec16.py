import pandas as pd
from pathlib import Path
import re

# ===================== 用户填写路径 =====================
geom_path = Path(r"/stu02/shiqc24/d_z0_ML/quanliucheng/rprf_code/output/16eqsec/geom1.txt")
d_path = Path(r"/stu02/shiqc24/LLK_rec/output/47m/martanod_re1/47m_d_results_sec16.csv")
z0_path = Path(r"/stu02/shiqc24/LLK_rec/output/47m/martanod_re1/47m_z0_results_sec16.csv")

mac_path = Path(r"/stu02/shiqc24/LLK_rec/data/dz0_mac_kanda_rur_BJ/dz0_mac.dat")
kanda_path = Path(r"/stu02/shiqc24/LLK_rec/data/dz0_mac_kanda_rur_BJ/dz0_kanda.dat")
rur_path = Path(r"/stu02/shiqc24/LLK_rec/data/dz0_mac_kanda_rur_BJ/dz0_rur-vs-ref.dat")

out_path = Path(r"/stu02/shiqc24/d_z0_ML/quanliucheng/data/BJ_325_47VS120/BJ_model_input.csv")

# ===================== 读取 CSV，自动尝试编码 =====================
def read_csv_auto(path):
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"无法读取 CSV 文件：{path}\n最后一次错误：{last_error}")


# ===================== 通用读取 txt/dat 文件 =====================
def read_table_clean_header(path):
    """
    兼容如下格式：
    >> sec, lambda, eta, h, sigmaH, HHmax <<
    d_mac(m) z0_mac(m)
    d_kanda(m) z0_kanda(m)

    默认第一行为表头，后面为数据。
    支持空格、Tab、英文逗号分隔。
    自动去掉 >>、<<、逗号。
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"文件内容不足，无法读取：{path}")

    # 读取表头
    header_line = lines[0]
    header_line = header_line.replace(">>", " ").replace("<<", " ")

    columns = [
        x.strip().rstrip(",")
        for x in re.split(r"[,\s]+", header_line)
        if x.strip()
    ]

    # 读取数据行
    data = []
    for line in lines[1:]:
        line = line.replace(">>", " ").replace("<<", " ")

        parts = [
            x.strip().rstrip(",")
            for x in re.split(r"[,\s]+", line)
            if x.strip()
        ]

        if len(parts) == 0:
            continue

        if len(parts) != len(columns):
            raise ValueError(
                f"文件某一行列数与表头不一致：{path}\n"
                f"表头列数：{len(columns)}\n"
                f"当前行列数：{len(parts)}\n"
                f"当前行内容：{line}\n"
                f"当前表头：{columns}"
            )

        data.append(parts)

    df = pd.DataFrame(data, columns=columns)

    # 尽量转换为数值
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    return df


# ===================== 1. 读取 geom1.txt =====================
geom = read_table_clean_header(geom_path)

geom_cols_map = {
    "lambda": "lp",
    "eta": "lf",
    "h": "have",
    "sigmaH": "hstd",
    "HHmax": "hmax",
}

geom_keep_cols = ["sec"] + list(geom_cols_map.keys())

missing_geom_cols = [c for c in geom_keep_cols if c not in geom.columns]
if missing_geom_cols:
    raise KeyError(
        f"geom1.txt 缺少这些列：{missing_geom_cols}\n"
        f"当前读取到的列名为：{list(geom.columns)}"
    )

geom_selected = geom[geom_keep_cols].rename(columns=geom_cols_map)


# ===================== 2. 读取 47m_d_results_sec16.csv =====================
d_cols = [
    "d_RE1",
    "d_RE2",
    "d_IT1",
    "d_IT2",
    "d_Median",
]

d_data = read_csv_auto(d_path)

missing_d_cols = [c for c in d_cols if c not in d_data.columns]
if missing_d_cols:
    raise KeyError(
        f"47m_d_results_sec16.csv 缺少这些列：{missing_d_cols}\n"
        f"当前读取到的列名为：{list(d_data.columns)}"
    )

d_selected = d_data[d_cols]


# ===================== 3. 读取 47m_z0_results_sec16.csv =====================
z0_cols = [
    "z0_RE1",
    "z0_RE2",
    "z0_IT1",
    "z0_IT2",
    "z0_Median",
]

z0_data = read_csv_auto(z0_path)

missing_z0_cols = [c for c in z0_cols if c not in z0_data.columns]
if missing_z0_cols:
    raise KeyError(
        f"47m_z0_results_sec16.csv 缺少这些列：{missing_z0_cols}\n"
        f"当前读取到的列名为：{list(z0_data.columns)}"
    )

z0_selected = z0_data[z0_cols]


# ===================== 4. 读取 dz0_mac.dat =====================
mac_data = read_table_clean_header(mac_path)

mac_cols_map = {
    "d_mac(m)": "d_mac",
    "z0_mac(m)": "z0_mac",
}

missing_mac_cols = [c for c in mac_cols_map.keys() if c not in mac_data.columns]
if missing_mac_cols:
    raise KeyError(
        f"dz0_mac.dat 缺少这些列：{missing_mac_cols}\n"
        f"当前读取到的列名为：{list(mac_data.columns)}"
    )

mac_selected = mac_data[list(mac_cols_map.keys())].rename(columns=mac_cols_map)


# ===================== 5. 读取 dz0_kanda.dat =====================
kanda_data = read_table_clean_header(kanda_path)

kanda_cols_map = {
    "d_kanda(m)": "d_kanda",
    "z0_kanda(m)": "z0_kanda",
}

missing_kanda_cols = [c for c in kanda_cols_map.keys() if c not in kanda_data.columns]
if missing_kanda_cols:
    raise KeyError(
        f"dz0_kanda.dat 缺少这些列：{missing_kanda_cols}\n"
        f"当前读取到的列名为：{list(kanda_data.columns)}"
    )

kanda_selected = kanda_data[list(kanda_cols_map.keys())].rename(columns=kanda_cols_map)


# ===================== 6. 读取 dz0_rur-vs-ref.dat =====================
rur_data = read_table_clean_header(rur_path)

rur_cols = [
    "d_rur",
    "z0_rur",
]

missing_rur_cols = [c for c in rur_cols if c not in rur_data.columns]
if missing_rur_cols:
    raise KeyError(
        f"dz0_rur-vs-ref.dat 缺少这些列：{missing_rur_cols}\n"
        f"当前读取到的列名为：{list(rur_data.columns)}"
    )

# 按你的要求：d_rur、z0_mac 不改名，直接写入
rur_selected = rur_data[rur_cols]


# ===================== 7. 检查行数是否一致 =====================
dataframes = {
    "geom1.txt": geom_selected,
    "47m_d_results_sec16.csv": d_selected,
    "47m_z0_results_sec16.csv": z0_selected,
    "dz0_mac.dat": mac_selected,
    "dz0_kanda.dat": kanda_selected,
    "dz0_rur-vs-ref.dat": rur_selected,
}

row_counts = {name: len(df) for name, df in dataframes.items()}

if len(set(row_counts.values())) != 1:
    msg = "各文件行数不一致，不能直接按行横向合并：\n"
    for name, n in row_counts.items():
        msg += f"{name}: {n} 行\n"
    raise ValueError(msg)


# ===================== 8. 横向合并 =====================
merged = pd.concat(
    [
        geom_selected.reset_index(drop=True),
        d_selected.reset_index(drop=True),
        z0_selected.reset_index(drop=True),
        mac_selected.reset_index(drop=True),
        kanda_selected.reset_index(drop=True),
        rur_selected.reset_index(drop=True),
    ],
    axis=1
)


# ===================== 9. 输出 =====================
merged.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"合并完成：{out_path}")
print(f"输出数据形状：{merged.shape}")
print("输出列名：")
print(list(merged.columns))

# 检查重复列名
dup_cols = merged.columns[merged.columns.duplicated()].tolist()
if dup_cols:
    print(f"注意：输出中存在重复列名：{dup_cols}")