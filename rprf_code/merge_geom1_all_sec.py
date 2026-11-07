import pandas as pd
from pathlib import Path
import re

# ===================== 用户配置 =====================
height = 47
sec_list = [1, 2, 4, 8, 16]

# geom1.txt 所在根目录
# 自动读取：
# /stu02/.../output/1eqsec/geom1.txt
# /stu02/.../output/2eqsec/geom1.txt
# ...
geom_root = Path(r"/stu02/shiqc24/d_z0_ML/quanliucheng/rprf_code/output")

# d 和 z0 结果所在目录
dz0_result_dir = Path(r"/stu02/shiqc24/LLK_rec/output/47m/martanod_re1")

# 总输出文件
out_path = Path(
    r"/stu02/shiqc24/d_z0_ML/quanliucheng/data/BJ_325_47VS120/BJ_model_input_all_sec.csv"
)


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


# ===================== 通用读取 txt 文件 =====================
def read_table_clean_header(path):
    """
    兼容如下 geom1.txt 表头格式：
    >> sec, lambda, eta, hlam, heta, h, sigmaH, HHmax, karm, eta_ratio <<

    默认第一行为表头，后面为数据。
    支持空格、Tab、英文逗号分隔。
    自动去掉 >>、<<、逗号。
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"文件内容不足，无法读取：{path}")

    # ---------- 读取表头 ----------
    header_line = lines[0]
    header_line = header_line.replace(">>", " ").replace("<<", " ")

    columns = [
        x.strip().rstrip(",")
        for x in re.split(r"[,\s]+", header_line)
        if x.strip()
    ]

    # ---------- 读取数据 ----------
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
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


# ===================== 检查列是否存在 =====================
def check_missing_cols(df, required_cols, file_name):
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise KeyError(
            f"{file_name} 缺少这些列：{missing_cols}\n"
            f"当前读取到的列名为：{list(df.columns)}"
        )


# ===================== 单个 sec 合并函数 =====================
def build_one_sec(sec):
    print(f"\n正在处理 sec={sec} ...")

    # ---------- 1. 路径 ----------
    geom_path = geom_root / f"{sec}eqsec" / "geom1.txt"
    d_path = dz0_result_dir / f"{height}m_d_results_sec{sec}.csv"
    z0_path = dz0_result_dir / f"{height}m_z0_results_sec{sec}.csv"

    if not geom_path.exists():
        raise FileNotFoundError(f"找不到 geom1.txt：{geom_path}")

    if not d_path.exists():
        raise FileNotFoundError(f"找不到 d 结果文件：{d_path}")

    if not z0_path.exists():
        raise FileNotFoundError(f"找不到 z0 结果文件：{z0_path}")

    # ---------- 2. 读取 geom1.txt ----------
    geom = read_table_clean_header(geom_path)

    geom_cols_map = {
        "lambda": "lf",
        "eta": "lp",
        "h": "have",
        "sigmaH": "hstd",
        "HHmax": "hmax",
    }

    geom_keep_cols = ["sec"] + list(geom_cols_map.keys())

    check_missing_cols(geom, geom_keep_cols, f"{sec}eqsec/geom1.txt")

    geom_selected = geom[geom_keep_cols].rename(columns=geom_cols_map)

    # ---------- 3. 读取 d 结果 ----------
    d_cols = [
        "d_RE1",
        "d_RE2",
        "d_IT1",
        "d_IT2",
        "d_Median",
    ]

    d_data = read_csv_auto(d_path)
    check_missing_cols(d_data, d_cols, f"{height}m_d_results_sec{sec}.csv")

    d_selected = d_data[d_cols]

    # ---------- 4. 读取 z0 结果 ----------
    z0_cols = [
        "z0_RE1",
        "z0_RE2",
        "z0_IT1",
        "z0_IT2",
        "z0_Median",
    ]

    z0_data = read_csv_auto(z0_path)
    check_missing_cols(z0_data, z0_cols, f"{height}m_z0_results_sec{sec}.csv")

    z0_selected = z0_data[z0_cols]

    # ---------- 5. 检查行数 ----------
    dataframes = {
        f"{sec}eqsec/geom1.txt": geom_selected,
        f"{height}m_d_results_sec{sec}.csv": d_selected,
        f"{height}m_z0_results_sec{sec}.csv": z0_selected,
    }

    row_counts = {name: len(df) for name, df in dataframes.items()}

    if len(set(row_counts.values())) != 1:
        msg = f"sec={sec} 各文件行数不一致，不能直接按行横向合并：\n"
        for name, n in row_counts.items():
            msg += f"{name}: {n} 行\n"
        raise ValueError(msg)

    # ---------- 6. 横向合并 ----------
    merged_one = pd.concat(
        [
            geom_selected.reset_index(drop=True),
            d_selected.reset_index(drop=True),
            z0_selected.reset_index(drop=True),
        ],
        axis=1,
    )

    # 标记该行来自几分区方案
    merged_one.insert(0, "n_sector", sec)

    print(f"sec={sec} 合并完成，数据形状：{merged_one.shape}")

    return merged_one


# ===================== 汇总所有 sec =====================
all_merged = []

for sec in sec_list:
    merged_one = build_one_sec(sec)
    all_merged.append(merged_one)

merged_all = pd.concat(all_merged, axis=0, ignore_index=True)


# ===================== 输出 =====================
out_path.parent.mkdir(parents=True, exist_ok=True)
merged_all.to_csv(out_path, index=False, encoding="utf-8-sig")

print("\n全部合并完成")
print(f"输出文件：{out_path}")
print(f"输出数据形状：{merged_all.shape}")
print("输出列名：")
print(list(merged_all.columns))

print("\n每个 n_sector 的行数：")
print(merged_all["n_sector"].value_counts().sort_index())