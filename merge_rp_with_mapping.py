# -*- coding: utf-8 -*-
# merge_rp_with_mapping.py

import os
import re
from typing import Dict, Any, List, Optional
import pandas as pd


def _require_cols(df: pd.DataFrame, cols: List[str], tag: str):
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"[{tag}] 缺少列：{c}")


def _read_rp_block(csv_path: str,
                   sheet=None,
                   sep: str = ",",
                   encoding: Optional[str] = None) -> pd.DataFrame:
    """
    读取 1m 步长 rp 结果（CSV 版本），提取：
      - ID
      - rp_ave / rp_std
      - rp_0, rp_1, ...  等 rp_* 列
      - U_0, U_1, ...     等 U_* 列（如果存在）

    注意：
      - CSV 不存在 sheet 概念，sheet 参数仅为兼容旧调用，直接忽略。
      - 建议确保 CSV 表头中 ID / rp_ave / rp_std 存在。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[rp] CSV 不存在：{csv_path}")

    # 尝试多编码，避免 Excel/Windows 导出编码导致的读取失败
    encodings = []
    if encoding:
        encodings.append(encoding)
    encodings += ["utf-8-sig", "utf-8", "gbk"]

    last_err = None
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, sep=sep, encoding=enc)
            last_err = None
            break
        except Exception as e:
            last_err = e

    if df is None:
        raise RuntimeError(f"[rp] 读取 CSV 失败：{csv_path}；最后一次错误：{last_err}")

    _require_cols(df, ["ID", "rp_ave", "rp_std"], f"rp({os.path.basename(csv_path)})")

    # rp_* 系列列（如 rp_0, rp_1, ...；注意排除 rp_ave/rp_std）
    rp_series_cols = [c for c in df.columns if re.match(r"^rp_\d+$", str(c))]

    # U_* 系列列（如 U_0, U_1, ...）
    U_series_cols = [c for c in df.columns if re.match(r"^U_\d+$", str(c))]

    keep = ["ID", "rp_ave", "rp_std"] + rp_series_cols + U_series_cols
    keep = [c for c in keep if c in df.columns]

    return df[keep].copy()


def build_model_input_from_cfg(cfg: Dict[str, Any]) -> str:
    """
    目标：输出 1 个文件 model_input_data.csv
    合并三块数据：①metadata 全量；②flowdata 全量；③ rp(ideal+real 两份1m结果)
    依据对照表(cnm_path)中 flow_name / meta_name / folder_name 的对应关系进行拼接：
      - metadata.NameE  ↔ cnm.meta_name
      - flowdata.NameE  ↔ cnm.flow_name
      - rp.ID           ↔ cnm.folder_name

    YAML 需提供：
      paths:
        meta_csv, flow_csv, rp_1m_ideal_path, rp_1m_real_path, cnm_path, model_input_csv
      params:
        sep,
        flow_encoding, meta_encoding, map_encoding（可选）,
        rp_encoding（可选，用于 rp CSV）,
        encoding   # 统一“输出”编码
        rp_ideal_sheet（可选，已忽略）, rp_real_sheet（可选，已忽略）
      columns:
        xy_col:  metadata 中 “320m x 192m” 这一列的列名
    """
    paths = cfg["paths"]
    cols = cfg.get("columns", {})
    params = cfg.get("params", {})

    sep = params.get("sep", ",")
    flow_enc = params.get("flow_encoding")
    meta_enc = params.get("meta_encoding")
    map_enc = params.get("map_encoding")
    rp_enc = params.get("rp_encoding")  # 新增：rp CSV 编码（可选）
    out_encoding = params.get("encoding", "utf-8-sig")

    # 兼容保留（CSV 不用 sheet，这两个值不会再参与逻辑）
    rp_ideal_sheet = params.get("rp_ideal_sheet")
    rp_real_sheet = params.get("rp_real_sheet")

    # 1) 读取 metadata、flowdata、对照表
    df_meta = pd.read_csv(paths["meta_csv"], sep=sep, encoding=meta_enc)
    df_flow = pd.read_csv(paths["flow_csv"], sep=sep, encoding=flow_enc)
    df_cnm = pd.read_csv(paths["cnm_path"], sep=sep, encoding=map_enc)

    _require_cols(df_meta, ["NameE"], "metadata")
    _require_cols(df_flow, ["NameE"], "flowdata")
    _require_cols(df_cnm, ["flow_name", "meta_name", "folder_name"], "cnm")

    # 1.1 解析 metadata 中的 x-y 列，例如 "320m x 192m" -> X=320, Y=192
    xy_col = cols.get("xy_col")
    if xy_col is not None and xy_col in df_meta.columns:
        def _parse_xy(val):
            if pd.isna(val):
                return pd.Series({"X": pd.NA, "Y": pd.NA})
            s = str(val)
            m = re.search(r'(\d+(?:\.\d+)?)\s*m?\s*[x×]\s*(\d+(?:\.\d+)?)\s*m?', s)
            if m:
                try:
                    return pd.Series({"X": float(m.group(1)), "Y": float(m.group(2))})
                except ValueError:
                    return pd.Series({"X": pd.NA, "Y": pd.NA})
            return pd.Series({"X": pd.NA, "Y": pd.NA})

        xy_df = df_meta[xy_col].apply(_parse_xy)
        df_meta["X"] = xy_df["X"]
        df_meta["Y"] = xy_df["Y"]
        print(f"[model_input] 从 metadata.{xy_col} 解析得到 X/Y 列")

    # 2) 读取两份 rp 1m 结果（ideal + real），合并（现在是 CSV）
    df_rp_ideal = _read_rp_block(
        paths["ideal_rp_csv"],
        sheet=rp_ideal_sheet,  # 兼容参数，忽略
        sep=sep,
        encoding=rp_enc
    )
    df_rp_real = _read_rp_block(
        paths["real_rp_csv"],
        sheet=rp_real_sheet,   # 兼容参数，忽略
        sep=sep,
        encoding=rp_enc
    )

    df_rp = pd.concat([df_rp_ideal, df_rp_real], ignore_index=True)
    # 如需按 ID 去重可启用：
    # df_rp = df_rp.drop_duplicates(subset=["ID"], keep="first")

    # 3) 基于对照表对齐三块数据
    # 3.1 rp(ID) ↔ folder_name
    model_df = pd.merge(df_cnm, df_rp, left_on="folder_name", right_on="ID", how="left")

    # 3.2 metadata(NameE) ↔ meta_name
    meta_key = "NameE"
    df_meta_pref = df_meta.copy()
    df_meta_pref.columns = [f"meta__{c}" if c != meta_key else c for c in df_meta_pref.columns]
    model_df = pd.merge(model_df, df_meta_pref, left_on="meta_name", right_on=meta_key, how="left")

    # 3.3 flowdata(NameE) ↔ flow_name
    flow_key = "NameE"
    df_flow_pref = df_flow.copy()
    df_flow_pref.columns = [f"flow__{c}" if c != flow_key else c for c in df_flow_pref.columns]
    model_df = pd.merge(model_df, df_flow_pref, left_on="flow_name", right_on=flow_key, how="left")

    # 4) 列顺序（把核心键放前面；其余保持）
    core = ["flow_name", "meta_name", "folder_name", "ID", "rp_ave", "rp_std"]
    core = [c for c in core if c in model_df.columns]
    out_cols = core + [c for c in model_df.columns if c not in core]

    # 5) 输出：model_input_data.csv
    out_path = paths.get("model_input_csv") or os.path.join(os.getcwd(), "model_input_data.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model_df[out_cols].to_csv(out_path, index=False, encoding=out_encoding)

    # 简要日志：rp 匹配成功统计（以 rp_ave 非空判断更合理）
    total = len(model_df)
    hit_rp = model_df["rp_ave"].notna().sum() if "rp_ave" in model_df.columns else 0
    print(f"[model_input] 写出：{out_path}")
    print(f"  总行数: {total}, rp 匹配成功: {hit_rp}, 未匹配: {total - hit_rp}")
    return out_path