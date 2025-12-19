# -*- coding: utf-8 -*-
# merge_rp_with_mapping.py
import os
import re
from typing import Dict, Any, List
import pandas as pd

def _require_cols(df: pd.DataFrame, cols: List[str], tag: str):
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"[{tag}] 缺少列：{c}")

def _read_rp_block(xlsx_path: str, sheet=None) -> pd.DataFrame:
    """
    读取1m步长rp结果，提取：
      - ID
      - rp_ave / rp_std
      - rp_0, rp_1, ...  等 rp_* 列
      - U_0, U_1, ...     等 U_* 列
    """
    df = pd.read_excel(xlsx_path, sheet_name=0)
    _require_cols(df, ["ID", "rp_ave", "rp_std"], f"rp({os.path.basename(xlsx_path)})")

    # rp_* 系列列（如 rp_0, rp_1, ...）
    rp_series_cols = [c for c in df.columns if re.match(r"^rp_\d+", str(c))]

    # U_* 系列列（如 U_0, U_1, ...）
    U_series_cols = [c for c in df.columns if re.match(r"^U_\d+", str(c))]

    # 最终保留列：ID + rp_ave/rp_std + rp_* + U_*
    keep = ["ID", "rp_ave", "rp_std"] \
           + [c for c in rp_series_cols if c not in ("rp_ave", "rp_std")] \
           + U_series_cols

    # 过滤掉不在 df 中的（防止某些列本来就不存在）
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
        sep, flow_encoding, meta_encoding, map_encoding（可选）,
        encoding   # 统一“输出”编码
        rp_ideal_sheet（可选）, rp_real_sheet（可选）
      columns:
        xy_col:  metadata 中 “320m x 192m” 这一列的列名
    """
    paths  = cfg["paths"]
    cols   = cfg.get("columns", {})    # ⭐ 改成 get，防止没配时报错
    params = cfg.get("params", {})

    sep            = params.get("sep", ",")
    flow_enc       = params.get("flow_encoding")
    meta_enc       = params.get("meta_encoding")
    map_enc        = params.get("map_encoding")
    out_encoding   = params.get("encoding", "utf-8-sig")  # 统一输出编码
    rp_ideal_sheet = params.get("rp_ideal_sheet")
    rp_real_sheet  = params.get("rp_real_sheet")

    # 1) 读取 metadata、flowdata、对照表
    df_meta = pd.read_csv(paths["meta_csv"], sep=sep, encoding=meta_enc)
    df_flow = pd.read_csv(paths["flow_csv"], sep=sep, encoding=flow_enc)
    df_cnm  = pd.read_csv(paths["cnm_path"],  sep=sep, encoding=map_enc)

    _require_cols(df_meta, ["NameE"], "metadata")
    _require_cols(df_flow, ["NameE"], "flowdata")
    _require_cols(df_cnm,  ["flow_name", "meta_name", "folder_name"], "cnm")

    # ⭐⭐ 1.1 解析 metadata 中的 x-y 列，例如 "320m x 192m" -> X=320, Y=192
    xy_col = cols.get("xy_col")    # 例如 "XY"
    if xy_col is not None and xy_col in df_meta.columns:
        def _parse_xy(val):
            """
            支持格式大致为:
              "320m x 192m"
              "320 x 192"
              "320 m×192 m" 等
            解析出两个数值 (X, Y)，否则返回 (NA, NA)
            """
            if pd.isna(val):
                return pd.Series({"X": pd.NA, "Y": pd.NA})
            s = str(val)
            # 匹配: 数字 [可带小数] + 可选 m + x/× + 数字 [可带小数] + 可选 m
            m = re.search(r'(\d+(?:\.\d+)?)\s*m?\s*[x×]\s*(\d+(?:\.\d+)?)\s*m?', s)
            if m:
                try:
                    x_val = float(m.group(1))
                    y_val = float(m.group(2))
                    return pd.Series({"X": x_val, "Y": y_val})
                except ValueError:
                    return pd.Series({"X": pd.NA, "Y": pd.NA})
            else:
                return pd.Series({"X": pd.NA, "Y": pd.NA})

        # 在 df_meta 中新增两列 X、Y
        xy_df = df_meta[xy_col].apply(_parse_xy)
        df_meta["X"] = xy_df["X"]
        df_meta["Y"] = xy_df["Y"]
        print(f"[model_input] 从 metadata.{xy_col} 解析得到 X/Y 列")

    # 2) 读取两份 rp 1m 结果（ideal + real），合并
    df_rp_ideal = _read_rp_block(paths["rp_1m_ideal_path"], sheet=rp_ideal_sheet)
    df_rp_real  = _read_rp_block(paths["rp_1m_real_path"],  sheet=rp_real_sheet)

    df_rp = pd.concat([df_rp_ideal, df_rp_real], ignore_index=True)
    # 如果 ID 有重复且你希望去重，可启用：
    # df_rp = df_rp.drop_duplicates(subset=["ID"], keep="first")

    # 3) 基于对照表对齐三块数据
    # 3.1 rp(ID) ↔ folder_name
    model_df = pd.merge(df_cnm, df_rp, left_on="folder_name", right_on="ID", how="left")

    # 3.2 metadata(NameE) ↔ meta_name
    # 为避免列名冲突，给 metadata / flowdata 全量列加前缀
    meta_key = "NameE"
    flow_key = "NameE"

    df_meta_pref = df_meta.copy()
    df_meta_pref.columns = [f"meta__{c}" if c != meta_key else c for c in df_meta_pref.columns]
    model_df = pd.merge(model_df, df_meta_pref, left_on="meta_name", right_on=meta_key, how="left")

    # 3.3 flowdata(NameE) ↔ flow_name
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

    # 简要日志
    total = len(model_df)
    hit_rp = model_df["ID"].notna().sum() if "ID" in model_df.columns else 0
    print(f"[model_input] 写出：{out_path}")
    print(f"  总行数: {total}, rp 匹配成功: {hit_rp}, 未匹配: {total - hit_rp}")
    return out_path
