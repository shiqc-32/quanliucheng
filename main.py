# -*- coding: utf-8 -*-
# main_rp.py
import argparse
import yaml
import pandas as pd

# 你的已有模块
from rp_stats import process_all_topos                              # 如不需要可注释
from case_name_mapping import build_case_name_mapping
from merge_rp_with_mapping import build_model_input_from_cfg        # 新增：合并模块
from select_and_rename import select_and_rename_from_cfg            # 数据整合与重命名模块
from clean_zscore import clean_with_zscore_from_cfg                 # 新增：数据清洗模块
from dist_checker import run_dist_check_from_cfg                    # 分布检查模块
from phys_schemes import run_mac_kanda_from_cfg                     # 物理方案模块
from d_z0_plot import run_d_z0_plot_from_cfg                        # d_z0 可视化模块
from phys_input_builder import run_phys_input_from_cfg              # 物理方案数据制备模块
import urban_model as um                             # rur方案失效检查

# TabPFN 模块（如需要）
from tabpfn_cv_runner import run_tabpfn_tasks_from_cfg
# RF 模块（如需要）
from rf_runner import run_rf_tasks_from_cfg
# GBRM 模块（如需要）
from gbr_cv_runner import run_gbr_tasks_from_cfg
# SVR 模块（如需要）
from svr_cv_runner import run_svr_tasks_from_cfg
# # Ridge 模块（如需要）
# from ridge_cv_runner import run_ridge_tasks_from_cfg

def main():
    # ── CLI ───────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Run RP pipeline: mapping + merge")
    parser.add_argument(
        "--config",
        default="/stu02/shiqc24/d_z0_ML/quanliucheng/rp_run.yaml",
        help="YAML 配置文件路径（默认使用项目内路径）"
    )
    args = parser.parse_args()

    # ── 读取 YAML ─────────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths  = cfg["paths"]
    compute = cfg.get("compute", {})
    cols   = cfg["columns"]
    params = cfg.get("params", {})

    # # ── rp_ave/std 计算/读取逐层风速（一次性，已经算完）────────────────────────
    # # 生成的文件：d_z0_ML/quanliucheng/output/translation_data/rp_ave_std_1m_ideal.xlsx
    # # 生成的文件：d_z0_ML/quanliucheng/output/translation_data/rp_ave_std_1m_real.xlsx
    # process_all_topos(
    #     root_dir=paths["ideal_root"],           # 或 paths["ideal_root"]
    #     output_excel=paths["rp_1m_ideal_path"],
    #     method=compute.get("method", "step1m"),
    #     step=compute.get("step", 1.0),
    # )
    # process_all_topos(
    #     root_dir=paths["real_root"],            # 或 paths["real_root"]
    #     output_excel=paths["rp_1m_real_path"],
    #     method=compute.get("method", "step1m"),
    #     step=compute.get("step", 1.0),
    # )

    # # ── 案例名称对应：flow ↔ metadata ↔ 文件夹（ideal/real）（一次性，已经算完） ─────────────
    # # 生成的文件：d_z0_ML/quanliucheng/output/translation_data/case_name_mapping.csv
    # df_map, stats = build_case_name_mapping(
    #     flow_csv_path=paths["flow_csv"],
    #     metadata_csv_path=paths["meta_csv"],
    #     flow_case_col=cols["flow_case_col"],
    #     meta_case_col=cols["meta_case_col"],
    #     meta_config_col=cols["meta_config_col"],
    #     idealized_root=paths["ideal_root"],
    #     realistic_root=paths["real_root"],
    #     cutoff=float(params.get("cutoff", 0.72)),
    #     flow_encoding=params.get("flow_encoding"),
    #     meta_encoding=params.get("meta_encoding"),
    #     sep=params.get("sep", ","),
    # )

    # # 打印配对统计
    # print("\n=== 配对统计 ===")
    # for k, v in stats.items():
    #     print(f"{k}: {v}")

    # # 保存映射表（注意编码：优先用 YAML 的 map_encoding；默认 utf-8-sig）
    # map_encoding = params.get("map_encoding", "utf-8-sig")   # 或设置为 "gbk"
    # df_map.to_csv(paths["cnm_path"], index=False, encoding=map_encoding)
    # print("\n已保存映射表到：", paths["cnm_path"])

    # # ── 合并：rp(ID, rp_ave, rp_std) + metadata + flowdata （一次性，已经算完）────────────
    # # 直接使用内存中的 df_map，更稳也避免再次读写
    # # 生成的文件：d_z0_ML/quanliucheng/output/translation_data/model_input_data.csv
    # build_model_input_from_cfg(cfg)
    # print("\n[Done] model_input_data.csv 生成完毕。")

    # # ── 选择与重命名（一次性，已经算完）───────────────────────────────
    # # urban数据
    # select_and_rename_from_cfg(
    #     cfg,
    #     in_csv_path=paths["model_input_csv"],
    #     out_csv_path=paths["urban_model_input"],
    #     task_key="urban",
    # )
    # print(f"[Done] 选择并改名（urban）已输出：{paths['urban_model_input']}")

    # # kanda数据
    # select_and_rename_from_cfg(
    #     cfg,
    #     in_csv_path=paths["kanda_path"],
    #     out_csv_path=paths["kanda_input_csv"],
    #     task_key="kanda",
    # )
    # print(f"[Done] 选择并改名（kanda）已输出：{paths['kanda_input_csv']}")

    # # ── Z-score 清洗 +删除负值（一次性，已经算完）──────────────────────────────────────────────────────────────────────
    # # 1) urban：model_input_csv → urban_zclean_output
    # clean_with_zscore_from_cfg(
    #     cfg,
    #     in_csv_path=paths["urban_model_input"],
    #     out_csv_path=paths["urban_zclean_output"],
    # )

    # # 2) kanda：kanda_path → kanda_zclean_output
    # clean_with_zscore_from_cfg(
    #     cfg,
    #     in_csv_path=paths["kanda_input_csv"],
    #     out_csv_path=paths["kanda_zclean_output"],
    # )

    # print("\n[Done] 两份数据的 Z-score 清洗完成。")

    # # ── 分布检查（一次性）──────────────────────────────────────────────────────────────────────
    # run_dist_check_from_cfg(cfg)

    # # ─ 物理方案计算：Macdonald + Kanda （一次性）────────────────────────────
    # mac_kanda_cfg = cfg["phys"]["mac_kanda"]
    # run_mac_kanda_from_cfg(mac_kanda_cfg)
    # print("\n[Done] Macdonald + Kanda 物理方案计算完成。")

    ####################################################
    # 分箱以后结果变好的证明
    ####################################################

    # # ── TabPFN ───────────────────────────
    # run_tabpfn_tasks_from_cfg(
    #     cfg,
    #     task_keys=["urban_z0_nobin", "urban_z0_bin_uni",
    #                "urban_d_nobin", "urban_d_bin_uni",
    #                "kanda_z0_nobin", "kanda_z0_bin_uni",
    #                "kanda_d_nobin", "kanda_d_bin_uni",
    #                ],
    # )
    # print("[Done] TabPFN 交叉验证完成。")
    # # ──────────────────────────────────────

    # # ── 随机森林 ───────────────────────────
    # run_rf_tasks_from_cfg(
    #     cfg,
    #     task_keys=[
    #         "urban_z0_nobin",
    #         "urban_z0_bin_uni",
    #         "urban_d_nobin",
    #         "urban_d_bin_uni",
    #         "kanda_z0_nobin",
    #         "kanda_z0_bin_uni",
    #         "kanda_d_nobin",
    #         "kanda_d_bin_uni",
    #     ],
    # )
    # print("[Done] 随机森林交叉验证完成。")  
    # # ──────────────────────────────────────

    # # ── GBRM ───────────────────────────
    # run_gbr_tasks_from_cfg(
    #     cfg,
    #     task_keys=[
    #         "urban_z0_nobin",
    #         "urban_z0_bin_uni",
    #         "urban_d_nobin",
    #         "urban_d_bin_uni",
    #         "kanda_z0_nobin",
    #         "kanda_z0_bin_uni",
    #         "kanda_d_nobin",
    #         "kanda_d_bin_uni",
    #     ],
    # )
    # print("[Done] GBRM 交叉验证完成。")
    # # ──────────────────────────────────────

    # # ── SVR ───────────────────────────
    # run_svr_tasks_from_cfg(
    #     cfg,
    #     task_keys=[
    #         "urban_z0_nobin",
    #         "urban_z0_bin_uni",
    #         "urban_d_nobin",
    #         "urban_d_bin_uni",
    #         "kanda_z0_nobin",
    #         "kanda_z0_bin_uni",
    #         "kanda_d_nobin",
    #         "kanda_d_bin_uni",
    #     ],
    # )
    # print("[Done] SVR 交叉验证完成。")
    # # ──────────────────────────────────────

    ####################################################
    # 加入d和不加入d预测的结果差别
    ####################################################

    # # ── TabPFN ───────────────────────────
    # run_tabpfn_tasks_from_cfg(
    #     cfg,
    #     task_keys=[
    #         "kanda_z0_no_d_tabpfn",
    #         "kanda_z0_with_d_tabpfn",
    #         "urban_z0_no_d_tabpfn",
    #         "urban_z0_with_d_tabpfn",
    #     ],
    # )
    # print("[Done] TabPFN 交叉验证完成。")
    # # ──────────────────────────────────────

    # # ── 随机森林 ───────────────────────────
    # run_rf_tasks_from_cfg(
    #     cfg,
    #     task_keys=[
    #         "kanda_z0_no_d_rf",
    #         "kanda_z0_with_d_rf",
    #         "urban_z0_no_d_rf",
    #         "urban_z0_with_d_rf",
    #     ],
    # )
    # print("[Done] 随机森林交叉验证完成。")
    # # ──────────────────────────────────────

    # # ── GBRM ───────────────────────────
    # run_gbr_tasks_from_cfg( 
    #     cfg,
    #     task_keys=[
    #         "kanda_z0_no_d_gbr",
    #         "kanda_z0_with_d_gbr",
    #         "urban_z0_no_d_gbr",
    #         "urban_z0_with_d_gbr",
    #     ],
    # )
    # print("[Done] GBRM 交叉验证完成。")
    # # ──────────────────────────────────────

    # # ── SVR ───────────────────────────
    # run_svr_tasks_from_cfg(
    #     cfg,
    #     task_keys=[
    #         "kanda_z0_no_d_svr",
    #         "kanda_z0_with_d_svr",
    #         "urban_z0_no_d_svr",
    #         "urban_z0_with_d_svr",
    #     ],
    # )
    # print("[Done] SVR 交叉验证完成。")
    # # ──────────────────────────────────────

    # # ───── d_z0 可视化模块 （一次性）────────────────────────────
    # run_d_z0_plot_from_cfg(cfg)
    # print("\n[Done] d_z0 可视化模块完成。")
    # # ────────────────────────────────────────────────

    # ───── 物理方案数据生成 ────────────────────────────
    run_phys_input_from_cfg(cfg)
    # ────────────────────────────────────────────────

    # ───── rur方案失效检查 ────────────────────────────
    # um.run_rur_from_cfg(cfg)        # 对所有站点做 z0 NaN 检查
    # # ────────────────────────────────────────────────

    # # ── TabPFN交换两套数据测试训练 ───────────────────────────
    # run_tabpfn_tasks_from_cfg(
    #     cfg,
    #     task_keys=[
    #         "kanda_to_urban_d",
    #         "kanda_to_urban_z0",
    #         "urban_to_kanda_d",
    #         "urban_to_kanda_z0",
    #         "urban_and_kanda_d",
    #         "urban_and_kanda_z0",
    #     ],
    # )
    # print("[Done] TabPFN 交叉验证完成。")
    # # ──────────────────────────────────────

if __name__ == "__main__":
    main()
