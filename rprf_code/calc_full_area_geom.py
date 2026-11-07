from pathlib import Path
from collections import deque
import math
import numpy as np


# ===================== 用户配置区 =====================
INPUT_FILE = Path(
    "/stu02/shiqc24/d_z0_ML/quanliucheng/rprf_code/output/lapc325m.txt"
)

OUTPUT_FILE = Path(
    "/stu02/shiqc24/d_z0_ML/quanliucheng/rprf_code/output/1eqsec/geom1.txt"
)

# 原 Fortran 中的网格尺寸
NX = 201
NY = 201

# 原 Fortran 中 dx=10, dy=10, 单位 m
DX = 10.0
DY = 10.0

# 原 lapc325m.txt 前两行是表头，所以跳过 2 行
SKIP_ROWS = 2

# 建筑判定阈值：高度 > 0 认为是建筑
BUILDING_THRESHOLD = 0.0

# lambda 是方向相关参数。
# 如果想模拟 Fortran 中 isec=1 的全域单扇区中心风向，使用 180°。
# 如果你想算东西向迎风，可用 0 或 180；
# 如果想算南北向迎风，可用 90 或 270。
WIND_DIRECTION_DEG = 180.0

# 是否按原 Fortran 的 3 m 高度层分组识别建筑块
# True：更接近原 1.blockcheck_sec.f90 + 2.prof_dens_sec.f90 的逻辑
# False：直接对所有高度 > 0 的格点做连通块识别
USE_HEIGHT_BINS = True

# 原 Fortran 高度层：75, 72, ..., 3, 0
HEIGHT_LEVELS = list(range(75, -1, -3))
# =====================================================


def read_topo_matrix(path: Path, nx: int, ny: int, skip_rows: int) -> np.ndarray:
    """
    读取 lapc325m.txt。
    原 Fortran 逻辑是跳过前两行，然后读取 201×201 高度矩阵。
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    topo = np.loadtxt(path, skiprows=skip_rows)

    if topo.shape != (ny, nx):
        raise ValueError(
            f"Unexpected matrix shape: {topo.shape}. "
            f"Expected ({ny}, {nx}). Please check NX, NY, SKIP_ROWS."
        )

    # numpy 读出来是 [row, col]，这里统一为 [j, i]
    return topo.astype(float)


def connected_components(mask: np.ndarray):
    """
    8 邻域连通块识别，对应 Fortran block_check 中的 8-neighbor 逻辑。
    mask: True 表示建筑格点。
    返回每个连通块的格点坐标列表。
    """
    ny, nx = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    components = []

    for j in range(ny):
        for i in range(nx):
            if not mask[j, i] or visited[j, i]:
                continue

            q = deque()
            q.append((j, i))
            visited[j, i] = True
            comp = []

            while q:
                cj, ci = q.popleft()
                comp.append((cj, ci))

                for dj, di in neighbors:
                    nj, ni = cj + dj, ci + di
                    if 0 <= nj < ny and 0 <= ni < nx:
                        if mask[nj, ni] and not visited[nj, ni]:
                            visited[nj, ni] = True
                            q.append((nj, ni))

            components.append(comp)

    return components


def get_building_blocks(topo: np.ndarray, use_height_bins: bool = True):
    """
    识别建筑块，并计算每个建筑块的：
    - 平均高度 hb
    - 占据格点数 eta1
    - x 方向包络宽度 width_grid
    - y 方向包络宽度 length_grid

    use_height_bins=True 时，按 Fortran 的 3 m 高度层分别识别连通块。
    """
    blocks = []

    if use_height_bins:
        # 按 Fortran 高度层：75~72, 72~69, ..., 3~0
        for upper, lower in zip(HEIGHT_LEVELS[:-1], HEIGHT_LEVELS[1:]):
            mask = (topo <= upper) & (topo > lower)

            comps = connected_components(mask)

            for comp in comps:
                js = np.array([p[0] for p in comp])
                is_ = np.array([p[1] for p in comp])
                values = topo[js, is_]

                hb = float(values.mean())
                eta1 = int(len(values))

                width_grid = int(is_.max() - is_.min() + 1)
                length_grid = int(js.max() - js.min() + 1)

                blocks.append(
                    {
                        "hb": hb,
                        "eta1": eta1,
                        "width_grid": width_grid,
                        "length_grid": length_grid,
                    }
                )
    else:
        mask = topo > BUILDING_THRESHOLD
        comps = connected_components(mask)

        for comp in comps:
            js = np.array([p[0] for p in comp])
            is_ = np.array([p[1] for p in comp])
            values = topo[js, is_]

            hb = float(values.mean())
            eta1 = int(len(values))

            width_grid = int(is_.max() - is_.min() + 1)
            length_grid = int(js.max() - js.min() + 1)

            blocks.append(
                {
                    "hb": hb,
                    "eta1": eta1,
                    "width_grid": width_grid,
                    "length_grid": length_grid,
                }
            )

    return blocks


def calc_full_area_geom(topo: np.ndarray, wind_direction_deg: float):
    """
    计算全域城市形态参数。

    lambda 计算方式尽量沿用 Fortran 的 lam3 逻辑：
    lam3 = (abs(width_grid*cos(alpha)) + abs(length_grid*sin(alpha))) * hb

    lambda = sum(lam3 * DX) / total_ground_area

    其中：
    total_ground_area = NX * DX * NY * DY
    """
    blocks = get_building_blocks(topo, use_height_bins=USE_HEIGHT_BINS)

    total_cells = topo.size
    total_ground_area = NX * DX * NY * DY

    building_mask = topo > BUILDING_THRESHOLD
    building_values_grid = topo[building_mask]

    if len(building_values_grid) == 0 or len(blocks) == 0:
        return {
            "sec": 1,
            "lambda": 0.0,
            "eta": 0.0,
            "h": 0.0,
            "sigmaH": 0.0,
            "HHmax": 0.0,
            "hmax_block_mean": 0.0,
            "nblock": 0,
        }

    alpha = math.radians(wind_direction_deg)

    lam3_sum = 0.0
    block_heights = []
    building_cell_count = 0

    for b in blocks:
        hb = b["hb"]
        width_grid = b["width_grid"]
        length_grid = b["length_grid"]
        eta1 = b["eta1"]

        lam3 = (
            abs(width_grid * math.cos(alpha))
            + abs(length_grid * math.sin(alpha))
        ) * hb

        lam3_sum += lam3
        block_heights.append(hb)
        building_cell_count += eta1

    block_heights = np.array(block_heights, dtype=float)

    lambda_value = lam3_sum * DX / total_ground_area
    eta_value = building_cell_count / total_cells

    # h / sigmaH 按建筑块平均高度统计，更接近原 Fortran geom1.txt 中 h 和 sigmaH 的含义
    h = float(block_heights.mean())
    sigmaH = float(block_heights.std(ddof=0))

    # hmax 给真实格点最大高度
    hmax = float(building_values_grid.max())

    # hmax_block_mean 是建筑块平均高度中的最大值，接近原 Fortran HHmax 的计算逻辑
    hmax_block_mean = float(block_heights.max())

    return {
        "sec": 1,
        "lambda": lambda_value,
        "eta": eta_value,
        "h": h,
        "sigmaH": sigmaH,
        "HHmax": hmax_block_mean,
        "nblock": len(blocks),
    }


def write_geom_like_output(result: dict, output_file: Path):
    """
    输出类似 geom1.txt 的文本文件。
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        f.write(
            ">> sec, lambda, eta, h, sigmaH, HHmax, nblock <<\n"
        )
        f.write(
            f"{result['sec']:4d}"
            f"{result['lambda']:12.6f}"
            f"{result['eta']:12.6f}"
            f"{result['h']:12.3f}"
            f"{result['sigmaH']:12.3f}"
            f"{result['HHmax']:12.3f}"
            f"{result['nblock']:10d}\n"
        )


def main():
    topo = read_topo_matrix(INPUT_FILE, NX, NY, SKIP_ROWS)

    result = calc_full_area_geom(
        topo=topo,
        wind_direction_deg=WIND_DIRECTION_DEG,
    )

    write_geom_like_output(result, OUTPUT_FILE)

    print("Done.")
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()