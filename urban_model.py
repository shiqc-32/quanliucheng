import os
import numpy as np

# Constants
CD = 1.2
KAPPA = 0.4
ALPHA = 0.81
A_MIN = 1.0e-6

# ==========================================
#  Helper Functions (IO & Formatting)
# ==========================================

def read_line_as_floats(file_obj):
    """读取CSV行，返回浮点列表"""
    line = file_obj.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return []
    parts = line.split(',')
    # 过滤空字符串
    valid_parts = [p.strip() for p in parts if p.strip()]
    try:
        return [float(p) for p in valid_parts]
    except ValueError as e:
        return []

def _resolve_path(cfg, path_str, is_output=False):
    """解析路径"""
    if not isinstance(path_str, str) or not path_str.startswith("@paths."):
        return path_str
    key = path_str.split('.')[1]
    paths_cfg = cfg.get('paths', {})
    filename = paths_cfg.get(key, key)
    base_dir = paths_cfg.get('output_dir', './') if is_output else paths_cfg.get('input_dir', './')
    return os.path.join(base_dir, filename)

# ==========================================
#  Scientific Calculation Functions
# ==========================================

def s_rp_debug(rp):
    nk = len(rp)
    # 修正逻辑：如果下层 >= 1，取上层值
    for i in range(nk - 2, -1, -1):
        if rp[i] >= 1.0:
            rp[i] = rp[i + 1]
    return rp

def s_rf_profile(dz, rp, rf_tot):
    rp_sum = np.sum(rp)
    rf_density = np.zeros_like(rp)
    if rp_sum > 0:
        rf_density = rf_tot * (rp / rp_sum) / dz
    return rf_density

def s_reconstructed_layers(rp, nL_default, drp_threshold):
    """
    重构层数计算
    Returns: (nL, error_msg)
    """
    nk = len(rp)
    rp2 = rp.copy()
    rp2[rp2 < drp_threshold] = 0.0
    
    kl_top = 0
    for k in range(nk - 1, -1, -1):
        if rp2[k] >= drp_threshold:
            kl_top = k + 1 
            break
            
    if kl_top == 0:
        return 0, "Warning: No effective building layers (all rp < threshold)"

    delta_rp = []
    for k in range(kl_top): 
        val_next = rp2[k+1] if (k + 1) < nk else 0.0
        delta_rp.append(rp2[k] - val_next)
    delta_rp = np.array(delta_rp)
    
    nL_max = np.sum(delta_rp >= drp_threshold)
    
    if nL_default <= nL_max:
        nL = nL_default
    else:
        nL = nL_max
        
    if nL > kl_top:
        nL = kl_top
        
    return nL, "" # No error

def s_pre_dz0(dz, rp, rf_density, nL, drp_threshold):
    """
    准备 d 和 z0 计算所需的廓线参数
    """
    nk = len(rp)
    rp2 = rp.copy()
    rf2 = rf_density.copy()
    mask = rp2 < drp_threshold
    rp2[mask] = 0.0
    rf2[mask] = 0.0
    
    kl_top = 0
    for k in range(nk - 1, -1, -1):
        if rp2[k] >= drp_threshold:
            kl_top = k + 1
            break
            
    delta_rp_list = []
    for k in range(kl_top - 1):
        val_next = rp2[k+1] if (k+1) < nk else 0.0
        delta_rp_list.append(rp2[k] - val_next)
    delta_rp_arr = np.array(delta_rp_list)
    
    if len(delta_rp_arr) > 0:
        sorted_indices = np.argsort(delta_rp_arr)[::-1]
        indices = sorted_indices[:nL-1]
        indices = np.sort(indices)
    else:
        indices = np.array([], dtype=int)

    Have = 0.0
    rp_base = rp2[0] if rp2[0] > 0 else 1.0 
    for k in range(nk - 1):
        term = (k + 1) * dz * (rp2[k] - rp2[k+1]) / rp_base
        Have += term
        
    kl_up = np.zeros(nL, dtype=int)
    if nL > 1:
        valid_indices = indices[indices < (kl_top-1)] 
        kl_up[:len(valid_indices)] = valid_indices + 1
    kl_up[nL-1] = kl_top
    
    kl_low = np.zeros(nL, dtype=int)
    kl_low[0] = 1
    for k in range(1, nL):
        kl_low[k] = kl_up[k-1] + 1
        
    H_arr = np.zeros(nL + 2) 
    eta = np.zeros(nL)
    lamb = np.zeros(nL)
    
    for k in range(nL): 
        if k >= len(kl_low) or k >= len(kl_up): break
        ik0 = max(0, min(nk-1, kl_low[k] - 1))
        ik1 = max(0, min(nk-1, kl_up[k] - 1))
        
        eta[k] = rp2[ik0]
        H_arr[k+2] = (ik1 + 1) * dz
        sum_rf = 0.0
        if ik1 >= ik0:
            sum_rf = np.sum(rf2[ik0:ik1+1]) * dz
        lamb[k] = sum_rf
        
    eta2 = np.zeros(nk)
    lambda2 = np.zeros(nk)
    zr = np.zeros(nk)
    
    H_max = H_arr[nL+1]
    dzr = H_max / float(nk) if nk > 0 else 0.0
    
    for kr in range(nk):
        z_curr = (kr + 1) * dzr
        zr[kr] = z_curr
        for k in range(nL):
            h_top = H_arr[k+2]
            h_bot = H_arr[k+1]
            if h_bot < z_curr <= h_top:
                dH = h_top - h_bot
                eta2[kr] = eta[k]
                if dH > 1e-6:
                    lambda2[kr] = lamb[k] / dH
                break
    return Have, H_arr, eta, lamb, eta2, lambda2, zr

def s_dz0(nL, Have, H_arr, eta, lamb):
    """
    主计算程序
    Returns: (d_o_H, z0_o_H, error_msg)
    """
    error_msg = ""
    
    if Have < 1e-6:
        return np.nan, np.nan, "Fail: Have approx 0"

    Hmax = H_arr[nL+1]
    
    deltaH = np.zeros(nL + 1)
    for k in range(nL + 1):
        deltaH[k] = H_arr[k+1] - H_arr[k]
        
    eta_base = eta[0]
    lambda_tot = np.sum(lamb)
    
    if lambda_tot <= 1e-9:
         # 物理上没有迎风面积，可能是平地
         return np.nan, np.nan, "Fail: lambda_tot approx 0"

    a_const = 9.6 * lambda_tot
    al = np.zeros(nL + 1)
    
    eta_eff = 0.0
    for k in range(nL):
        eta_eff += eta[k] * lamb[k] / lambda_tot
            
    for k in range(nL):
        val = 0.0
        if eta_eff > 1e-9:
            val = a_const * (eta[k]/eta_eff)**(-0.05)
        else:
            # eta_eff 过小会导致除零
            return np.nan, np.nan, "Fail: eta_eff approx 0 in alpha calc"
        al[k+1] = val
        
    AA = (ALPHA**2) * (CD * 2.0 * a_const) / (4.0 * KAPPA**2) * np.exp(-2.0 * a_const)
    B = 0.0
    
    for k in range(1, nL + 1):
        ctmp = 0.0
        for kt in range(0, k):
            if Hmax > 1e-6:
                ctmp += 2.0 * al[kt] * deltaH[kt] / Hmax
            
        term1 = AA * lamb[k-1] * Hmax 
        term2 = al[k] * deltaH[k]
        
        # 严格检查分母
        if abs(term2) < 1e-12: 
             return np.nan, np.nan, f"Fail: Zero denominator in B integral at layer {k}"

        term = (term1 / term2) * np.exp(ctmp) * (np.exp(2.0 * al[k] * deltaH[k] / Hmax) - 1.0)
        B += term
            
    # 计算 d/H
    d_o_H = 0.0
    factor = 0.0
    if eta_base > 0:
            factor = (4.43**(-eta_base)) * (eta_base - 1.0)
    d_o_H = (1.0 + factor) * Hmax / Have
        
    # 计算 z0/H
    z0_o_H = np.nan
    
    # ---------------------------------------------------------
    # 核心检查点：B 的状态
    # ---------------------------------------------------------
    if B <= 0:
        error_msg = f"Fail: B value non-positive (B={B:.4e}). Cannot do B**(-0.5)"
        return d_o_H, np.nan, error_msg
    else:
        try:
            # 尝试计算
            exponent_arg = -(B**(-0.5))
            # 检查指数是否过小 (下溢) 或过大
            # 这里我们不自动修复，而是报告异常
            z0_o_H = (Hmax / Have - d_o_H) * np.exp(exponent_arg)
            
            # 物理合理性检查
            if z0_o_H < 0:
                error_msg = f"Warning: Calc result negative z0/H ({z0_o_H:.4f})"
                # 此时还是返回负值，让用户看到
                
        except Exception as e:
            return d_o_H, np.nan, f"Fail: Math error in z0 calc ({str(e)})"
        
    return d_o_H, z0_o_H, error_msg

# ==========================================
#  Main Driver Function
# ==========================================

def run_rur_from_cfg(global_cfg):
    """
    执行 RUR 方案计算
    """
    case_cfg = global_cfg.get('dz0_check', {})
    if not case_cfg.get('enable', False):
        print("dz0_check is disabled.")
        return

    print("Running rur calculation check (with remarks)...")

    nsites = case_cfg['nsites']
    nk = case_cfg['nk']
    dz = case_cfg['dz']
    nL_default = case_cfg['nL_default']
    drp_threshold = case_cfg['drp_threshold']
    
    path_rp = _resolve_path(global_cfg, case_cfg['rp_profiles_csv'], is_output=False)
    path_rf = _resolve_path(global_cfg, case_cfg['r_f_total_csv'], is_output=False)
    path_ref = _resolve_path(global_cfg, case_cfg['table1_csv'], is_output=False)
    path_out = _resolve_path(global_cfg, case_cfg['out_problem_csv'], is_output=True)
    
    output_dir = os.path.dirname(path_out)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        f_rp = open(path_rp, 'r', encoding='utf-8-sig')
        f_rf = open(path_rf, 'r', encoding='utf-8-sig')
        f_ref = open(path_ref, 'r', encoding='utf-8-sig')
        f_out_summary = open(path_out, 'w', encoding='utf-8')
        
        _ = f_rp.readline()
        _ = f_rf.readline()
        header_ref_line = f_ref.readline().strip()
        varlist = header_ref_line.split(',')
        
        # 增加 Remark 列
        new_headers = ['Have_rur', 'Hmax_rur', 'd_rur(m)', 'z0_rur(m)', 'd_rur_o_Have', 'z0_rur_o_Have', 'Remark']
        full_header = varlist + new_headers
        f_out_summary.write(','.join(full_header) + '\n')
        
        for site_id in range(1, nsites + 1):
            remark_list = [] # 收集当前站点的所有信息
            
            # --- 读取输入 ---
            rp_data_line = read_line_as_floats(f_rp)
            if not rp_data_line: 
                print(f"Stopping at site {site_id} (EOF)")
                break
            
            rp_raw_values = rp_data_line[1:] # 忽略 ID
            
            # 检查数据长度
            len_raw = len(rp_raw_values)
            if len_raw < nk:
                remark_list.append(f"Info: Data padded (raw={len_raw}, req={nk})")
                padding = np.zeros(nk - len_raw)
                rp = np.concatenate((np.array(rp_raw_values), padding))
            else:
                rp = np.array(rp_raw_values[:nk])
            
            rf_data_line = read_line_as_floats(f_rf)
            rf_tot = rf_data_line[0] if rf_data_line else 0.0
            
            ref_data = read_line_as_floats(f_ref)
            if not ref_data: ref_data = []

            # --- 算法步骤 ---
            rp = s_rp_debug(rp)
            rf_density = s_rf_profile(dz, rp, rf_tot)
            
            # 1. 重构层数
            nL, nL_msg = s_reconstructed_layers(rp, nL_default, drp_threshold)
            if nL_msg:
                remark_list.append(nL_msg)
            
            if nL == 0:
                # 无法计算
                Have = 0.0
                Hmax = 0.0
                d_val = np.nan
                z0_val = np.nan
                d_o_H = np.nan
                z0_o_H = np.nan
                remark_list.append("Fail: nL=0 (Skipping calc)")
            else:
                # 2. 准备参数
                Have, H_arr, eta, lamb, eta2, lambda2, zr = s_pre_dz0(
                    dz, rp, rf_density, nL, drp_threshold
                )
                
                # 3. 主计算 (接收错误信息)
                d_o_H, z0_o_H, error_msg = s_dz0(nL, Have, H_arr, eta, lamb)
                Hmax = H_arr[nL+1]
                
                if error_msg:
                    remark_list.append(error_msg)
                
                # 结果计算
                d_val = d_o_H * Have
                z0_val = z0_o_H * Have

            # --- 写入输出 ---
            # 拼接备注
            full_remark = "; ".join(remark_list)
            
            out_vals = ref_data + [Have, Hmax, d_val, z0_val, d_o_H, z0_o_H, full_remark]
            
            # 格式化: 浮点数保留小数，NaN 自动变成 'nan'，字符串原样输出
            out_str_parts = []
            for x in out_vals:
                if isinstance(x, str):
                    out_str_parts.append(x)
                elif isinstance(x, float):
                    out_str_parts.append(f"{x:.6g}")
                else:
                    out_str_parts.append(str(x))
            
            f_out_summary.write(",".join(out_str_parts) + '\n')
            
            # 写入廓线 (即使计算失败也写入几何廓线)
            id_str = f"{site_id:03d}"
            fn1 = os.path.join(output_dir, f"ID{id_str}_rpf_profiles_ori.dat")
            fn2 = os.path.join(output_dir, f"ID{id_str}_rpf_profiles_reconstructed.dat")
            
            with open(fn1, 'w') as f1:
                f1.write(f"{'Height':>12}{'rp':>12}{'rf':>12}\n")
                for k in range(nk):
                    ztmp = (k + 1) * dz
                    f1.write(f"{ztmp:12.2f}{rp[k]:12.6f}{rf_density[k]:12.6f}\n")
                    
            with open(fn2, 'w') as f2:
                f2.write(f"{'Height':>12}{'rp':>12}{'rf':>12}\n")
                if nL > 0: # 只有 nL > 0 才有重构数据
                    for k in range(nk):
                        f2.write(f"{zr[k]:12.4f}{eta2[k]:12.6f}{lambda2[k]:12.6f}\n")

        print("Calculation finished. Check 'Remark' column in output.")

    except IOError as e:
        print(f"File Error: {e}")
    finally:
        if 'f_rp' in locals() and not f_rp.closed: f_rp.close()
        if 'f_rf' in locals() and not f_rf.closed: f_rf.close()
        if 'f_ref' in locals() and not f_ref.closed: f_ref.close()
        if 'f_out_summary' in locals() and not f_out_summary.closed: f_out_summary.close()