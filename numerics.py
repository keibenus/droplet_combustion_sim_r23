# =========================================
#         numerics.py (r4)
# =========================================
# numerics.py
"""Numerical schemes and helper functions, primarily for FVM."""

import numpy as np
import config

def interpolate_quick_uneven(phi_LL, phi_L, phi_R, r_LL, r_L, r_R, r_f):
    """
    不等間隔格子におけるラグランジュ補間を用いたQUICKスキーム。
    3点 (LL, L, R) を通る放物線を構築し、界面fでの値を返す。
    """
    # ゼロ割を避けるための微小量
    eps = 1e-15
    
    # ラグランジュ補間多項式の各項
    term_R = phi_R * (r_f - r_LL) * (r_f - r_L) / ((r_R - r_LL) * (r_R - r_L) + eps)
    term_L = phi_L * (r_f - r_LL) * (r_f - r_R) / ((r_L - r_LL) * (r_L - r_R) + eps)
    term_LL = phi_LL * (r_f - r_L) * (r_f - r_R) / ((r_LL - r_L) * (r_LL - r_R) + eps)
    
    return term_R + term_L + term_LL

def van_leer_limiter(r):
    """van Leer流束制限関数"""
    return (r + abs(r)) / (1.0 + abs(r) + 1e-15) # ゼロ割防止

# --- FVM Interpolation Schemes ---
def interpolate_face_value(
    # 必須の引数
    phi_L, phi_R, u_face,
    # スキームに応じて必要となる引数
    r_center_LL=None, r_center_L=None, r_center_R=None, r_center_RR=None, r_face=None,
    phi_LL=None, phi_RR=None, 
    scheme=config.ADVECTION_SCHEME):
    """
    指定されたスキームを用いて、面の物理量φを補間する。
    """
    if scheme == 'upwind':
        return phi_L if u_face >= 0.0 else phi_R

    elif scheme == 'central':
        if r_center_L is None or r_center_R is None or r_face is None:
            return 0.5 * (phi_L + phi_R)
        dr = r_center_R - r_center_L
        if abs(dr) < 1e-15:
            return 0.5 * (phi_L + phi_R)
        weight_R = (r_face - r_center_L) / dr
        weight_L = (r_center_R - r_face) / dr
        return weight_L * phi_L + weight_R * phi_R

    elif scheme == 'quick':
        if u_face >= 0.0: # 流れが正 (L -> R)
            if phi_LL is not None and all(v is not None for v in [r_center_LL, r_center_L, r_center_R, r_face]):
                return interpolate_quick_uneven(phi_LL, phi_L, phi_R, r_center_LL, r_center_L, r_center_R, r_face)
            else:
                return phi_L # 3点使えない場合は風上差分
        else: # 流れが負 (R -> L)
            if phi_RR is not None and all(v is not None for v in [r_center_L, r_center_R, r_center_RR, r_face]):
                # 風上側が R, RR になるため、引数の順番を入れ替えて補間関数を呼び出す
                return interpolate_quick_uneven(phi_RR, phi_R, phi_L, r_center_RR, r_center_R, r_center_L, r_face)
            else:
                return phi_R # 3点使えない場合は風上差分

    elif scheme == 'vanleer':
        if u_face >= 0.0: # 流れが正
            phi_upwind = phi_L
            if phi_LL is not None and r_center_L is not None and r_center_LL is not None:
                grad_local = (phi_R - phi_L) / (r_center_R - r_center_L + 1e-15)
                grad_upwind = (phi_L - phi_LL) / (r_center_L - r_center_LL + 1e-15)
                r_ratio = grad_upwind / (grad_local + 1e-15)
                
                phi_face = phi_L + van_leer_limiter(r_ratio) * 0.5 * (phi_R - phi_L)
                return phi_face
            else:
                return phi_upwind # フォールバック
        else: # 流れが負
            phi_upwind = phi_R
            if phi_RR is not None and r_center_R is not None and r_center_RR is not None:
                grad_local = (phi_L - phi_R) / (r_center_L - r_center_R + 1e-15)
                grad_upwind = (phi_R - phi_RR) / (r_center_R - r_center_RR + 1e-15)
                r_ratio = grad_upwind / (grad_local + 1e-15)
                
                phi_face = phi_R + van_leer_limiter(r_ratio) * 0.5 * (phi_L - phi_R)
                return phi_face
            else:
                return phi_upwind # フォールバック

    else:
        if config.LOG_LEVEL >= 0: print(f"Warning: Unknown advection scheme '{scheme}', defaulting to 'upwind'.")
        return phi_L if u_face >= 0.0 else phi_R
    
'''
def interpolate_face_value(phi_L, phi_R, u_face, r_center_L=None, r_center_R=None, r_face=None,
                           phi_LL=None, phi_RR=None, scheme=config.ADVECTION_SCHEME):
    """
    指定されたスキームを用いて、面の物理量φを補間する。
    'central'は不等間隔格子に対応した線形補間。
    'quick'は風上2点、風下1点を用いた二次補間。

    Args:
        phi_L, phi_R: 面を挟む左右のセル中心値
        u_face: 面での流速 (L -> R が正)
        r_center_L, r_center_R, r_face: 'central'スキーム用の座標
        phi_LL: 'quick'スキーム用の風上側から2番目のセル中心値
        scheme: スキーム選択 ('upwind', 'central', 'quick')
    """
    if scheme == 'upwind':
        # 風上差分 (変更なし)
        return phi_L if u_face >= 0.0 else phi_R
    
    elif scheme == 'central':
        # 2次精度中心差分（不等間隔格子対応の線形補間）
        if r_center_L is None or r_center_R is None or r_face is None:
            # 座標情報がない場合は、単純な算術平均にフォールバック
            return 0.5 * (phi_L + phi_R)
        
        dr = r_center_R - r_center_L
        if abs(dr) < 1e-15:
            return 0.5 * (phi_L + phi_R) # セル中心が重なっている場合
            
        # 面の位置に応じて線形補間
        weight_R = (r_face - r_center_L) / dr
        weight_L = (r_center_R - r_face) / dr
        
        return weight_L * phi_L + weight_R * phi_R

    elif scheme == 'quick':
        # 不等間隔格子に対応したQUICKスキーム
        # r_LL, r_L, r_R が必要だが、簡単のため等間隔格子と仮定した古典的なQUICKを実装
        # 3/8, 6/8, -1/8 の重みを使用
        if u_face >= 0.0:
            # 流れが正（左から右）の場合、風上点は L と LL
            if phi_LL is not None:
                # L(i-1), LL(i-2), R(i)
                # phi_face = 0.5*(phi_L + phi_R) - 0.125*(phi_R - 2*phi_L + phi_LL)
                # より一般的な形 (Hirsch, 1990)
                phi_face = (6.0/8.0)*phi_L + (3.0/8.0)*phi_R - (1.0/8.0)*phi_LL
                return phi_face
            else:
                # i=1 の面など、phi_LL がない場合は風上差分にフォールバック
                return phi_L
        else: # u_face < 0.0
            # 流れが負（右から左）の場合、風上点は R と RR
            # この1次元球対称コードでは、風下側（i+2）の値は直接渡せないため、
            # 右から左への流れの場合は、単純な中心差分（または風上差分）にフォールバックさせるのが安全策
            # ここでは、実装の対称性から風上差分（phi_R）を返す
            return phi_R

    elif scheme == 'vanleer':
        if u_face >= 0.0: # 流れが正 (L -> R)
            # 風上は L セル
            # 勾配比 r_i-1/2 = (phi_i - phi_i-1) / (phi_i-1 - phi_i-2)
            if phi_LL is not None:
                # ゼロ割を避ける
                denominator = phi_L - phi_LL
                if abs(denominator) < 1e-15:
                    r = 1.0 # 勾配が一定とみなす
                else:
                    r = (phi_R - phi_L) / denominator
                
                # 風上差分流束に制限関数を適用した高次項を追加
                return phi_L + 0.5 * van_leer_limiter(r) * (phi_L - phi_LL)
            else:
                # 3点使えない場合は風上差分にフォールバック
                return phi_L
        else: # 流れが負 (R -> L)
            # 風上は R セル
            # 勾配比 r_i+1/2 = (phi_i - phi_i+1) / (phi_i+1 - phi_i+2)
            if phi_RR is not None:
                denominator = phi_RR - phi_R
                if abs(denominator) < 1e-15:
                    r = 1.0
                else:
                    r = (phi_L - phi_R) / denominator
                
                return phi_R + 0.5 * van_leer_limiter(r) * (phi_R - phi_RR)
            else:
                return phi_R

    else:
        # 不明なスキームの場合は風上差分にフォールバック
        if config.LOG_LEVEL >= 0: print(f"Warning: Unknown advection scheme '{scheme}', defaulting to 'upwind'.")
        return phi_L if u_face >= 0.0 else phi_R
'''

# --- FVM Gradient Calculation (at faces, using adjacent cell centers) ---

def gradient_at_face(phi_L, phi_R, r_center_L, r_center_R):
    """Calculate gradient at the face between cells L and R."""
    dr_centers = r_center_R - r_center_L
    # Use a larger epsilon to avoid issues with very close centers
    if abs(dr_centers) < 1e-15:
        # Avoid division by zero if cell centers coincide
        return 0.0
    else:
        return (phi_R - phi_L) / dr_centers

# --- Property Averaging for Faces ---

def harmonic_mean(val_L, val_R):
    """Calculate harmonic mean, suitable for conductivities/diffusivities."""
    # Add small epsilon to avoid division by zero if val is exactly zero
    eps = 1e-15
    val_L_arr = np.asarray(val_L)
    val_R_arr = np.asarray(val_R)
    # Ensure values are non-negative before taking reciprocal
    val_L_safe = np.maximum(val_L_arr, eps)
    val_R_safe = np.maximum(val_R_arr, eps)
    denom = (1.0 / val_L_safe) + (1.0 / val_R_safe)
    # Avoid division by zero in the final step if both inputs were near zero
    result = np.where(denom > eps, 2.0 / denom, 0.0)
    return result

def arithmetic_mean(val_L, val_R):
    """Calculate arithmetic mean."""
    return 0.5 * (np.asarray(val_L) + np.asarray(val_R))

# --- Tridiagonal Matrix Solver (Thomas Algorithm) ---
def solve_tridiagonal(a, b, c, d):
    """
    Solves a tridiagonal system Ax = d using the Thomas algorithm.
    a: lower diagonal (a[0] is ignored)
    b: main diagonal
    c: upper diagonal (c[-1] is ignored)
    d: right hand side vector
    Returns: solution vector x, or None if solver fails.
    """
    n = len(d)
    if n == 0: return np.array([])
    if n == 1:
        if abs(b[0]) > 1e-15:
             return d / b[0]
        else:
             print("Error: Zero diagonal element in 1x1 system (TDMA).")
             return None # Indicate error

    # Create copies to avoid modifying input arrays if they are reused
    a_ = np.copy(a)
    b_ = np.copy(b)
    c_ = np.copy(c)
    d_ = np.copy(d)
    x = np.zeros(n)

    # <<< デバッグ用フラグ (config等から制御しても良い) >>>
    TDMA_DEBUG = config.LOG_LEVEL >= 3 # 例: LOG_LEVEL 3以上でTDMA内部を表示
    if TDMA_DEBUG: print("  --- TDMA Start ---")
    if TDMA_DEBUG: print(f"    Input d: {d[:min(5,n)]}...") # 入力dの先頭を表示

    # Forward elimination
    if abs(b_[0]) < 1e-15:
         print("Error: Zero pivot element b[0] in Thomas algorithm.")
         return None
    # <<< ここから修正・追加 >>>
    try:
        c_orig_0 = c_[0] # Store original c[0] for clarity
        c_[0] = c_[0] / b_[0]
        d_[0] = d_[0] / b_[0]
        if TDMA_DEBUG and n > 0: print(f"    FE i=0: c'={c_[0]:.4e}, d'={d_[0]:.4e} (c={c_orig_0:.4e}, b={b_[0]:.4e}, d={d[0]:.4e})") # d[0]は元の入力dを使う

        for i in range(1, n):
            a_i = a_[i] # Lower diagonal for row i
            c_prev_prime = c_[i-1] # c' from previous step
            d_prev_prime = d_[i-1] # d' from previous step
            b_i = b_[i] # Original diagonal for row i
            d_i_orig = d[i] # Original RHS for row i (use the uncopied 'd' here)
            c_i = c_[i] # Original upper diagonal c_i (needed only for c'_i calc)

            temp = b_i - a_i * c_prev_prime
            if abs(temp) < 1e-15: # ... (エラー処理) ...
                print("Error: temp < 1e-15")

            # Store original c[i] before modifying it
            c_i_orig = c_[i] if i < n - 1 else 0.0

            # Modify d first, as it uses the unmodified c'[i-1] and d'[i-1]
            d_i_new = (d_i_orig - a_i * d_prev_prime) / temp # d'_i calculation, use original d[i] from input
            d_[i] = d_i_new # Update d_ array

            # Modify c if not the last row
            if i < n - 1:
                c_i_new = c_i / temp # c'_i calculation
                c_[i] = c_i_new # Update c_ array

            if TDMA_DEBUG:
                 c_prime_disp = c_[i] if i < n-1 else float('nan')
                 print(f"    FE i={i}: temp={temp:.4e}, c'={c_prime_disp:.4e}, d'={d_[i]:.4e} (a={a_i:.4e}, b={b_i:.4e}, c={c_i_orig:.4e}, d={d_i_orig:.4e})")

    except Exception as e_fe:
        print(f"ERROR during TDMA Forward Elimination: {e_fe}")
        return None
    # <<< ここまで修正・追加 >>>

    # Back substitution
    # <<< ここから修正・追加 >>>
    try:
        x[n-1] = d_[n-1] # Start with the last element
        if TDMA_DEBUG: print(f"    BS i={n-1}: x={x[n-1]:.4e} (d'={d_[n-1]:.4e})")

        for i in range(n-2, -1, -1):
            d_prime_i = d_[i] # d' calculated during FE
            c_prime_i = c_[i] # c' calculated during FE
            x_next = x[i+1]  # x already calculated in previous BS step

            x_i_new = d_prime_i - c_prime_i * x_next
            x[i] = x_i_new # Update solution vector

            if TDMA_DEBUG: print(f"    BS i={i}: x={x[i]:.4e} (d'={d_prime_i:.4e}, c'={c_prime_i:.4e}, x_next={x_next:.4e})")

    except Exception as e_bs:
        print(f"ERROR during TDMA Back Substitution: {e_bs}")
        return None
    # <<< ここまで修正・追加 >>>

    # Check for NaNs in solution
    if np.any(np.isnan(x)):
        print("Error: NaN detected in TDMA solution.")
        return None

    if TDMA_DEBUG: print(f"    Output x: {x[:min(5,n)]}...")
    if TDMA_DEBUG: print("  --- TDMA End ---")
    
    return x

def calculate_adaptive_dt(
    u_g_faces: np.ndarray,      # Velocity at gas cell faces [m/s] (Ng+1,)
    lambda_g_centers: np.ndarray,# Thermal conductivity at gas cell centers [W/mK] (Ng,)
    rho_g_centers: np.ndarray,   # Density at gas cell centers [kg/m^3] (Ng,)
    cp_g_centers: np.ndarray,    # Specific heat at gas cell centers [J/kgK] (Ng,)
    Dk_g_centers: np.ndarray,   # Diffusion coeffs at gas cell centers [m^2/s] (Nsp, Ng)
    lambda_l_centers: np.ndarray,# Thermal conductivity at liquid cell centers [W/mK] (Nl,)
    rho_l_centers: np.ndarray,   # Density at liquid cell centers [kg/m^3] (Nl,)
    cp_l_centers: np.ndarray,    # Specific heat at liquid cell centers [J/kgK] (Nl,)
    r_g_nodes: np.ndarray,       # Gas face radii [m] (Ng+1,)
    r_l_nodes: np.ndarray,       # Liquid face radii [m] (Nl+1,)
    current_dt: float,           # Previous time step (for limiting increase)
    nsp: int,
    is_gas_only_phase: bool
    ):
    """
    Calculates the maximum stable time step based on CFL conditions for FVM.
    Handles both two-phase and gas-only phase calculations adaptively.
    """
    if is_gas_only_phase:
        # 液滴が消滅した場合、configで指定された固定の時間刻み幅を返す
        # これにより、微小な格子幅に起因するdtの過度な減少を回避する
        dt_grow_after_evap = current_dt * config.DT_INCREASE_FACTOR_EVAP
        dt_new = min(config.DT_POST_EVAPORATION, dt_grow_after_evap)
        return max(dt_new, config.DT_MIN_VALUE)
    
    if not config.USE_ADAPTIVE_DT:
        return config.DT_INIT # Return fixed small dt if adaptive is off

    Ng = len(rho_g_centers)
    Nl = len(rho_l_centers)

    # Initialize minimum dt allowed by stability to a large value
    dt_stab = config.T_END # Start with a large value

    # --- Calculate characteristic cell sizes ---
    dr_g = np.diff(r_g_nodes) if Ng > 0 else np.array([])
    dr_l = np.diff(r_l_nodes) if Nl > 0 else np.array([])
    dr_g = np.maximum(dr_g, 1e-12) # Avoid zero dr
    dr_l = np.maximum(dr_l, 1e-12)

    min_dt_adv_g = np.inf
    min_dt_diff_T_g = np.inf
    min_dt_diff_Y_g = np.inf
    min_dt_diff_T_l = np.inf

    # --- Advection Limit (Gas) ---
    if Ng > 0:
        # Use max velocity associated with each cell (max of face velocities)
        u_abs_max_per_cell = np.maximum(np.abs(u_g_faces[:-1]), np.abs(u_g_faces[1:]))
        dt_adv_g_local = config.CFL_NUMBER * dr_g / (u_abs_max_per_cell + 1e-9) # Add epsilon for u=0
        if len(dt_adv_g_local) > 0:
            min_dt_adv_g = np.min(dt_adv_g_local)
            dt_stab = min(dt_stab, min_dt_adv_g)

    # --- Diffusion Limit (Gas - Thermal) ---
    if Ng > 0:
        alpha_g = lambda_g_centers / (rho_g_centers * cp_g_centers + 1e-9)
        alpha_g = np.maximum(alpha_g, 1e-12) # Prevent zero alpha
        dt_diff_T_g_local = 0.5 * config.CFL_NUMBER * dr_g**2 / alpha_g
        if len(dt_diff_T_g_local) > 0:
            min_dt_diff_T_g = np.min(dt_diff_T_g_local)
            dt_stab = min(dt_stab, min_dt_diff_T_g)

    # --- Diffusion Limit (Gas - Species) ---
    if Ng > 0 and nsp > 0 and Dk_g_centers.size > 0:
        # Use Dk at cell centers and cell width dr_g
        Dk_max_g = np.max(Dk_g_centers, axis=0) # Max Dk at each cell center
        Dk_max_g = np.maximum(Dk_max_g, 1e-12) # Prevent zero Dk
        dt_diff_Y_g_local = 0.5 * config.CFL_NUMBER * dr_g**2 / Dk_max_g
        if len(dt_diff_Y_g_local) > 0:
            min_dt_diff_Y_g = np.min(dt_diff_Y_g_local)
            dt_stab = min(dt_stab, min_dt_diff_Y_g)

    # --- Diffusion Limit (Liquid - Thermal) ---
    if Nl > 0:
        # Use properties at cell centers and cell width dr_l
        alpha_l = lambda_l_centers / (rho_l_centers * cp_l_centers + 1e-9)
        alpha_l = np.maximum(alpha_l, 1e-12) # Prevent zero alpha
        dt_diff_T_l_local = 0.5 * config.CFL_NUMBER * dr_l**2 / alpha_l
        if len(dt_diff_T_l_local) > 0:
             min_dt_diff_T_l = np.min(dt_diff_T_l_local)
             dt_stab = min(dt_stab, min_dt_diff_T_l)

    # --- Apply Limits ---
    # Limit the increase from the previous time step
    dt_new = min(dt_stab, current_dt * config.DT_MAX_INCREASE_FACTOR)
    # Apply absolute max and min limits
    dt_new = min(dt_new, config.DT_MAX)
    dt_new = max(dt_new, config.DT_MIN_VALUE) # Ensure dt doesn't become zero or negative

    if config.LOG_LEVEL >= 2:
         # Use np.min on the local arrays to handle cases where Ng=0 or Nl=0 correctly
         min_dt_adv_g_disp = np.min(dt_adv_g_local) if Ng>0 and len(dt_adv_g_local)>0 else np.inf
         min_dt_diff_T_g_disp = np.min(dt_diff_T_g_local) if Ng>0 and len(dt_diff_T_g_local)>0 else np.inf
         min_dt_diff_Y_g_disp = np.min(dt_diff_Y_g_local) if Ng>0 and len(dt_diff_Y_g_local)>0 else np.inf
         min_dt_diff_T_l_disp = np.min(dt_diff_T_l_local) if Nl>0 and len(dt_diff_T_l_local)>0 else np.inf

         print(f"      DEBUG dt Calc: AdvG={min_dt_adv_g_disp:.2e} "
               f"DiffTg={min_dt_diff_T_g_disp:.2e} "
               f"DiffYg={min_dt_diff_Y_g_disp:.2e} "
               f"DiffTl={min_dt_diff_T_l_disp:.2e} -> dt_stab={dt_stab:.2e} -> dt_new={dt_new:.2e}", end='\r', flush=True)

    return dt_new

'''
    if not config.USE_ADAPTIVE_DT:
        return config.DT_INIT

    Ng = len(rho_g_centers)
    Nl = len(rho_l_centers)

    # 安定性から決まるdtの最小値を保持する変数
    dt_stab = config.T_END

    # --- Gas Phase Stability Calculations ---
    if Ng > 0:
        dr_g = np.diff(r_g_nodes)
        dr_g = np.maximum(dr_g, 1e-12)

        # Advection Limit (Gas)
        u_abs_max_per_cell = np.maximum(np.abs(u_g_faces[:-1]), np.abs(u_g_faces[1:]))
        dt_adv_g_local = config.CFL_NUMBER * dr_g / (u_abs_max_per_cell + 1e-9)
        dt_stab = min(dt_stab, np.min(dt_adv_g_local))

        # Thermal Diffusion Limit (Gas)
        alpha_g = lambda_g_centers / (rho_g_centers * cp_g_centers + 1e-9)
        alpha_g = np.maximum(alpha_g, 1e-12)
        dt_diff_T_g_local = 0.5 * config.CFL_NUMBER * dr_g**2 / alpha_g
        dt_stab = min(dt_stab, np.min(dt_diff_T_g_local))

        # Species Diffusion Limit (Gas)
        if nsp > 0 and Dk_g_centers.size > 0:
            Dk_max_g = np.max(Dk_g_centers, axis=0)
            Dk_max_g = np.maximum(Dk_max_g, 1e-12)
            dt_diff_Y_g_local = 0.5 * config.CFL_NUMBER * dr_g**2 / Dk_max_g
            dt_stab = min(dt_stab, np.min(dt_diff_Y_g_local))

    # --- Liquid Phase Stability Calculations (ONLY if droplet exists) ---
    if not is_gas_only_phase and Nl > 0:
        dr_l = np.diff(r_l_nodes)
        dr_l = np.maximum(dr_l, 1e-12)
        
        # Thermal Diffusion Limit (Liquid)
        alpha_l = lambda_l_centers / (rho_l_centers * cp_l_centers + 1e-9)
        alpha_l = np.maximum(alpha_l, 1e-12)
        dt_diff_T_l_local = 0.5 * config.CFL_NUMBER * dr_l**2 / alpha_l
        dt_stab = min(dt_stab, np.min(dt_diff_T_l_local))

    # --- Final dt Determination (Corrected Logic) ---
    # 1. まず、前回のdtをスケール倍して、増加させる候補dtを計算
    dt_grown = current_dt * config.DT_MAX_INCREASE_FACTOR
    
    # 2. 増加させたdtが、CFL条件から決まる安定限界dtや絶対最大dtを超えないように、min関数で制限をかける
    #    つまり、dt_grown, dt_stab, DT_MAX の中で最も小さい値を選択する
    dt_new = min(dt_grown, dt_stab, config.DT_POST_EVAPORATION)
    
    # 3. 絶対最小dtよりは小さくならないようにする
    dt_new = max(dt_new, config.DT_MIN_VALUE)

    if config.LOG_LEVEL >= 2:
        print(f"      DEBUG dt Calc: dt_stab={dt_stab:.2e}, dt_grown={dt_grown:.2e} -> dt_new={dt_new:.2e} (is_gas_only={is_gas_only_phase})", end='\r', flush=True)

    return dt_new
'''