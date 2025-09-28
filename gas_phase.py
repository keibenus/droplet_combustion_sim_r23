# =========================================
#          gas_phase.py (r4)
# =========================================
# gas_phase.py
"""Calculates the RHS for gas phase equations using FVM."""
import numpy as np
import cantera as ct
import config
import properties
import numerics
from properties import GasProperties # For type hinting
from grid import face_areas # Need face areas
from numerics import interpolate_face_value, gradient_at_face, harmonic_mean, arithmetic_mean
import reactions # Import reactions module

# --- Function for Explicit Advection Term Calculation ---
def calculate_gas_advection_rhs(
    T_g: np.ndarray, Y_g: np.ndarray, rho_g: np.ndarray, cp_g: np.ndarray, h_k_g: np.ndarray,
    u_g_faces: np.ndarray, # Velocities at faces (Ng+1,) based on t_n or previous iteration
    r_g_centers: np.ndarray, r_g_nodes: np.ndarray, volumes_g: np.ndarray,
    Ng: int, nsp: int,
    mdot_boundary: float, # Mass flux density [kg/m2/s] crossing face 0 (positive outward from droplet)
    Y_k_boundary: np.ndarray, # Mass fractions of species crossing face 0
    gas_props: GasProperties, P: float
    ):
    """
    Calculates the RHS contribution from advection only using FVM.
    Fluxes are defined based on face velocities (u_g_faces).
    Returns dT_dt_adv [K/s] and dY_dt_adv [1/s].
    """
    dT_dt_adv = np.zeros(Ng)
    dY_dt_adv = np.zeros((nsp, Ng))
    if Ng <= 0: return dT_dt_adv, dY_dt_adv

    A_g_faces = face_areas(r_g_nodes) # Areas of faces 0 to Ng

    # Allocate flux arrays (Rate of transport across face i, positive in +r direction)
    AdvFluxRate_H = np.zeros(Ng + 1) # Advected enthalpy flux rate [W]
    AdvFluxRate_Yk = np.zeros((nsp, Ng + 1)) # Advected species mass flux rate [kg/s]

    # Calculate fluxes through internal faces (i=1 to Ng-1)
    for i in range(1, Ng):
        u_f = u_g_faces[i] # Velocity at face i
        # Mass flux rate across face i = rho_face * u_face * A_face [kg/s]
        rho_f = arithmetic_mean(rho_g[i-1], rho_g[i])
        mass_flux_rate = rho_f * u_f * A_g_faces[i]
        
        # スキームが必要とする全ての点の座標と値を取得
        r_face = r_g_nodes[i]
        r_c_L = r_g_centers[i-1]
        r_c_R = r_g_centers[i]
        phi_L_T, phi_R_T = T_g[i-1], T_g[i]
        phi_L_Y, phi_R_Y = Y_g[:, i-1], Y_g[:, i]
        
        # 境界条件を考慮し、範囲外の点はNoneとする
        r_c_LL = r_g_centers[i-2] if i >= 2 else None
        phi_LL_T = T_g[i-2] if i >= 2 else None
        phi_LL_Y = Y_g[:, i-2] if i >= 2 else None
        
        r_c_RR = r_g_centers[i+1] if i < Ng - 1 else None
        phi_RR_T = T_g[i+1] if i < Ng - 1 else None
        phi_RR_Y = Y_g[:, i+1] if i < Ng - 1 else None

        T_adv = numerics.interpolate_face_value(
            phi_L_T, phi_R_T, u_f, 
            r_center_LL=r_c_LL, r_center_L=r_c_L, r_center_R=r_c_R, r_center_RR=r_c_RR, r_face=r_face,
            phi_LL=phi_LL_T, phi_RR=phi_RR_T
        )
        Yk_adv = np.array([numerics.interpolate_face_value(
                                phi_L_Y[k], phi_R_Y[k], u_f,
                                r_center_LL=r_c_LL, r_center_L=r_c_L, r_center_R=r_c_R, r_center_RR=r_c_RR, r_face=r_face,
                                phi_LL=(phi_LL_Y[k] if phi_LL_Y is not None else None),
                                phi_RR=(phi_RR_Y[k] if phi_RR_Y is not None else None)
                            ) for k in range(nsp)])
        '''
        # Interpolate values at face i using specified scheme (e.g., upwind)
        ##T_adv = interpolate_face_value(T_g[i-1], T_g[i], u_f) # Advected temperature
        ##Yk_adv = np.array([interpolate_face_value(Y_g[k, i-1], Y_g[k, i], u_f) for k in range(nsp)])
        # --- [修正箇所: 座標情報を渡して補間値を計算] ---
        r_face = r_g_nodes[i]
        r_c_L = r_g_centers[i-1]
        r_c_R = r_g_centers[i]
        # --- [QUICKスキーム用のphi_LLを定義] ---
        # スキームが必要とする隣接点の情報を準備
        T_LL = T_g[i-2] if i >= 2 else None
        Yk_LL = Y_g[:, i-2] if i >= 2 else None
        
        T_RR = T_g[i+1] if i < Ng - 1 else None
        Yk_RR = Y_g[:, i+1] if i < Ng - 1 else None

        T_adv = interpolate_face_value(T_g[i-1], T_g[i], u_f, r_c_L, r_c_R, r_face,
                                       phi_LL=T_LL, phi_RR=T_RR)
        Yk_adv = np.array([interpolate_face_value(Y_g[k, i-1], Y_g[k, i], u_f, r_c_L, r_c_R, r_face,
                                                  phi_LL=(Yk_LL[k] if Yk_LL is not None else None),
                                                  phi_RR=(Yk_RR[k] if Yk_RR is not None else None))
                           for k in range(nsp)])
        '''
        # --- [ここまで修正] ---

        #T_adv = numerics.interpolate_face_value(T_g[i-1], T_g[i], u_f, r_c_L, r_c_R, r_face)
        #Yk_adv = np.array([numerics.interpolate_face_value(Y_g[k, i-1], Y_g[k, i], u_f, r_c_L, r_c_R, r_face) for k in range(nsp)])
        # ------------------------------------------------
        # Ensure normalized advected mass fractions
        Yk_adv = np.maximum(Yk_adv, 0.0)
        sum_yk_adv = np.sum(Yk_adv)
        if sum_yk_adv > 1e-9: Yk_adv /= sum_yk_adv

        # Enthalpy at face (approximate using advected T, Yk)
        # Need to get hk at T_adv, Yk_adv. Requires setting Cantera state.
        # Optimization: Use hk values from cell centers and interpolate?
        # hk_adv = np.array([interpolate_face_value(h_k_g[k, i-1], h_k_g[k, i], u_f) for k in range(nsp)])
        # Let's recalculate hk based on T_adv, Yk_adv for better accuracy (may slow down)
        # Need GasProperties object? Or pass Cantera object? Assuming we pass hk_g evaluated at cell centers.
        hk_L = h_k_g[:, i-1]; hk_R = h_k_g[:, i]
        hk_adv = np.array([interpolate_face_value(hk_L[k], hk_R[k], u_f) for k in range(nsp)])


        # Enthalpy and Species Mass Flux Rate across face i [W] and [kg/s]
        AdvFluxRate_H[i] = mass_flux_rate * np.sum(Yk_adv * hk_adv)
        AdvFluxRate_Yk[:, i] = mass_flux_rate * Yk_adv

    # --- Boundary Face 0 (r=R) ---
    # Mass flux rate = mdot_boundary * Area [kg/s] (positive outward)
    mass_flux_rate_0 = mdot_boundary * A_g_faces[0]
    Yk_boundary = Y_k_boundary # Use provided boundary mass fractions
    # Need enthalpy at the boundary. Use state at surface (T_s, Y_eq) from previous step?
    # Or use state of first gas cell T_g[0]? Let's use T_g[0], Y_k_boundary for enthalpy h_k_boundary.
    # This needs careful consideration - enthalpy flux depends on what T crosses the boundary.
    # Let's use the enthalpy from the first gas cell as approximation for advected enthalpy.
    hk_boundary = h_k_g[:, 0] # Approximated with first cell enthalpy

    ##AdvFluxRate_H[0] = mass_flux_rate_0 * np.sum(Yk_boundary * hk_boundary)
    ##AdvFluxRate_Yk[:, 0] = mass_flux_rate_0 * Yk_boundary
    AdvFluxRate_H[0] = 0.0
    AdvFluxRate_Yk[:, 0] = 0.0
    
    # --- Boundary Face Ng (r=rmax) ---
    '''
    # Zero velocity, zero mass flux at the closed outer boundary
    AdvFluxRate_H[Ng] = 0.0
    AdvFluxRate_Yk[:, Ng] = 0.0
    '''
    # --- Boundary Face Ng (r=rmax) ---
    # 開放境界('OPEN_FIXED_AIR')か閉鎖境界('CLOSED')かで処理を分岐
    if config.OUTER_BOUNDARY_TYPE == 'OPEN_FIXED_AIR':
        u_f_outer = u_g_faces[Ng] # 最外殻の流速
        rho_f_outer = rho_g[Ng-1] # 最も外側のセルの密度で代用

        # 質量流束率 [kg/s]
        mass_flux_rate_outer = rho_f_outer * u_f_outer * A_g_faces[Ng]

        # 流入か流出かで運ばれる物理量を決定 (Upwind)
        if u_f_outer >= 0: # 流出の場合
            # 最も外側のセルの状態 (T_g[Ng-1], Y_g[:, Ng-1]) が出ていく
            T_adv_outer = T_g[Ng-1]
            Yk_adv_outer = Y_g[:, Ng-1]
            # エンタルピーも最も外側のセルのものを使用
            hk_adv_outer = h_k_g[:, Ng-1]

        else: # 流入の場合
            # 周囲空気の状態 (T_amb, Y_air) が入ってくる
            # この関数は main ループから呼ばれるため、最新の周囲温度を別途取得する必要がある
            # 簡単のため、ここでは気相第一セルの温度で代用するが、より正確には T_amb を渡すべき
            # (main側の修正で対応するため、ここでは気相最終セルの温度で近似)
            T_adv_outer = T_g[Ng-1] # 本来は T_amb
            
            # 周囲空気の組成 Y_k_air を取得
            # ここでは簡単のため、初期の周囲空気組成で代用
            gas_props.set_state(T_adv_outer, P, config.X_INF_INIT)
            Yk_adv_outer = gas_props.gas.Y
            hk_adv_outer = gas_props.get_partial_enthalpies_mass(T_adv_outer, P, Yk_adv_outer)

        # 流束を計算
        AdvFluxRate_H[Ng] = mass_flux_rate_outer * np.sum(Yk_adv_outer * hk_adv_outer)
        AdvFluxRate_Yk[:, Ng] = mass_flux_rate_outer * Yk_adv_outer
    
    else: # 'CLOSED' の場合（従来通り）
        AdvFluxRate_H[Ng] = 0.0
        AdvFluxRate_Yk[:, Ng] = 0.0

    # Calculate RHS = (Flux_In - Flux_Out) / (rho * V) for each cell
    for i in range(Ng):
        if rho_g[i] < 1e-9 or volumes_g[i] < 1e-25 or cp_g[i] < 1e-6: continue # Skip invalid cells
        # Net Rate of Change due to Advection = Flux In(i) - Flux Out(i+1) [W or kg/s]
        net_adv_H_rate = AdvFluxRate_H[i] - AdvFluxRate_H[i+1]
        net_adv_Yk_rate = AdvFluxRate_Yk[:, i] - AdvFluxRate_Yk[:, i+1]

        # Calculate d()/dt = NetRate / (rho * cp * V) or NetRate / (rho * V)
        # Need enthalpy rate, not temperature rate directly from flux balance
        # d(rho*h*V)/dt = Sum(Flux_H) => rho*V*dh/dt ~ Sum(Flux_H)
        # Use dh/dt ~ cp * dT/dt
        dT_dt_adv[i] = net_adv_H_rate / (rho_g[i] * cp_g[i] * volumes_g[i])
        dY_dt_adv[:, i] = net_adv_Yk_rate / (rho_g[i] * volumes_g[i])
        if i == 0:
            print(f"   adv  u_g_faces[i]={u_g_faces[i]:.4e} AdvFluxRate_Yk[fuel,0]={AdvFluxRate_Yk[0, i]:.4e} net_adv_Yk_rate={net_adv_Yk_rate[0]:.4e} mass_flux_rate_0={mass_flux_rate_0:.4e} dY_dt_adv[fuel,i]={dY_dt_adv[0, i]:.4e}")

    return dT_dt_adv, dY_dt_adv

# --- Functions for Implicit Diffusion Matrix Coefficients ---

def build_diffusion_coefficients_gas_T(
    rho_g: np.ndarray, cp_g: np.ndarray, lambda_g: np.ndarray, # Props at cell centers (t^n)
    r_g_centers: np.ndarray, r_g_nodes: np.ndarray, volumes_g: np.ndarray,
    Ng: int, dt: float,
    A_g_faces: np.ndarray # Face areas array (Ng+1,)
    ):
    """
    Builds the tridiagonal matrix coefficients (a, b, c) and diagonal term (diag_coeff)
    for implicit heat diffusion in the gas phase (Backward Euler).
    Boundary conditions are NOT applied here.
    """
    if Ng <= 0: return None, None, None, None
    a, b, c = np.zeros(Ng), np.zeros(Ng), np.zeros(Ng)

    # Check for NaN/Inf in inputs
    if np.any(np.isnan(rho_g)) or np.any(np.isnan(cp_g)) or np.any(np.isnan(lambda_g)):
        print("ERROR: NaN input detected in build_diffusion_coefficients_gas_T")
        return None, None, None, None

    # Diagonal term from time derivative: (rho*cp*V)/dt
    diag_coeff = rho_g * cp_g * volumes_g / dt
    if np.any(np.isnan(diag_coeff)) or np.any(diag_coeff <= 0):
        print(f"Error: Invalid diag_coeff in gas T diffusion: {diag_coeff}")
        return None, None, None, None

    # Diffusion terms D = lambda * A / dr_centers
    # Internal faces: i = 1 to Ng-1
    if Ng >= 2:
        lambda_f_internal = harmonic_mean(lambda_g[:-1], lambda_g[1:]) # lambda at face i
        dr_c_internal = r_g_centers[1:] - r_g_centers[:-1] # dist between centers i and i-1
        dr_c_internal = np.maximum(dr_c_internal, 1e-15)
        D_internal = lambda_f_internal * A_g_faces[1:Ng] / dr_c_internal
        D_internal = np.nan_to_num(D_internal)

        a[1:] = -D_internal
        c[:-1] = -D_internal
        b[1:] += D_internal
        b[:-1] += D_internal

    # Face i=0 (surface): Boundary condition handled in main loop RHS
    # D_surf is NOT added to b[0] because the BC flux is handled in RHS.
    # (Optional: Keep D_surf calculation if needed elsewhere, but don't add to b[0])
    # if Ng >= 1:
    #      lambda_face_0 = lambda_g[0]
    #      dr_centers_0 = r_g_centers[0] - r_g_nodes[0]
    #      dr_centers_0 = max(dr_centers_0, 1e-15)
    #      D_surf = lambda_face_0 * A_g_faces[0] / dr_centers_0
    #      D_surf = np.nan_to_num(D_surf)
         # b[0] += D_surf # <<<--- DELETED/COMMENTED OUT

    # Face i=Ng (outer wall): Zero flux boundary condition is implicitly handled
    # by not having a c[Ng-1] term and the calculation of D_internal up to Ng-1 for a[Ng-1]

    # Add diagonal time derivative term
    b[:] += diag_coeff[:]

    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        print("ERROR: NaN detected in gas T diffusion coefficients (a,b,c).")
        return None, None, None, None

    return a, b, c, diag_coeff

def build_diffusion_coefficients_gas_Y(
    k: int, # Species index
    rho_g: np.ndarray, Dk_g: np.ndarray, # Props at t^n [Ng], [nsp, Ng]
    r_g_centers: np.ndarray, r_g_nodes: np.ndarray, volumes_g: np.ndarray,
    Ng: int, nsp: int, dt: float,
    A_g_faces: np.ndarray, fuel_idx: int # Face areas array (Ng+1,)
    ):
    """
    Builds the tridiagonal matrix coefficients (a, b, c) and diagonal term (diag_coeff)
    for implicit species diffusion in the gas phase (Backward Euler).
    Boundary conditions are NOT applied here.
    Uses Fick's law approximation: Flux_k = - rho * Dk * grad(Yk)
    """
    if Ng <= 0: return None, None, None, None
    a, b, c = np.zeros(Ng), np.zeros(Ng), np.zeros(Ng)

    Dk_g_k = Dk_g[k, :] # Diffusion coefficient for species k at cell centers

    if np.any(np.isnan(rho_g)) or np.any(np.isnan(Dk_g_k)):
        print(f"ERROR: NaN input detected in build_diffusion_coefficients_gas_Y for k={k}")
        return None, None, None, None

    # Diagonal term from time derivative: (rho*V)/dt
    diag_coeff = rho_g * volumes_g / dt
    if np.any(np.isnan(diag_coeff)) or np.any(diag_coeff <= 0):
        print(f"Error: Invalid diag_coeff in gas Y diffusion k={k}: {diag_coeff}")
        return None, None, None, None

    # Diffusion terms D = (rho*Dk) * A / dr_centers
    # Internal faces: i = 1 to Ng-1
    if Ng >= 2:
        rhoDk_L = rho_g[:-1] * Dk_g_k[:-1]; rhoDk_R = rho_g[1:] * Dk_g_k[1:]
        # Use harmonic mean for rho*Dk at the face
        rhoDk_f_internal = harmonic_mean(rhoDk_L, rhoDk_R)
        dr_c_internal = r_g_centers[1:] - r_g_centers[:-1] # dist between centers i and i-1
        dr_c_internal = np.maximum(dr_c_internal, 1e-15)
        D_internal = rhoDk_f_internal * A_g_faces[1:Ng] / dr_c_internal
        D_internal = np.nan_to_num(D_internal)

        a[1:] = -D_internal
        c[:-1] = -D_internal
        b[1:] += D_internal
        b[:-1] += D_internal

    # Face i=0 (surface): Boundary condition handled in main loop RHS
    # Need D_surf = (rho*Dk)_face_0 * A_face_0 / dr_centers_0
    #if Ng >= 1:
    #     rhoDk_face_0 = rho_g[0] * Dk_g_k[0] # Approx rho*Dk at face 0 with value in cell 0
    #     dr_centers_0 = r_g_centers[0] - r_g_nodes[0] # Distance from first cell center to surface
    #     dr_centers_0 = max(dr_centers_0, 1e-15)
    #     D_surf = rhoDk_face_0 * A_g_faces[0] / dr_centers_0
    #     D_surf = np.nan_to_num(D_surf)
    #     b[0] += D_surf # Contribution to b_0 from face 0 (surface)

    # Face i=Ng (outer wall): Zero flux BC handled implicitly

    # Add diagonal time derivative term
    b[:] += diag_coeff[:]

    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        print(f"ERROR: NaN detected in gas Y diffusion coefficients (a,b,c) for k={k}.")
        return None, None, None, None

    # +++ デバッグ出力追加 +++
    fuel_idx_local = fuel_idx
    if k == fuel_idx_local and config.LOG_LEVEL >= 2: # 燃料種かつ十分なログレベルの場合
        print(f"      DEBUG Matrix Coeffs (Gas Y, k={k}, fuel Dk_g[{k},0]={Dk_g[k,0]:.3e}):")
        num_disp = min(3, Ng) # 表示する要素数
        print(f"        a[{1}:{num_disp+1}]: {a[1:num_disp+1]}") # a[0]は未使用
        print(f"        b[0:{num_disp}]: {b[0:num_disp]}")
        print(f"        c[0:{num_disp}]: {c[0:num_disp]}")
        # 対角優位性の簡易チェック (i=0)
        if Ng > 0:
             diag_dominance_i0 = abs(b[0]) - (abs(c[0]) if Ng > 1 else 0.0)
             is_dominant_i0 = diag_dominance_i0 >= -1e-9 # 誤差を考慮
             print(f"        Diag Dominance Check (i=0): |b0|={abs(b[0]):.3e}, |c0|={abs(c[0]) if Ng > 1 else 0.0:.3e} -> Dominant: {is_dominant_i0} (Diff: {diag_dominance_i0:.3e})")
    # +++ デバッグ出力追加ここまで +++


    return a, b, c, diag_coeff

def apply_gas_boundary_conditions(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
    Ng: int,
    boundary_type: str,
    boundary_value: float
):
    """
    気相の連立一次方程式の係数行列とRHSベクトルに境界条件を適用する。
    【修正版】最外殻(i=Ng-1)のディリクレ条件を正しく実装する。

    Args:
        a, b, c: 三重対角行列の係数
        d: 右辺ベクトル
        Ng: 気相のセル数
        boundary_type: 'CLOSED' または 'OPEN_FIXED_AIR'
        boundary_value: 境界で固定したい値 (温度または化学種質量分率)
    
    Returns:
        修正後の (a, b, c, d)
    """
    if Ng <= 0:
        return a, b, c, d

    if boundary_type == 'OPEN_FIXED_AIR':
        # --- ディリクレ境界条件の正しい適用 ---
        # 1. 最も外側のセル (i = Ng-1) の方程式を T[Ng-1] = boundary_value に置き換える。
        #    これにより、このセルの値が強制的に境界値になる。
        if Ng >= 1:
            a[Ng - 1] = 0.0
            b[Ng - 1] = 1.0  # 対角係数を1に
            c[Ng - 1] = 0.0  # (cの最後の要素は元々使われないが、念のため)
            d[Ng - 1] = boundary_value # 右辺に境界値を直接設定

        # 2. 境界の一つ内側のセル (i = Ng-2) の方程式を修正する。
        #    T[Ng-1] が固定値になったため、このセルからの熱流束はソース項として扱う。
        if Ng >= 2:
            # 元の方程式: a[Ng-2]*T[Ng-3] + b[Ng-2]*T[Ng-2] + c[Ng-2]*T[Ng-1] = d_orig[Ng-2]
            # T[Ng-1]は既知のboundary_valueなので、右辺に移項する。
            # d_new[Ng-2] = d_orig[Ng-2] - c[Ng-2] * boundary_value
            d[Ng - 2] -= c[Ng - 2] * boundary_value
            
            # T[Ng-1]への依存性はなくなったので、係数行列から切り離す。
            c[Ng - 2] = 0.0

    elif boundary_type == 'CLOSED':
        # 閉鎖境界（ゼロ勾配）の場合は何もしない。
        pass
    
    else:
        raise ValueError(f"Unknown OUTER_BOUNDARY_TYPE: {boundary_type}")
        
    return a, b, c, d