# =========================================
#     main_operator_splitting_r4.py
# =========================================
import numpy as np
import cantera as ct
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import scipy.optimize as op

# Import simulation modules
import config
import grid
import properties
import interface
import reactions
import numerics
import liquid_phase
import gas_phase
import plotting
import output
import engine
import turbulence
import shutil

# Global variables (optional, could be managed within a class)
step_count_total = 0
# Store final converged interface values from the previous step
mdot_prev_step = 0.0
q_liq_out_surf_prev_step = 0.0
q_gas_out_surf_prev_step = 0.0
v_g_surf_prev_step = 0.0
T_s_prev_step = config.T_L_INIT + 50 # Initialize with liquid temp
Y_eq_prev_step = None # Will be calculated

def run_simulation_split():
    """Runs the droplet simulation using operator splitting."""
    global step_count_total, mdot_prev_step, q_liq_out_surf_prev_step, \
           q_gas_out_surf_prev_step, v_g_surf_prev_step, T_s_prev_step, Y_eq_prev_step

    # --- 出力ディレクトリとファイルパスの準備 ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    csv_filename = os.path.join(config.OUTPUT_DIR, 'time_history_live.csv')
    restart_filepath = os.path.join(config.OUTPUT_DIR, config.RESTART_FILE)

    # --- Initialization ---
    print("Initializing simulation (Operator Splitting r12)...")
    start_time_init = time.time()
    try:
        gas_props = properties.GasProperties(config.MECH_FILE, config.USE_RK_EOS)
        fuel_molar_mass = gas_props.molecular_weights[gas_props.fuel_idx] / 1000.0
        liquid_props = properties.LiquidProperties(config.LIQUID_PROP_FILE,
                                                   fuel_molar_mass=fuel_molar_mass)
        fugacity_interpolator = properties.FugacityInterpolator(
            filename=config.FUGACITY_MAP_FILE
        )
        engine_model = engine.EngineCompressionModel()
        turbulence_model = turbulence.TurbulenceModel()
        print(f"Mechanism: {config.MECH_FILE}, Liquid props: {config.LIQUID_PROP_FILE}")
        print(f"RK EOS: {config.USE_RK_EOS}, Diffusion: {config.DIFFUSION_OPTION}, Advection: {config.ADVECTION_SCHEME}")
        print(f"Reaction Type: {config.REACTION_TYPE}")
        print(f"Log Level: {config.LOG_LEVEL}")
        print(f"Interface Iteration: Enabled (MaxIter={config.MAX_INTERFACE_ITER}, TolT={config.INTERFACE_ITER_TOL_T:.1e}K, TolMdotRel={config.INTERFACE_ITER_TOL_MDOT:.1e}, Relax={config.INTERFACE_RELAX_FACTOR:.2f})")
        print(f"Reaction Cutoff: Enabled={config.ENABLE_REACTION_CUTOFF}, Tmin={config.REACTION_CALC_MIN_TEMP}K, Xf_min={config.REACTION_CALC_MIN_FUEL_MOL_FRAC:.1e}")
    except Exception as e: print(f"CRITICAL ERROR during property initialization: {e}"); return

    nsp = gas_props.nsp
    fuel_idx = gas_props.fuel_idx
    Nl, Ng, R0 = config.NL, config.NG, config.R0
    V_vessel = (4.0/3.0) * np.pi * config.RMAX**3

    # --- グローバルで使用する初期値を設定 ---
    # initial_Y_arrayは常に同じ値なので、ここで一度だけ定義する
    try:
        gas_props.set_state(config.T_INF_INIT, config.P_INIT, config.X_INF_INIT)
        initial_Y_array = gas_props.gas.Y.copy()
    except Exception as e:
        print(f"CRITICAL ERROR setting initial_Y_array: {e}")
        return

    # --- 変数の初期化 ---
    current_t = 0.0
    dt = config.DT_INIT
    step_count_total = 0
    output_count_total = 1
    is_gas_phase_only = False
    is_just_evaporated = False
    status = "Running"
    delta_nu_prev_step = 1e-3 # (適当な値)

    rho_l_avg_prev_step = liquid_props.get_prop('density', config.T_L_INIT)

    # --- リスタート処理 ---
    if config.USE_RESTART and os.path.exists(restart_filepath):
        print(f"--- Attempting to restart from '{restart_filepath}' ---")
        try:
            with np.load(restart_filepath) as data:
                current_t = data['current_t'].item()
                dt = data['dt'].item()
                step_count_total = data['step_count_total'].item()
                R = data['R'].item()
                P = data['P'].item()
                T_l = data['T_l']
                T_g = data['T_g']
                Y_g = data['Y_g']
                is_gas_phase_only = data['is_gas_phase_only'].item()
                mdot_prev_step = data['mdot_prev_step'].item()
                T_s_prev_step = data['T_s_prev_step'].item()
                Y_eq_prev_step = data['Y_eq_prev_step']

                if 'rho_l_avg_prev_step' in data:
                    rho_l_avg_prev_step = data['rho_l_avg_prev_step'].item()

            # リスタート時にCSVファイルから再開時刻以降の行を削除
            if os.path.exists(csv_filename):
                df_log = pd.read_csv(csv_filename)
                df_log = df_log[df_log['Time (s)'] <= current_t]
                df_log.to_csv(csv_filename, index=False, float_format='%.6e')
                print(f"Trimmed '{csv_filename}' to restart time.")

            print(f"Successfully loaded state from t = {current_t:.4e} s")
        except Exception as e:
            print(f"Warning: Error loading restart file: {e}. Starting a new simulation.")
            step_count_total = 0 # エラー時は新規実行とする

    if step_count_total == 0: # 新規実行またはリスタート失敗時
        print("--- Initializing new simulation ---")
        T_l = np.full(Nl, config.T_L_INIT)
        T_g = np.full(Ng, config.T_INF_INIT)
        Y_g = np.zeros((nsp, Ng))
        Y_g[:, :] = initial_Y_array[:, np.newaxis] # 上で定義した変数を使用
        R, P = R0, config.P_INIT
        _, Y_eq_prev_step = interface.calculate_fuel_mass_fraction_surf(T_l[-1] if Nl > 0 else T_g[0], P, gas_props, liquid_props, fugacity_interpolator)
        current_t = 0.0
        dt = config.DT_INIT

        # 新規実行時にCSVファイルを初期化
        header_df = pd.DataFrame(columns=['Time (s)', 'Radius (m)', 'Pressure (Pa)', 'T_liquid_surf_cell (K)', 'T_gas_surf_cell (K)', 'T_solved_interface (K)', 'Mdot (kg/m2/s)', 'MaxGasTemp (K)'])
        header_df.to_csv(csv_filename, index=False)
        print(f"Created new CSV log file: {csv_filename}")
        # t=0 の状態をCSVに書き込み
        init_data = pd.DataFrame({
            'Time (s)': [0.0], 'Radius (m)': [R0], 'Pressure (Pa)': [config.P_INIT],
            'T_liquid_surf_cell (K)': [T_l[-1] if Nl > 0 else np.nan],
            'T_gas_surf_cell (K)': [T_g[0] if Ng > 0 else np.nan],
            'T_solved_interface (K)': [T_s_prev_step], 'Mdot (kg/m2/s)': [0.0],
            'MaxGasTemp (K)': [np.max(T_g) if Ng > 0 else np.nan]
        })
        init_data.to_csv(csv_filename, mode='a', header=False, index=False, float_format='%.6e')

    # --- 保存用リストと次の保存時刻の初期化 ---
    # メモリ上のリストは最終的なプロット用に依然として使用
    saved_times = [current_t]
    saved_results_list = [{'T_l': T_l.copy(), 'T_g': T_g.copy(), 'Y_g': Y_g.copy(), 'R': R, 'P': P}]
    next_save_time = (np.floor(current_t / config.SAVE_INTERVAL_DT) + 1) * config.SAVE_INTERVAL_DT
    
    # 時間ベースのプロファイル出力のための次の出力時刻を初期化
    next_profile_output_time = (np.floor(current_t / config.OUTPUT_TIME_INTERVAL) + 1) * config.OUTPUT_TIME_INTERVAL
    if current_t == 0.0 and config.SAVE_RADIAL_PROFILES:
        # t=0の初期状態も保存する
        output.save_radial_profile_to_csv(current_t, step_count_total, R, T_l, T_g, Y_g, gas_props)

    end_time_init = time.time()
    print(f"Initialization complete ({end_time_init - start_time_init:.2f} s)")
    if config.USE_ADAPTIVE_DT: print(f"Using Adaptive time step starting near dt = {dt:.2e} s (CFL={config.CFL_NUMBER})")
    else: print(f"Using fixed time step dt = {dt:.2e} s")

    # --- Canteraリアクターの準備 ---
    reactor = ct.IdealGasReactor(gas_props.gas)
    reactor_net = ct.ReactorNet([reactor])

    start_time_loop = time.time()
    print("-" * 60)
    print(f"Starting time integration from t = {current_t:.2e} up to t = {config.T_END:.2e} s...")
    print("-" * 60)

    # --- Main Time Integration Loop ---
    while current_t < config.T_END and status == "Running":
        step_start_time = time.time()
        if config.LOG_LEVEL >= 1: print(f"\n--- Step {step_count_total + 1}, t={current_t:.4e}, dt={dt:.2e} ---")

        # --- 0. Store state at t^n ---
        T_l_n = T_l.copy(); T_g_n = T_g.copy(); Y_g_n = Y_g.copy(); R_n = R; P_n = P

        # --- 1. Calculate Grid, Properties, dt (using state at t^n) ---
        R_safe = max(R_n, config.R_TRANSITION_THRESHOLD * 0.5) if is_gas_phase_only else max(R_n, 1e-9)
        try:
            r_l_centers, r_l_nodes, volumes_l = grid.liquid_grid_fvm(R_safe, Nl)
            r_g_centers, r_g_nodes, volumes_g = grid.gas_grid_fvm(R_safe, config.RMAX, Ng)
            A_g_faces = grid.face_areas(r_g_nodes)
            A_l_faces = grid.face_areas(r_l_nodes)
            if config.LOG_LEVEL >=1 and step_count_total % 10 == 0 : # 既存のログ出力と同様の条件
                 print(f"  volumes_g:Step1 volumes_g[0]={volumes_g[0]:.4e} volumes_g[-1]={volumes_g[-1]:.4e}")
        except Exception as e: print(f"ERROR grid gen R={R_safe:.2e}: {e}"); status="Grid Error"; break

        # ==============================================================
        # <<< グローバル質量保存チェック: ステップ開始時の気相総質量計算 >>>
        # ==============================================================
        M_gas_start_of_step = 0.0
        if Ng > 0:
            for i_mgas in range(Ng):
                # P_n (ステップ開始時の圧力) と、ステップ開始時の T_g_n, Y_g_n を使用
                density_at_cell_start = gas_props.get_density(T_g_n[i_mgas], P_n, Y_g_n[:, i_mgas])
                if not (np.isnan(density_at_cell_start) or density_at_cell_start <= 0):
                    M_gas_start_of_step += density_at_cell_start * volumes_g[i_mgas] # volumes_g は現在の R_n に基づく
                else:
                    print(f"    WARNING: Invalid density for M_gas_start_of_step at cell {i_mgas}, T={T_g_n[i_mgas]:.1f}, P={P_n:.2e}")
        if config.LOG_LEVEL >= 1: # ログレベル1以上で表示
            print(f"    DEBUG Mass: M_gas_start_of_step = {M_gas_start_of_step:.6e} kg (at t={current_t:.4e} with P_n={P_n:.3e})")
        # ==============================================================

        total_mass_gas_before_step = 0
        if Ng > 0:
            for i in range(Ng):
                # 注意: P_n (ステップ開始時の圧力) を使って密度を計算
                density_at_cell = gas_props.get_density(T_g_n[i], P_n, Y_g_n[:, i]) # T_g_n, Y_g_n はステップ開始時の値
                if not (np.isnan(density_at_cell) or density_at_cell <= 0):
                    total_mass_gas_before_step += density_at_cell * volumes_g[i] # volumes_g はステップ開始時のもの
                else:
                    # エラー処理
                    pass

        # get turbulence properties
        k_t, epsilon_t = turbulence_model.get_turbulence_properties(current_t)

        # Calculate properties based on T^n, Y^n, P^n
        rho_g_n = np.zeros(Ng); rho_l_n = np.zeros(Nl)
        cp_g_n = np.zeros(Ng); cp_l_n = np.zeros(Nl)
        lambda_g_n = np.zeros(Ng); lambda_l_n = np.zeros(Nl)
        Dk_g_n = np.zeros((nsp, Ng)); h_k_g_n = np.zeros((nsp, Ng))
        h_g_n = np.zeros(Ng); h_l_n = np.zeros(Nl)

        lambda_g_n_eff = np.zeros(Ng) # <<< 実効値用の配列を準備
        mu_g_n_eff = np.zeros(Ng)     # <<< 実効値用の配列を準備
        Dk_g_n_eff = np.zeros((nsp, Ng)) # <<< 実効値用の配列を準備

        valid_state = True
        for i in range(Ng):
             T_g_cell = max(200.0, min(T_g_n[i], 5000.0))
             rho_g_n[i] = gas_props.get_density(T_g_cell, P_n, Y_g_n[:,i])
             cp_g_n[i] = gas_props.get_cp_mass(T_g_cell, P_n, Y_g_n[:,i])
             lambda_g_n[i] = gas_props.get_thermal_conductivity(T_g_cell, P_n, Y_g_n[:,i])
             Dk_g_n[:, i] = gas_props.get_diffusion_coeffs(T_g_cell, P_n, Y_g_n[:,i])
             h_k_g_n[:, i] = gas_props.get_partial_enthalpies_mass(T_g_cell, P_n, Y_g_n[:, i])
             h_g_n[i] = gas_props.get_enthalpy_mass(T_g_cell, P_n, Y_g_n[:, i])

             lambda_eff_i, mu_eff_i, Dk_eff_i = gas_props.get_effective_transport_properties(
                 T_g_cell, P_n, Y_g_n[:,i],
                 k_t, epsilon_t,
                 R_n, r_g_centers[i],
                 delta_nu_prev_step)
             lambda_g_n_eff[i] = lambda_eff_i
             mu_g_n_eff[i] = mu_eff_i
             Dk_g_n_eff[:, i] = Dk_eff_i

             if np.isnan(rho_g_n[i]) or rho_g_n[i] <= 0 or np.isnan(cp_g_n[i]) or cp_g_n[i] <= 0 or np.isnan(lambda_g_n[i]) or np.any(np.isnan(Dk_g_n[:,i])) or np.any(np.isnan(h_k_g_n[:,i])) or np.isnan(h_g_n[i]):
                 print(f"Error: NaN property at gas cell {i}, T={T_g_cell:.1f}")
                 valid_state = False; break
        for j in range(Nl):
             T_l_cell = max(200.0, min(T_l_n[j], 2000.0))
             props_l = liquid_props.get_properties(T_l_cell)
             rho_l_n[j] = props_l.get('density', np.nan); cp_l_n[j] = props_l.get('specific_heat', np.nan); lambda_l_n[j] = props_l.get('thermal_conductivity', np.nan); h_l_n[j] = props_l.get('enthalpy', np.nan)
             if np.isnan(rho_l_n[j]) or rho_l_n[j] <= 0 or np.isnan(cp_l_n[j]) or cp_l_n[j] <= 0 or np.isnan(lambda_l_n[j]) or np.isnan(h_l_n[j]):
                 print(f"Error: NaN property at liquid cell {j}, T={T_l_cell:.1f}")
                 valid_state=False; break
        if not valid_state: print("Error getting props for dt calc."); status="Prop Error"; break

        # --- Calculate dt (using properties at n, velocity from n-1) ---
        # Estimate velocity field for dt calculation based on previous mdot
        u_g_faces_dt = np.zeros(Ng + 1)
        u_g_faces_dt[0] = v_g_surf_prev_step # From prev step converged val
        mass_flux_rate_prev = mdot_prev_step * A_g_faces[0]
        for i in range(1, Ng):
            rho_face_i = numerics.arithmetic_mean(rho_g_n[i-1], rho_g_n[i])
            if A_g_faces[i] > 1e-15 and rho_face_i > 1e-6:
                # Continuity: (rho * u * A)_i = (rho * u * A)_i-1 => mass_flux_rate_prev
                u_g_faces_dt[i] = mass_flux_rate_prev / (rho_face_i * A_g_faces[i])
            else:
                u_g_faces_dt[i] = 0.0
        u_g_faces_dt[Ng] = 0.0 # Wall BC

        dt = numerics.calculate_adaptive_dt(
             u_g_faces_dt, lambda_g_n, rho_g_n, cp_g_n, Dk_g_n,
             lambda_l_n, rho_l_n, cp_l_n,
             r_g_nodes, r_l_nodes, dt, nsp,
             is_gas_only_phase=is_gas_phase_only) # Pass current dt
        if is_just_evaporated:
            dt = min(dt, config.DT_TRANSITION_LIMIT)
            if config.LOG_LEVEL >= 1:
                print(f"      NOTE: Applying strict transition dt for one step: {dt:.2e} s")
            # この処理は1ステップのみ適用するため、すぐにフラグを降ろす
            is_just_evaporated = False

        dt = min(dt, config.T_END - current_t + 1e-15) # Prevent overshooting T_END
        if dt < config.DT_MIN_VALUE: status = "dt too small"; break
        if status != "Running": break # Exit if error in grid/prop/dt calc

        if config.LOG_LEVEL >= 1:
            print(f"  [Step 1 Done] dt calculated: {dt:.3e} s")
            # Add basic property checks if needed
            print(f"    rho_g range: [{np.min(rho_g_n):.2e}, {np.max(rho_g_n):.2e}]")
            print(f"    lambda_g range: [{np.min(lambda_g_n):.2e}, {np.max(lambda_g_n):.2e}]")

        M_gas_before_advection = 0.0
        if Ng > 0:
            # T_g_n, Y_g_n, P_n, volumes_g (R_nベース) を使用
            for i_mgas in range(Ng):
                density_val = gas_props.get_density(T_g_n[i_mgas], P_n, Y_g_n[:, i_mgas])
                if not (np.isnan(density_val) or density_val <= 0):
                    M_gas_before_advection += density_val * volumes_g[i_mgas]
        if config.LOG_LEVEL >= 2:
            print(f"    DEBUG Advection: M_gas_before_advection = {M_gas_before_advection:.6e} kg")

        # --- 2. Explicit Advection Step: phi* = phi^n + dt * Adv(phi^n) ---
        # Calculate gas velocity field based on mdot_prev_step and continuity at t^n
        u_g_faces_adv = np.zeros(Ng + 1)
        u_g_faces_adv[0] = v_g_surf_prev_step # Velocity at droplet surface (BC)
        mass_flux_rate_adv = mdot_prev_step * A_g_faces[0]
        for i in range(1, Ng):
            rho_face_i = numerics.arithmetic_mean(rho_g_n[i-1], rho_g_n[i])
            if A_g_faces[i] > 1e-15 and rho_face_i > 1e-6:
                u_g_faces_adv[i] = mass_flux_rate_adv / (rho_face_i * A_g_faces[i])
            else:
                u_g_faces_adv[i] = 0.0
        u_g_faces_adv[Ng] = 0.0 # Wall BC

        dTg_dt_adv, dYg_dt_adv = gas_phase.calculate_gas_advection_rhs(
             T_g_n, Y_g_n, rho_g_n, cp_g_n, h_k_g_n, u_g_faces_adv,
             r_g_centers, r_g_nodes, volumes_g, Ng, nsp,
             mdot_prev_step, # Use mdot from previous time step for explicit advection
             Y_eq_prev_step,    # Use Y_eq from previous time step
             gas_props, P_n
             )
        # Advection in liquid is neglected
        T_l_star = T_l_n.copy()
        # Update gas phase explicitly
        T_g_star = T_g_n + dt * dTg_dt_adv
        Y_g_star = Y_g_n + dt * dYg_dt_adv
        # Ensure mass fractions are physical after advection
        Y_g_star = np.maximum(Y_g_star, 0.0)
        sum_Yg_star = np.sum(Y_g_star, axis=0)
        mask_star = sum_Yg_star > 1e-9
        Y_g_star[:, mask_star] /= sum_Yg_star[mask_star]
        if np.any(~mask_star): Y_g_star[:, ~mask_star] = 0.0; Y_g_star[gas_props.n2_idx if gas_props.n2_idx>=0 else 0, ~mask_star] = 1.0

        if config.LOG_LEVEL >= 1:
            print(f"  [Step 2 Done] Advection calculated.")
            if np.any(np.isnan(T_g_star)) or np.any(np.isnan(Y_g_star)):
                print("    ERROR: NaN detected after Advection step!")
                print(f"    T_g_star range: [{np.min(T_g_star):.1f}, {np.max(T_g_star):.1f}] K")
                status = "NaN Error Advection"; break

        # ==============================================================
        # <<< ログ追加箇所 2: 移流による温度変化ログ >>>
        if config.LOG_LEVEL >= 2 and Ng > 0:
            delta_T_adv = T_g_star[0] - T_g_n[0] # セル0の温度変化量
            print(f"      DEBUG Advection dT (i=0): {delta_T_adv:+.3e} K (T_n={T_g_n[0]:.2f} -> T*={T_g_star[0]:.2f})")
        # ==============================================================
        print(f"      test Advection Yf(i=0): {Y_g_star[fuel_idx, 0]:+.3e}")


        M_gas_after_advection = 0.0
        if Ng > 0:
            # T_g_star, Y_g_star, P_n, volumes_g (R_nベース) を使用
            # 密度計算のための圧力は、まだ更新されていない P_n を使う
            for i_mgas in range(Ng):
                density_val = gas_props.get_density(T_g_star[i_mgas], P_n, Y_g_star[:, i_mgas])
                if not (np.isnan(density_val) or density_val <= 0):
                    M_gas_after_advection += density_val * volumes_g[i_mgas]

        expected_mass_influx_at_interface_adv = mdot_prev_step * A_g_faces[0] * dt # mdot_prev_step は interface で計算された今回の値

        if config.LOG_LEVEL >= 2:
            print(f"    DEBUG Advection: M_gas_after_advection = {M_gas_after_advection:.6e} kg")
            print(f"    DEBUG Advection: Change in M_gas (Advection) = {(M_gas_after_advection - M_gas_before_advection):.3e} kg")
            print(f"    DEBUG Advection: Expected mass influx at interface (mdot*A*dt) = {expected_mass_influx_at_interface_adv:.3e} kg")
            # 内部の流束による変化も考慮する必要があるため、単純比較は難しいが、傾向は掴める。
            # 特に、(M_gas_after_advection - M_gas_before_advection) が期待値よりも著しく大きいか小さいかに注目。

        # --- 3. Interface/Diffusion Iteration Loop: phi** from phi* ---
        iter_count = 0; converged = False
        T_s_iter_prev = T_s_prev_step # Use previous converged Ts as initial guess
        mdot_iter_prev = mdot_prev_step # Use previous converged mdot

        T_l_iter = T_l_star.copy(); T_g_iter = T_g_star.copy(); Y_g_iter = Y_g_star.copy()
        q_liq_out_surf_iter = q_liq_out_surf_prev_step # Initialize with previous step's value
        q_gas_out_surf_iter = q_gas_out_surf_prev_step # Initialize with previous step's value

        # --- Build Diffusion Coefficients (based on properties at t^n) ---
        # These coefficients (a,b,c) remain constant during the iteration
        a_l, b_l, c_l, diag_l = None, None, None, None
        a_gT, b_gT, c_gT, diag_gT = None, None, None, None
        a_gY, b_gY, c_gY, diag_gY = [], [], [], []
        matrix_build_ok = True
        if Nl > 0:
            a_l, b_l, c_l, diag_l = liquid_phase.build_diffusion_coefficients_liquid(
                r_l_centers, r_l_nodes, volumes_l, liquid_props, Nl, dt, T_l_n # Props based on T_n
            )
            if a_l is None: matrix_build_ok = False
        if Ng > 0:
            #a_gT, b_gT, c_gT, diag_gT = gas_phase.build_diffusion_coefficients_gas_T(
            #    rho_g_n, cp_g_n, lambda_g_n, r_g_centers, r_g_nodes, volumes_g, Ng, dt, A_g_faces # Props based on t_n
            #)
            a_gT, b_gT, c_gT, diag_gT = gas_phase.build_diffusion_coefficients_gas_T(
                rho_g_n, cp_g_n, lambda_g_n_eff, r_g_centers, r_g_nodes, volumes_g, Ng, dt, A_g_faces # Props based on t_n
            )

            if a_gT is None: matrix_build_ok = False
            for k in range(nsp):
                #a_gy_k, b_gy_k, c_gy_k, diag_gy_k = gas_phase.build_diffusion_coefficients_gas_Y(
                #    k, rho_g_n, Dk_g_n, r_g_centers, r_g_nodes, volumes_g, Ng, nsp, dt, A_g_faces # Props based on t_n
                #)
                a_gy_k, b_gy_k, c_gy_k, diag_gy_k = gas_phase.build_diffusion_coefficients_gas_Y(
                    k, rho_g_n, Dk_g_n_eff, r_g_centers, r_g_nodes, volumes_g, Ng, nsp, dt, A_g_faces, fuel_idx # Props based on t_n
                )

                if a_gy_k is None: matrix_build_ok = False; break
                a_gY.append(a_gy_k); b_gY.append(b_gy_k); c_gY.append(c_gy_k); diag_gY.append(diag_gy_k)
        if not matrix_build_ok: print("ERROR building diffusion matrices."); status="Matrix Coeff Error"; break

        # ==============================================================
        # <<< ログ追加箇所 1: 気相温度の係数行列ログ >>>
        if config.LOG_LEVEL >= 2 and Ng > 0:
            print("      DEBUG Matrix Coeffs (Gas T):")
            num_disp = min(3, Ng) # 表示する要素数（最大3）
            # a_gT は a_gT[0] を使わないので 1 から表示
            print(f"        a_gT[1:{num_disp+1}]: {a_gT[1:num_disp+1]}")
            print(f"        b_gT[0:{num_disp}]: {b_gT[0:num_disp]}")
            # c_gT は c_gT[Ng-1] を使わないので Ng-1 まで表示可能だが、ここでは先頭のみ表示
            print(f"        c_gT[0:{num_disp}]: {c_gT[0:num_disp]}")
            # 対角優位性の簡易チェック (セル0について)
            if Ng >= 1 and b_gT[0] > 0:
                 # セル0の方程式では a[0] は関与しない
                 off_diag_sum_0 = abs(c_gT[0]) if Ng > 1 else 0.0
                 print(f"        Diag Dominance Check (i=0): b[0]={b_gT[0]:.3e}, |c[0]|={off_diag_sum_0:.3e} -> Ratio={abs(b_gT[0]) / (off_diag_sum_0 + 1e-15):.1f}")
            elif Ng >= 1:
                 print(f"        WARNING: b_gT[0] is not positive: {b_gT[0]:.3e}")
        # ==============================================================

        if status != "Running": break # Exit if matrix build failed

        if config.LOG_LEVEL >= 1: print(f"  [Step 3 Start] Starting Interface/Diffusion Iteration...")
        T_s_final_prev_iter = T_s_iter_prev # ループ開始時の値で初期化 (T_s_prev_step)
        mdot_final_prev_iter = mdot_iter_prev # ループ開始時の値で初期化 (mdot_prev_step)

        # --- Start Iteration ---
        while iter_count < config.MAX_INTERFACE_ITER and not converged:
            iter_start_time = time.time()

            # --- 3a. Solve Interface Conditions (using T/Y from *previous* iteration) ---
            mdot_final_iter, q_gas_out_surf_final_iter, q_liq_out_surf_final_iter = 0.0, 0.0, 0.0
            v_g_surf_final_iter = 0.0; Y_eq_final_iter = np.copy(initial_Y_array); T_s_final_iter = T_s_iter_prev

            if not is_gas_phase_only:
                T_l_last_cell_iter = T_l_iter[Nl-1] if Nl > 0 else config.T_L_INIT
                T_g_first_cell_iter = T_g_iter[0] if Ng > 0 else config.T_INF_INIT
                T_g_second_cell_iter = T_g_iter[1] if Ng >= 2 else T_g_first_cell_iter # Ng<2の場合はg0の値で代用

                Y_g_first_cell_iter = Y_g_iter[:, 0] if Ng > 0 else initial_Y_array.copy()
                Y_g_second_cell_iter = Y_g_iter[:, 1] if Ng >= 2 else Y_g_first_cell_iter # Ng<2の場合はg0の値で代用

                r_l_last_c = r_l_centers[Nl-1] if Nl >= 1 else 0.0
                r_g_0_c = r_g_centers[0] if Ng > 0 else R_n + 1e-7
                r_g_1_c = r_g_centers[1] if Ng >= 2 else r_g_0_c + 1e-7 # Ng<2の場合はダミーの値

                try:
                     # Calculate raw interface values based on current T_l, T_g, Y_g
                     mdot_calc, q_gas_out_calc, q_liq_out_calc, v_g_surf_calc, Y_eq_calc, T_s_calc, delta_nu_final_iter = interface.solve_interface_conditions(
                         T_l_node_last=T_l_last_cell_iter, 
                         T_g_node0=T_g_first_cell_iter, T_g_node1=T_g_second_cell_iter, # 追加
                         Y_g_node0=Y_g_first_cell_iter, Y_g_node1=Y_g_second_cell_iter, # 追加
                         P=P_n, R=R_n,
                         r_l_node_last_center=r_l_last_c, 
                         r_g_node0_center=r_g_0_c, r_g_node1_center=r_g_1_c, # 追加
                         gas_props=gas_props, liquid_props=liquid_props, Nl=Nl, Ng=Ng, # Ngを追加
                         T_s_previous_step=T_s_iter_prev,
                         initial_Y_array=initial_Y_array,
                         fugacity_interpolator=fugacity_interpolator,
                         k=k_t, epsilon=epsilon_t
                     )
                except Exception as e: print(f"\nERROR interface solve iter {iter_count+1}: {e}"); status = "Interface Error"; break

                # Apply relaxation to the *newly calculated* values using *previous converged* values
                alpha = config.INTERFACE_RELAX_FACTOR
                T_s_final_iter = alpha * T_s_calc + (1.0 - alpha) * T_s_iter_prev
                mdot_final_iter = alpha * mdot_calc + (1.0 - alpha) * mdot_iter_prev
                # Recompute consistent fluxes and Y_eq based on relaxed T_s and mdot
                _, Y_eq_final_iter = interface.calculate_fuel_mass_fraction_surf(T_s_final_iter, P_n, gas_props, liquid_props, fugacity_interpolator)
                rho_s_relaxed = gas_props.get_density(T_s_final_iter, P_n, Y_eq_final_iter)
                v_g_surf_final_iter = mdot_final_iter / rho_s_relaxed if rho_s_relaxed > 1e-9 else 0.0
                # Recompute fluxes using the *relaxed* T_s
                # Re-calculate gradients based on T_s_final_iter
                dr_face_center_g = r_g_0_c - R_n; dr_face_center_l = R_n - r_l_last_c if Nl>1 else np.inf
                grad_Tg_relaxed = (T_g_first_cell_iter - T_s_final_iter) / dr_face_center_g if dr_face_center_g > 1e-15 else 0.0
                grad_Tl_relaxed = (T_s_final_iter - T_l_last_cell_iter) / dr_face_center_l if Nl>0 and dr_face_center_l > 1e-15 else 0.0
                # Properties at interface face
                lambda_g_s_relaxed = gas_props.get_thermal_conductivity(T_s_final_iter, P_n, Y_eq_final_iter)
                lambda_g_0_iter = gas_props.get_thermal_conductivity(T_g_first_cell_iter, P_n, Y_g_first_cell_iter)
                lambda_g_face_relaxed = numerics.harmonic_mean(lambda_g_s_relaxed, lambda_g_0_iter)
                lambda_l_s_relaxed = liquid_props.get_prop('thermal_conductivity', T_s_final_iter)
                lambda_l_last_iter = liquid_props.get_prop('thermal_conductivity', T_l_last_cell_iter) if Nl > 0 else 1e-3
                lambda_l_face_relaxed = numerics.harmonic_mean(lambda_l_s_relaxed, lambda_l_last_iter) if Nl > 0 else lambda_l_s_relaxed

                q_gas_out_surf_final_iter = -lambda_g_face_relaxed * grad_Tg_relaxed
                q_liq_out_surf_final_iter = -lambda_l_face_relaxed * grad_Tl_relaxed

                # Convergence Check (Compare *calculated* value with *previous iteration's converged* value)
                # Use change between consecutive *relaxed* values for convergence check
                T_s_change = abs(T_s_final_iter - T_s_final_prev_iter)
                mdot_rel_change = abs(mdot_final_iter - mdot_final_prev_iter) / (abs(mdot_final_prev_iter) + 1e-9)

                # Log the change between relaxed values
                print(f"      Iter {iter_count+1}: Ts={T_s_final_iter:.2f}K (dTs_iter={T_s_change:.1e}), mdot={mdot_final_iter:.2e} (dRelM_iter={mdot_rel_change:.1e})")
                ###print(f"      Iter {iter_count+1}: T_s_calc={T_s_calc:.2f}K T_s_iter_prev={T_s_iter_prev:.2f}K T_s_final_iter={T_s_final_iter:.2f}K T_l_last_cell_iter={T_l_last_cell_iter:.2f} q_liq_out_surf_final_iter={q_liq_out_surf_final_iter:.2f} q_gas_out_surf_final_iter={q_gas_out_surf_final_iter:.2f} mdot_final_iter={mdot_final_iter:3e}")

                ###if iter_count > 0 and T_s_change < config.INTERFACE_ITER_TOL_T and mdot_rel_change < config.INTERFACE_ITER_TOL_MDOT:
                if T_s_change < config.INTERFACE_ITER_TOL_T and mdot_rel_change < config.INTERFACE_ITER_TOL_MDOT:
                    converged = True
                    if config.LOG_LEVEL >= 1: # Make log level check consistent
                        print(f"      Iter {iter_count+1}: Converged! Ts={T_s_final_iter:.2f}K (dTs_iter={T_s_change:.1e}), mdot={mdot_final_iter:.2e} (dRelM_iter={mdot_rel_change:.1e})")

            else: # Gas phase only mode
                mdot_final_iter = 0.0; q_gas_out_surf_final_iter = 0.0; q_liq_out_surf_final_iter = 0.0
                v_g_surf_final_iter = 0.0; Y_eq_final_iter = Y_g_iter[:, 0].copy() if Ng > 0 else initial_Y_array
                T_s_final_iter = T_g_iter[0] if Ng > 0 else config.T_INF_INIT
                converged = True # No iteration needed

            if status != "Running": break
            ###if converged: break # Exit iteration loop

            # --- 3b. Implicit Diffusion Step (using *relaxed/finalized* interface conditions from this iteration) ---
            # Solve A * T_l_new_iter = d_l (where d_l depends on T_l_iter and BC)
            if Nl > 0 and not is_gas_phase_only:
                 # Let's assume build_diffusion_matrix includes the BC part now (as per test_liquid_phase)
                 # Check build_diffusion_matrix_liquid again... It does NOT apply BCs.
                 # Correct RHS assembly:
                 d_l_rhs = diag_l * T_l_iter # Use current iter T_l
                 d_l_rhs[Nl-1] += (-q_liq_out_surf_final_iter) * A_l_faces[Nl] # Add flux rate to RHS
                 ###d_l_rhs[Nl-1] += q_liq_out_surf_final_iter * A_l_faces[Nl] # Add flux rate to RHS

                 T_l_new_iter = numerics.solve_tridiagonal(a_l, b_l, c_l, d_l_rhs)
                 if T_l_new_iter is None: status = "Matrix Solve Error Liq"; break
                 T_l_iter = T_l_new_iter # Update for next iteration
            else: T_l_iter = np.array([])

            # Solve A * T_g_new_iter = d_gT
            if Ng > 0:
                 if config.OUTER_BOUNDARY_TYPE == 'OPEN_FIXED_AIR':
                    # --- エンジン計算モード (開放境界) のロジック ---

                    # 1. この時間ステップでの境界値と圧力変化率を取得
                    P_current, _ = engine_model.get_ambient_conditions(current_t)
                    P_new, T_amb_new = engine_model.get_ambient_conditions(current_t + dt)
                    dPdt = (P_new - P_current) / dt if dt > 1e-12 else 0.0
                    
                    # 2. 周囲空気の組成（質量分率）を準備
                    gas_props.set_state(T_amb_new, P_new, config.X_INF_INIT)
                    Y_k_air_boundary = gas_props.gas.Y.copy()

                    # 3. 温度場を解く
                    d_gT_rhs = diag_gT * T_g_iter
                    d_gT_rhs[0] += q_gas_out_surf_final_iter * A_g_faces[0] # 界面からの熱流束

                    # ★★★★★ 圧力仕事項をソース項として追加 ★★★★★
                    pressure_work_source = volumes_g * dPdt
                    d_gT_rhs += pressure_work_source
                    
                    a_gT_bc, b_gT_bc, c_gT_bc, d_gT_rhs_bc = gas_phase.apply_gas_boundary_conditions(
                        a_gT.copy(), b_gT.copy(), c_gT.copy(), d_gT_rhs.copy(), Ng,
                        boundary_type=config.OUTER_BOUNDARY_TYPE,
                        boundary_value=T_amb_new
                    )
                    T_g_new_iter = numerics.solve_tridiagonal(a_gT_bc, b_gT_bc, c_gT_bc, d_gT_rhs_bc)
                    if T_g_new_iter is None: status = "Matrix Solve Error Gas T"; break
                    T_g_iter = T_g_new_iter

                    # 4. 化学種場を解く
                    Y_g_new_iter = np.zeros_like(Y_g_iter)
                    for k in range(nsp):
                        d_gy_k_rhs = diag_gY[k] * Y_g_iter[k,:]
                        if not is_gas_phase_only:
                            flux_term = mdot_final_iter * A_g_faces[0] if k == fuel_idx else 0.0
                            d_gy_k_rhs[0] += flux_term
                        
                        a_gY_k_bc, b_gY_k_bc, c_gY_k_bc, d_gy_k_rhs_bc = gas_phase.apply_gas_boundary_conditions(
                            a_gY[k].copy(), b_gY[k].copy(), c_gY[k].copy(), d_gy_k_rhs.copy(), Ng,
                            boundary_type=config.OUTER_BOUNDARY_TYPE,
                            boundary_value=Y_k_air_boundary[k]
                        )
                        Yk_new_iter = numerics.solve_tridiagonal(a_gY_k_bc, b_gY_k_bc, c_gY_k_bc, d_gy_k_rhs_bc)
                        if Yk_new_iter is None: status = f"Matrix Solve Error Gas Y (k={k})"; break
                        Y_g_new_iter[k, :] = Yk_new_iter
                    if status != "Running": break

                 elif config.OUTER_BOUNDARY_TYPE == 'CLOSED':
                    # --- 閉境界モード（従来）のロジック ---
                    # RHS d_gT = diag_gT * T_g_iter + Source_Terms (BC)
                    d_gT_rhs = diag_gT * T_g_iter # Use current iter T_g
                    # BC at face 0: flux INTO first cell (i=0) = q_gas_out * A_face[0], because q>0 when r direction is positive.
                    flux_into_cell0_T = q_gas_out_surf_final_iter * A_g_faces[0] 
                    ###flux_into_cell0_T = -q_gas_out_surf_final_iter * A_g_faces[0] 
                    d_gT_rhs[0] += flux_into_cell0_T # Add flux rate to RHS

                    # デバッグ用: 流束の値を確認
                    if config.LOG_LEVEL >= 2:
                        print(f"      DEBUG Gas BC: q_gas_out={q_gas_out_surf_final_iter:.3e}, flux_into_cell0={flux_into_cell0_T:.3e}")
                        print(f"      DEBUG RHS[0] (Before Solve): d_gT_rhs[0]={d_gT_rhs[0]:.6e}")

                    T_g_new_iter = numerics.solve_tridiagonal(a_gT, b_gT, c_gT, d_gT_rhs)
                    if T_g_new_iter is None: status = "Matrix Solve Error Gas T"; break
                    T_g_iter = T_g_new_iter # Update for next iteration

                    # Solve A * Y_k_new_iter = d_gy_k for each species k
                    Y_g_new_iter = np.zeros_like(Y_g_iter)
                    ###fuel_idx = gas_props.fuel_idx
                    if config.LOG_LEVEL >= 1: # ログレベルに応じて表示
                        print(f"    DEBUG Pre-Diff BC: Yf_eq={Y_eq_final_iter[fuel_idx]:.6e}, YO2_eq={Y_eq_final_iter[gas_props.o2_idx]:.6e}, YN2_eq={Y_eq_final_iter[gas_props.n2_idx]:.6e}")
                        print(f"                       mdot={mdot_final_iter:.3e}, Ts={T_s_final_iter:.2f}")
                    for k in range(nsp):
                        # RHSの基本部分 = 対角係数 * 前の反復（または移流ステップ後）の値
                        d_gy_k_rhs = diag_gY[k] * Y_g_iter[k,:]

                        # --- 界面(i=0)での拡散流束による境界条件項を追加 ---
                        # FVMでは、セルに入る流束を正としてRHSに加える
                        # --- 界面(i=0)での【全流束】による境界条件項を追加 ---
                        # <<<<<<<<<<<<<<<<<<<< 修正箇所 START >>>>>>>>>>>>>>>>>>>>
                        total_flux_rate_into_cell0 = 0.0 # [kg/s]
                        if not is_gas_phase_only:
                            if k == fuel_idx:
                                # 燃料の場合: 全流束 Nk_s * A0 = mdot * A0 (セルへの流入)
                                total_flux_rate_into_cell0 = mdot_final_iter * A_g_faces[0]
                            else:
                                # 非燃料の場合: 全流束 Nk_s * A0 = 0
                                total_flux_rate_into_cell0 = 0.0

                            # RHSベクトルに加算
                            d_gy_k_rhs[0] += total_flux_rate_into_cell0
                            # +++ デバッグ出力 (加算後に表示) +++
                            if config.LOG_LEVEL >= 2 and k == fuel_idx:
                                print(f"        DEBUG RHS (Gas Y, k={k}, fuel, iter={iter_count+1}):")
                                rhs_diag_term_0 = (diag_gY[k] * Y_g_iter[k,:])[0] # 加算前のベース項
                                #print(f"          Iter {iter_count+1}: Jk_s={Jk_s:.3e}, BoundaryFluxTerm={diffusion_flux_rate_into_cell0:.3e}")
                                print(f"          Iter {iter_count+1}: RHS[0] before add = {rhs_diag_term_0:.6e}")
                                print(f"          Iter {iter_count+1}: RHS[0] AFTER add = {d_gy_k_rhs[0]:.6e}") # <<<--- 加算後の値を表示
                            # +++ デバッグ出力ここまで +++
                        # ---------------------------------------------------------

                        # --- 壁面(i=Ng-1)の境界条件 ---
                        # 壁面では拡散流束ゼロ (Neumann BC)
                        # これは行列係数(a, b, c)の作り方で暗黙的に処理されているはずなので、
                        # RHSへの追加項は不要。

                        # --- TDMAソルバーで新しいYkを計算 ---
                        Yk_new_iter = numerics.solve_tridiagonal(a_gY[k], b_gY[k], c_gY[k], d_gy_k_rhs)
                        if Yk_new_iter is None:
                            print(f"    ERROR: Matrix Solve Error Gas Y for k={k} (Species: {gas_props.species_names[k]})")
                            status = "Matrix Solve Error Gas Y"; break # kのループを抜ける
                        Y_g_new_iter[k, :] = Yk_new_iter
                    # --- End of species loop ---

                    if status != "Running": break # Iteration loop を抜ける

                 else:
                    raise ValueError(f"Unknown OUTER_BOUNDARY_TYPE: {config.OUTER_BOUNDARY_TYPE}")
            
                 # Normalize after solving all species
                 Y_g_new_iter = np.maximum(Y_g_new_iter, 0.0)
                 sum_Yg_new_iter = np.sum(Y_g_new_iter, axis=0)
                 mask_new_iter = sum_Yg_new_iter > 1e-9
                 Y_g_new_iter[:, mask_new_iter] /= sum_Yg_new_iter[mask_new_iter]
                 if np.any(~mask_new_iter): Y_g_new_iter[:, ~mask_new_iter] = 0.0; Y_g_new_iter[gas_props.n2_idx if gas_props.n2_idx>=0 else 0, ~mask_new_iter] = 1.0
                 Y_g_iter = Y_g_new_iter # Update species array for next interface calc

            else: Y_g_iter = np.zeros((nsp, 0))

            if status != "Running": break # Exit loop if error in diffusion solve

            # --- Update previous values for next iteration ---
            T_s_iter_prev = T_s_final_iter
            mdot_iter_prev = mdot_final_iter
            # Store the current relaxed values to compare in the *next* iteration's convergence check
            T_s_final_prev_iter = T_s_final_iter
            mdot_final_prev_iter = mdot_final_iter

            iter_count += 1
            # --- End of Iteration Loop ---

        if status != "Running": break # Exit time loop if error during iteration

        if not converged and config.LOG_LEVEL >= 0:
             T_s_diff_final = abs(T_s_calc - T_s_iter_prev) # Use last calculated T_s
             mdot_rel_diff_final = abs(mdot_calc - mdot_iter_prev) / (abs(mdot_iter_prev) + 1e-9) # Use last calculated mdot
             print(f"Warning: Interface iteration did not converge! t={current_t:.4e}s.")
             print(f"  Last diff: Ts={T_s_diff_final:.2e}K, mdot_rel={mdot_rel_diff_final:.2e}")

        # Store final converged/iterated values from interface/diffusion step
        T_l_dd = T_l_iter.copy() # T double star (after diffusion)
        T_g_dd = T_g_iter.copy()
        Y_g_dd = Y_g_iter.copy()
        # Store final converged interface values for use in the next step
        mdot_prev_step = mdot_final_iter
        q_liq_out_surf_prev_step = q_liq_out_surf_final_iter
        q_gas_out_surf_prev_step = q_gas_out_surf_final_iter
        v_g_surf_prev_step = v_g_surf_final_iter
        T_s_prev_step = T_s_final_iter
        Y_eq_prev_step = Y_eq_final_iter.copy()
        delta_nu_prev_step = delta_nu_final_iter # <<< 収束したdelta_nuを保存

        # ==============================================================
        # <<< ログ追加箇所 3: 拡散による温度変化ログ >>>
        if config.LOG_LEVEL >= 2 and Ng > 0:
            delta_T_diff = T_g_dd[0] - T_g_star[0] # 拡散ステップによるセル0の温度変化量
            total_delta_T = T_g_dd[0] - T_g_n[0]   # 移流+拡散による合計の温度変化量 (反応前)
            print(f"      DEBUG Diffusion dT (i=0): {delta_T_diff:+.3e} K (T*={T_g_star[0]:.2f} -> T**={T_g_dd[0]:.2f})")
            print(f"      DEBUG Total dT (Adv+Diff)(i=0): {total_delta_T:+.3e} K")
        # ==============================================================

        if config.LOG_LEVEL >= 1:
            print(f"  [Step 3 Done] Interface/Diffusion finished in {iter_count} iterations.")
            if np.any(np.isnan(T_g_dd)) or np.any(np.isnan(Y_g_dd)) or (Nl > 0 and np.any(np.isnan(T_l_dd))):
                 print("    ERROR: NaN detected after Diffusion step!")
                 status = "NaN Error Diffusion"; break
            # print(f"    T_g_dd range: [{np.min(T_g_dd):.1f}, {np.max(T_g_dd):.1f}] K")

        # --- 4. Reaction Step using Cantera ReactorNet --- ### <<< 変更 >>>
        T_l_new = T_l_dd.copy() # No reaction in liquid
        T_g_new = T_g_dd.copy() # Initialize with post-diffusion state
        Y_g_new = Y_g_dd.copy() # Initialize with post-diffusion state

        max_dTdt_current = 0.0 # <<< 温度上昇率を初期化

        if Ng > 0 and reactor is not None and reactor_net is not None: # Check if reactor was initialized
            if config.REACTION_TYPE != 'none': # <<< 追加: configに基づいて反応計算をスキップできるようにする >>>
                if config.LOG_LEVEL >= 2: print("      DEBUG Starting Cantera Reaction Step...")
                react_points = 0

                # --- Calculate density for each cell BEFORE reaction ---
                # Use P_n (pressure at start of step) and T_dd, Y_dd
                rho_g_dd = np.zeros(Ng)
                valid_rho = True
                for i in range(Ng):
                    # ### 注意: gas_props.set_state を使う場合、エラーハンドリングが必要 ###
                    # try:
                    #     gas_props.gas.TDY = T_g_dd[i], P_n, Y_g_dd[:, i] # Set state based on T, P, Y
                    #     rho_g_dd[i] = gas_props.gas.density
                    # except Exception as e_rho:
                    #     print(f"Error setting state for density calc cell {i}: {e_rho}")
                    #     rho_g_dd[i] = np.nan # Mark as invalid
                    # --- より安全な方法: properties.pyのget_densityを使う ---
                    rho_g_dd[i] = gas_props.get_density(T_g_dd[i], P_n, Y_g_dd[:, i])

                    if np.isnan(rho_g_dd[i]) or rho_g_dd[i] <= 0:
                        print(f"Error: Invalid density calc before reaction cell {i}, T={T_g_dd[i]:.1f}")
                        valid_rho = False; break
                if not valid_rho:
                    status = "Prop Error Reaction"; break

                #print(f"  REACTION CALC 1")
                # --- Loop through gas cells ---
                for i in range(Ng):
                    calculate_reactions = True
                    fuel_mole_frac = -1.0
                    hc_concentration = 0.0

                    # --- Reaction Cutoff Check (using T_dd, Y_dd) ---
                    if config.ENABLE_REACTION_CUTOFF:
                        if T_g_dd[i] < config.REACTION_CALC_MIN_TEMP or np.isnan(T_g_dd[i]) or T_g_dd[i] > 6000.0: # 温度上限も追加
                            calculate_reactions = False
                            #print(f"  REACTION CALC 2")
                        else:
                            #print(f"  REACTION CALC 3")
                            # 密度計算で state 設定済みのはずだが、念のため再度設定を試みる
                            # (gas_props.set_state は状態設定に失敗したら False を返す)
                            if gas_props.set_state(T_g_dd[i], P_n, Y_g_dd[:, i]):
                                for species in gas_props.gas.species():
                                    if species.composition.get('C', 0) > 0 or species.composition.get('H', 0) > 0:
                                        if species.name not in ['co2', 'h2o']:
                                            spi = gas_props.gas.species_index(species.name)
                                            hc_concentration += gas_props.gas.X[spi]
                                            #print(f"  REACTION CALC 4")
                                fuel_mole_frac = hc_concentration if fuel_idx >= 0 else -1.0
                            else:
                                #print(f"  REACTION CALC 5")
                                fuel_mole_frac = -1.0 # Failed state set -> no reaction
                                if config.LOG_LEVEL >= 1: print(f"Warning: State set fail in cutoff check cell {i}")

                            if fuel_mole_frac < config.REACTION_CALC_MIN_FUEL_MOL_FRAC:
                                 calculate_reactions = False
                                 #print(f"  REACTION CALC 6")
                    # --- End Cutoff Check ---

                    if calculate_reactions:
                        react_points += 1
                        try:
                            # Set the state of the single reactor's thermo object
                            # Use the constant density calculated before the loop for this cell
                            ###reactor.thermo.TDY = T_g_dd[i], rho_g_dd[i], Y_g_dd[:, i]
                            #print(f"  REACTION CALC")
                            reactor.thermo.TPY = T_g_dd[i], P_n, Y_g_dd[:, i]

                            # Reinitialize the reactor network's state and time
                            # (内部時間をリセットし、現在の状態から積分を開始する)
                            reactor.syncState()
                            reactor_net.reinitialize()

                            # Advance the network by dt
                            abs_target_time = reactor_net.time + dt
                            reactor_net.advance(abs_target_time) # <<< 注意: ターゲット時刻を指定

                            # Get the updated state FROM THE REACTOR
                            T_g_new[i] = reactor.T
                            Y_g_new[:, i] = reactor.thermo.Y

                        except Exception as e_react:
                            print(f"\nERROR during ReactorNet.advance() for cell {i} at t={current_t:.3e}: {e_react}")
                            # Keep the pre-reaction state for this cell if integration fails
                            T_g_new[i] = T_g_dd[i]
                            Y_g_new[:, i] = Y_g_dd[:, i]
                            # Optionally set status to error and break
                            # status = "ReactorNet Error"; break # 必要に応じてコメント解除
                    # else: T_g_new and Y_g_new retain the T_g_dd, Y_g_dd values

                # Ensure mass fractions are valid after reaction step for all cells
                Y_g_new = np.maximum(Y_g_new, 0.0)
                sum_Yg_new = np.sum(Y_g_new, axis=0)
                mask_new = sum_Yg_new > 1e-9
                Y_g_new[:, mask_new] /= sum_Yg_new[mask_new]
                if np.any(~mask_new): Y_g_new[:, ~mask_new] = 0.0; Y_g_new[gas_props.n2_idx if gas_props.n2_idx>=0 else 0, ~mask_new] = 1.0
                #print(f"  REACTION CALC 7")

                # <<< ここからが追加箇所 (ステップ4の最後) >>>
                # 反応による温度上昇率 (dT/dt) を計算
                # T_g_dd: 反応前の温度, T_g_new: 反応後の温度
                if dt > 1e-15:
                    dT_dt_react = (T_g_new - T_g_dd) / dt
                    max_dTdt_current = np.max(dT_dt_react) # 全ガスセル中の最大値を求める

                if config.LOG_LEVEL >= 1:
                    print(f"  [Step 4 Done] Reaction step (ReactorNet) completed. Reacted points: {react_points}/{Ng}")
                    if np.any(np.isnan(T_g_new)) or np.any(np.isnan(Y_g_new)):
                        print("    ERROR: NaN detected after ReactorNet step!")
                        status = "NaN Error Reaction"; # break # If status check is outside step 4

            else: # Reactions OFF in config
                 if config.LOG_LEVEL >= 1: print(f"  [Step 4 Done] Reaction step skipped (REACTION_TYPE='none').")
                 # State remains T_g_dd, Y_g_dd
        else: # Ng <= 0 or reactor objects not initialized
            if config.LOG_LEVEL >= 1: print(f"  [Step 4 Done] Reaction step skipped (No gas cells or reactor).")
        # --- End of Reaction Step Modification ---

        # --- Assign final state at t^{n+1} ---
        T_l = T_l_new; T_g = T_g_new; Y_g = Y_g_new

        # --- 5. Explicit Radius Update (using final converged mdot from step 3) ---
        # mdot_prev_step は、ステップ3の最後に mdot_final_iter で更新されているはず

                # --- 5. Radius Update (including Thermal Expansion) ---
        if not is_gas_phase_only:
            # 5a. 現在のステップ(n)の液滴平均密度を計算
            #     液滴内温度分布 T_l_dd と、ステップ開始時の格子体積 volumes_l を使用
            rho_l_avg_n = liquid_phase.calculate_average_liquid_density(
                T_l_dd, volumes_l, liquid_props
            )
            
            # 5b. 平均密度の時間変化率を後退差分で近似
            if not np.isnan(rho_l_avg_n) and dt > 1e-15:
                d_rho_l_avg_dt = (rho_l_avg_n - rho_l_avg_prev_step) / dt
            else:
                d_rho_l_avg_dt = 0.0

            # 5c. 新しいdR/dt計算関数を呼び出す
            dR_dt = liquid_phase.calculate_dR_dt(
                R_n, mdot_prev_step, rho_l_avg_n, d_rho_l_avg_dt
            )
            
            # 5d. 半径を更新
            R_new_step = R_n + dt * dR_dt
            R = max(R_new_step, 1e-12)

            # 5e. 次のステップのために現在の平均密度を保存
            rho_l_avg_prev_step = rho_l_avg_n

            if config.LOG_LEVEL >= 1:
                dRdt_evap = -mdot_prev_step / rho_l_avg_n if rho_l_avg_n > 1e-6 else 0.0
                dRdt_exp = dR_dt - dRdt_evap
                print(f"  [Step 5 Done] Radius updated: dR/dt={dR_dt:.3e} (Evap:{dRdt_evap:.3e}, Exp:{dRdt_exp:.3e})")

        else: # Gas phase only
            dR_dt = 0.0
            R = R_n

        '''
        if not is_gas_phase_only:
            # rho_l_s_final は、ステップ3で計算された T_s_prev_step (T_s_final_iter) における液滴表面密度
            rho_l_s_for_R_update = liquid_props.get_prop('density', T_s_prev_step) if Nl > 0 else 1000.0
            dR_dt = liquid_phase.calculate_dR_dt(mdot_prev_step, rho_l_s_for_R_update)
            R_new_step = R_n + dt * dR_dt # R_n はステップ開始時の半径
            R = max(R_new_step, 1e-12) # Prevent negative radius
        else:
            dR_dt = 0.0; R = R_n # Keep radius constant (at threshold)
        '''

        # --- Phase Transition Check ---
        if R < config.R_TRANSITION_THRESHOLD and not is_gas_phase_only:
            print(f"\n--- Transitioning to Gas Phase Only at t={current_t:.4e} s (R={R:.2e} m) ---")
            is_gas_phase_only = True
            is_just_evaporated = True
            Nl = 0 # Set liquid grid points to zero
            T_l = np.array([]) # Empty liquid arrays
            r_l_centers, r_l_nodes, volumes_l = np.array([]), np.array([]), np.array([])
            A_l_faces = np.array([])
            R = config.R_TRANSITION_THRESHOLD # Keep R fixed at threshold
            # Re-generate gas grid only based on new R? No, keep RMAX fixed.
            r_g_centers, r_g_nodes, volumes_g = grid.gas_grid_fvm(R, config.RMAX, Ng)
            A_g_faces = grid.face_areas(r_g_nodes)

        # ==============================================================
        # <<< グローバル質量保存チェック: 圧力更新直前の気相総質量計算 >>>
        # ==============================================================
        # このステップでの総蒸発量を計算
        M_evap_this_step = 0.0
        if not is_gas_phase_only and Nl > 0 :
             # mdot_prev_step はステップ3の反復計算で得られた今回の蒸発流束密度
             # R_n はステップ開始時の半径 (表面積計算に使う)
             M_evap_this_step = mdot_prev_step * (4.0 * np.pi * R_n**2) * dt

        # 移流・拡散・反応・半径更新後の気相の総質量を計算
        # 注意: この時点での volumes_g は、ステップ5で更新された最新の R に基づいて再計算するのが理想
        # ここでは簡単のため、ステップ1で R_n に基づいて計算された volumes_g を使う。
        # より正確な比較のためには、ここで再度 grid.gas_grid_fvm(R, config.RMAX, Ng) を呼ぶ必要がある。
        # ただし、Rの変化が小さければ誤差も小さい。
        #
        # 今回は、まず既存の volumes_g (R_n ベース) を使って傾向を掴む。
        # 次に、必要なら R ベースの volumes_g で再計算する。
        #
        # 密度計算に使う圧力は P_n (ステップ開始時の圧力) を使う。
        # (新しい圧力を求める前なので、ここでの密度評価は P_n ベースで行うのが一貫性がある)
        M_gas_end_of_step_calc = 0.0
        if Ng > 0:
            # grid を最新のRで更新 (より正確な体積のため)
            _, r_g_nodes_updated, volumes_g_updated = grid.gas_grid_fvm(R, config.RMAX, Ng) # Rは最新の半径
            for i_mgas in range(Ng):
                # T_g, Y_g は反応ステップ完了後の最新の値
                density_at_cell_end = gas_props.get_density(T_g[i_mgas], P_n, Y_g[:, i_mgas])
                if not (np.isnan(density_at_cell_end) or density_at_cell_end <= 0):
                    M_gas_end_of_step_calc += density_at_cell_end * volumes_g_updated[i_mgas] # 最新の体積を使用
                else:
                    print(f"    WARNING: Invalid density for M_gas_end_of_step_calc at cell {i_mgas}, T={T_g[i_mgas]:.1f}, P={P_n:.2e}")
        
        mass_discrepancy_in_step = (M_gas_end_of_step_calc - M_gas_start_of_step) - M_evap_this_step
        
        if config.LOG_LEVEL >= 1: # ログレベル1以上で表示
            print(f"    DEBUG Mass: M_gas_end_of_step_calc = {M_gas_end_of_step_calc:.6e} kg (P_n={P_n:.3e} for density eval)")
            print(f"    DEBUG Mass: M_evap_this_step = {M_evap_this_step:.6e} kg (mdot={mdot_prev_step:.3e}, R_n={R_n*1e6:.2f}um, dt={dt:.3e})")
            # 許容誤差は蒸発量の数%など、相対的な値と絶対的な値の組み合わせで設定すると良い
            relative_tolerance = 0.05 # 5%
            absolute_tolerance = 1e-15 # 小さな絶対許容値
            is_discrepancy_significant = abs(mass_discrepancy_in_step) > max(absolute_tolerance, abs(M_evap_this_step) * relative_tolerance)
            
            if is_discrepancy_significant or M_evap_this_step == 0: # 蒸発量がゼロの場合も不一致を表示 (M_gas_end が M_gas_start と異なる場合)
                print(f"    DEBUG Mass: Discrepancy = {mass_discrepancy_in_step:.3e} kg "
                      f"[ (M_end_calc - M_start) - M_evap ]")
                print(f"                 M_start={M_gas_start_of_step:.6e}, M_end_calc={M_gas_end_of_step_calc:.6e}, M_evap={M_evap_this_step:.6e}")
        # ==============================================================            

        # --- ステップ6の圧力更新ループの直前で、最新のRに基づく volumes_g を準備 ---
        if Ng > 0:
            try:
                # この volumes_g_for_P_update を圧力更新ループとループ最終ログの両方で使う
                _, _, volumes_g_for_P_update = grid.gas_grid_fvm(R, config.RMAX, Ng) # Rは最新の半径
            except Exception as e_grid_step6:
                print(f"    ERROR grid gen before Pressure Update (R={R:.3e}): {e_grid_step6}")
                status = "Grid Error for P Update"; break # ループを抜ける
        else:
            volumes_g_for_P_update = np.array([])

        # --- 6. Update Pressure (Uniform pressure assumption) ---
   
        # 気相体積の計算
        V_liquid = (4.0/3.0) * np.pi * R**3 if Nl > 0 and not is_gas_phase_only else 0.0
        V_gas_total = V_vessel - V_liquid
        if V_gas_total <= 1e-15: # 体積がゼロまたは負にならないように
            print(f"    ERROR: Calculated gas volume is non-positive ({V_gas_total:.2e}). V_vessel={V_vessel:.2e}, V_liq={V_liquid:.2e}")
            status = "Error Gas Volume P-Update"; break

        M_gas_target_for_P_update = M_gas_start_of_step + M_evap_this_step
        if config.LOG_LEVEL >= 1:
            print(f"    DEBUG Pressure Update: Target Gas Mass = {M_gas_target_for_P_update:.7e} kg (Start: {M_gas_start_of_step:.7e}, Evap: {M_evap_this_step:.7e})")

        if config.COMPRESSION_MODEL == 'NONE':
            # --- 従来モード: 質量保存則から圧力を計算 ---
            if config.LOG_LEVEL >= 1:
                print("  [Step 6] Updating pressure based on mass conservation (CLOSED system)...")

            # --- 圧力Pを求めるための関数を定義 (これは前回提案と同様) ---
            def get_gas_mass_at_pressure(P_eval, T_g_dist, Y_g_dist, volumes_g_dist, gas_props_obj, Ng_val, nsp_val):
                current_total_mass = 0.0
                if Ng_val == 0: return 0.0
                for i_cell in range(Ng_val):
                    density = gas_props_obj.get_density(T_g_dist[i_cell], P_eval, Y_g_dist[:, i_cell])
                    if np.isnan(density) or density <= 0:
                        if config.LOG_LEVEL >=1: # エラーを少し詳細に
                            print(f"        WARNING (get_gas_mass): Invalid density ({density:.2e}) at P_eval={P_eval:.2e} for cell {i_cell}, T={T_g_dist[i_cell]:.1f}")
                        return np.inf # または非常に大きな正の数
                    current_total_mass += density * volumes_g_dist[i_cell]
                return current_total_mass

            def pressure_residual_for_mass_target(P_guess, target_mass, # 追加引数
                                                current_T_g, current_Y_g, current_volumes_g,
                                                current_gas_props, current_Ng, current_nsp):
                # T_g, Y_g, volumes_g_for_P_update, gas_props, Ng, nsp は引数で受け取る
                calculated_mass = get_gas_mass_at_pressure(P_guess, current_T_g, current_Y_g, current_volumes_g,
                                                        current_gas_props, current_Ng, current_nsp)
                if np.isinf(calculated_mass):
                    return np.inf # ソルバーにエラーを伝える
                residual = calculated_mass - target_mass
                if config.LOG_LEVEL >= 3:
                    print(f"        P_Resid: P_guess={P_guess/1e6:.6f} MPa, Calc_M={calculated_mass:.8e}, Target_M={target_mass:.8e}, Resid={residual:.4e}")
                return residual

            # --- 非線形ソルバーを使用して P を見つける ---
            P_final_this_step = P_n # 初期化
            
            # T_g, Y_g は反応・移流拡散後の最新の値を使用
            # volumes_g_for_P_update は最新のRに基づくセル体積

            if Ng > 0 and M_gas_target_for_P_update > 1e-18: # 目標質量が極小でない場合のみ実行
                try:
                    # scipy.optimize.newton を使用
                    # 初期推定値として P_n を使用
                    # args に pressure_residual_for_mass_target が必要とする追加引数を渡す
                    # T_g, Y_g は現在のステップの最新の値 (反応ステップ後、半径更新後)
                    # volumes_g_for_P_update は最新のRに基づいた体積
                    P_solution = op.newton(pressure_residual_for_mass_target, P_n,
                                            args=(M_gas_target_for_P_update,
                                                T_g, Y_g, volumes_g_for_P_update,
                                                gas_props, Ng, nsp),
                                            tol=1e-6,      # Paオーダーの目標質量に対する圧力の許容誤差。より厳しくしても良い。
                                                        # 例えば、(目標質量 * 1e-9) 程度の質量誤差を生む圧力変化を tol にするなど。
                                                        # dM/dP ~ M/P から dP ~ P * (dM/M) なので、相対質量誤差1e-9なら、P*1e-9 程度。
                                                        # M_gas_target_for_P_update * 1e-9 (kg) の変化を引き起こす dP を tol とする。
                                                        # dP = (P/M)dM。tol_P = (P_n / M_gas_target_for_P_update) * (M_gas_target_for_P_update*1e-9) = P_n * 1e-9
                                            #tol = P_n * 1e-8, # 圧力の相対許容誤差に変更するのも手
                                            maxiter=25)     # 最大反復回数

                    if np.isnan(P_solution) or P_solution < 1e3:
                        if config.LOG_LEVEL >= 0: print(f"    Warning: Newton solver for P returned invalid value ({P_solution:.2e}). Using P_n.")
                        P_final_this_step = P_n
                    else:
                        P_final_this_step = max(P_solution, 1e3) # 物理的な下限
                        # ソルバー収束後の質量を再確認
                        final_mass_check = get_gas_mass_at_pressure(P_final_this_step, T_g, Y_g, volumes_g_for_P_update, gas_props, Ng, nsp)
                        if config.LOG_LEVEL >= 1:
                            print(f"    P Solver: Converged. P_new={P_final_this_step/1e6:.6f} MPa. "
                                f"Resulting Mgas={final_mass_check:.8e} (Target={M_gas_target_for_P_update:.8e})")
                            if abs(final_mass_check - M_gas_target_for_P_update) > M_gas_target_for_P_update * 1e-7 : # 許容誤差 0.00001%
                                print(f"    ---> SIGNIFICANT Mass Discrepancy from Target after P solve: {(final_mass_check - M_gas_target_for_P_update):.3e} kg")


                except RuntimeError as e_newton:
                    if config.LOG_LEVEL >= 0: print(f"    Warning: Newton solver for P failed: {e_newton}. Using P_n.")
                    P_final_this_step = P_n
                except Exception as e_psolve:
                    if config.LOG_LEVEL >= 0: print(f"    ERROR during pressure solve for mass target: {e_psolve}")
                    P_final_this_step = P_n
            else: # Ng=0 or M_gas_target_for_P_update is too small
                P_final_this_step = P_n # Keep previous pressure
                if Ng > 0 and M_gas_target_for_P_update <= 1e-18 and config.LOG_LEVEL >= 1:
                    print(f"    DEBUG Pressure Update: Target gas mass {M_gas_target_for_P_update:.2e} is negligible. Keeping P=P_n.")


            P = P_final_this_step # 更新された圧力を格納

        else:
            # --- 新規モード: エンジンモデルから圧力・温度を取得 ---
            if config.LOG_LEVEL >= 1:
                print(f"  [Step 6] Updating ambient conditions via '{config.COMPRESSION_MODEL}' model...")

            P_new, T_amb_new = engine_model.get_ambient_conditions(current_t + dt)
            P = P_new # このステップの最終的な圧力として更新

            if config.LOG_LEVEL >= 1:
                print(f"    P={P/1e6:.4f} MPa, T_amb={T_amb_new:.2f} K")

        # --- 圧力更新後の最終的な気相総質量と、それに基づく平均物性値の再計算 ---
        # (ログ出力や次のステップのために重要)
        total_mass_gas = 0.0
        mass_weighted_Yk_sum_for_avg = np.zeros(nsp) # ループ内で使う変数名と区別
        mass_weighted_T_sum_for_avg = 0.0
        # mean_mw_avg_iter と T_avg_iter をこのスコープで定義・初期化
        mean_mw_avg_final = gas_props.gas.mean_molecular_weight # フォールバック値
        T_avg_final = np.mean(T_g) if Ng > 0 else config.T_INF_INIT # フォールバック値

        if Ng > 0:
            for i in range(Ng):
                density_final = gas_props.get_density(T_g[i], P, Y_g[:, i]) # 更新された P を使用
                if not (np.isnan(density_final) or density_final <= 0):
                    mass_i = density_final * volumes_g_for_P_update[i]
                    total_mass_gas += mass_i
                    mass_weighted_Yk_sum_for_avg += mass_i * Y_g[:, i]
                    mass_weighted_T_sum_for_avg += mass_i * T_g[i]
                else:
                    print(f"    WARNING: Invalid density for final log mass calc at cell {i}. Using target mass for log.")
                    total_mass_gas = M_gas_target_for_P_update # 最善の推測値として目標値を使う
                    break # このループを抜けて、total_mass_gas に基づく平均物性計算へ

        if total_mass_gas > 1e-15: # total_mass_gas が有効な場合のみ平均値を計算
            Y_g_avg_final = mass_weighted_Yk_sum_for_avg / total_mass_gas
            Y_g_avg_final = np.maximum(Y_g_avg_final, 0.0)
            sum_Y_avg_final = np.sum(Y_g_avg_final)
            if sum_Y_avg_final > 1e-9: Y_g_avg_final /= sum_Y_avg_final
            else: Y_g_avg_final = gas_props.X_amb_init # Fallback

            T_avg_final = mass_weighted_T_sum_for_avg / total_mass_gas
            T_avg_final = np.clip(T_avg_final, 200.0, 6000.0)

            if gas_props.set_state(T_avg_final, P, Y_g_avg_final): # 更新されたPで状態設定
                mean_mw_avg_final = gas_props.gas.mean_molecular_weight
            else: # 状態設定失敗時はフォールバック
                print(f"    Warning: Failed to set state for final avg MW. Using previous P-iter or ambient.")
                X_amb_local = gas_props.X_amb_init # properties.py で初期化時に計算・保持
                mean_mw_avg_final = np.sum(X_amb_local * gas_props.molecular_weights)
        elif Ng > 0 : # total_mass_gas がほぼゼロだが、気相セルは存在する場合
            print(f"    Warning: Total gas mass for averaging is {total_mass_gas:.2e}. Using ambient/initial for T_avg, MW_avg.")
            T_avg_final = config.T_INF_INIT
            gas_props.set_state(T_avg_final, P, gas_props.X_amb_init)
            mean_mw_avg_final = gas_props.gas.mean_molecular_weight
        # Ng=0 の場合は、total_mass_gas=0, T_avg_final, mean_mw_avg_final は初期化されたフォールバック値のまま。

        total_moles_gas = total_mass_gas / mean_mw_avg_final if mean_mw_avg_final > 1e-3 else 0.0

        if config.LOG_LEVEL >= 1:
             print(f"    DEBUG State After P Update (Step 6): P={P:.6e}, R={R*1e6:.3f}um, T_g[0]={T_g[0] if Ng>0 else np.nan:.3f}, Y_g[fuel_idx,0]={Y_g[fuel_idx,0] if Ng>0 else np.nan:.6e}, FinalActualMgas={total_mass_gas:.7e} (TargetMgas={M_gas_target_for_P_update:.7e})")
             mass_diff_from_target = total_mass_gas - M_gas_target_for_P_update
             if abs(mass_diff_from_target) > max(1e-15, M_gas_target_for_P_update * 1e-8) : # 許容誤差を相対値も考慮
                 print(f"    ---> Mass Discrepancy from Target after P solve: {mass_diff_from_target:.3e} kg")


        if np.isnan(P):
            print("    ERROR: NaN detected during Pressure update!")
            status = "NaN Error Pressure"; break

        if config.LOG_LEVEL >= 1:
            # ログには最終的に計算された total_mass_gas とそれに基づく値を使用
            print(f"  [Step 6 Done] Pressure updated: {P/1e6:.4f} MPa (ActualMgas={total_mass_gas:.3e} kg, Ngas={total_moles_gas:.3e} kmol, Tavg={T_avg_final:.1f} K, MWavg={mean_mw_avg_final:.2f} kg/kmol, Vgas={V_gas_total:.3e} m^3)")

        # --- 7. Update Time and Step Count ---
        #current_t += dt
        #step_count_total += 1
        final_iter_count = iter_count # Store the final iteration count for this step

        # --- 8. Data Saving & Logging ---
        # ----------------------------------------------------------------------
        # 半径方向プロファイルのCSV出力（時間ベース）
        # ----------------------------------------------------------------------
        if config.SAVE_RADIAL_PROFILES and current_t >= next_profile_output_time:
            # 現在のステップ完了後の状態でCSVファイルに保存
            #output.save_radial_profile_to_csv(current_t, step_count_total, R, T_l, T_g, Y_g, gas_props)
            output.save_radial_profile_to_csv(current_t, output_count_total, R, T_l, T_g, Y_g, gas_props)
            
            # 次の出力時刻を更新
            next_profile_output_time += config.OUTPUT_TIME_INTERVAL
            
            # プロット用にメモリに状態を保存（任意）
            # 注意：時間ステップが細かい場合、メモリ使用量が増大する可能性があります。
            # 大規模計算では、このメモリ保存は無効化し、プロットはCSVから直接行うのが望ましいです。
            saved_times.append(current_t)
            saved_results_list.append({'T_l': T_l.copy(), 'T_g': T_g.copy(), 'Y_g': Y_g.copy(), 'R': R, 'P': P})
            output_count_total += 1
        # ----------------------------------------------------------------------
        
        save_this_step = False
        time_after_step = current_t + dt # このステップが完了した後の時刻

        if time_after_step >= next_save_time - 1e-12: # 浮動小数点誤差を考慮
            save_this_step = True
            next_save_time += config.SAVE_INTERVAL_DT


        # --- save_this_step が True ならデータを保存 --- (または一定ステップごと)
        if save_this_step or (step_count_total % 100 == 0):
            # a. メモリ上のリストに詳細結果を保存（プロット用）
            saved_times.append(time_after_step)
            saved_results_list.append({'T_l': T_l.copy(), 'T_g': T_g.copy(), 'Y_g': Y_g.copy(), 'R': R, 'P': P})
            # b. CSVファイルにライブロギング（追記）
            try:
                live_data = pd.DataFrame({
                    'Time (s)': [time_after_step], 'Radius (m)': [R], 'Pressure (Pa)': [P],
                    'T_liquid_surf_cell (K)': [T_l[-1] if not is_gas_phase_only and Nl > 0 else np.nan],
                    'T_gas_surf_cell (K)': [T_g[0] if Ng > 0 else np.nan],
                    'T_solved_interface (K)': [T_s_prev_step],
                    'Mdot (kg/m2/s)': [mdot_prev_step],
                    'MaxGasTemp (K)': [np.max(T_g) if Ng > 0 else np.nan]
                })
                live_data.to_csv(csv_filename, mode='a', header=False, index=False, float_format='%.6e')
            except Exception as e_csv:
                print(f"Warning: Error writing to live CSV: {e_csv}")

            # c. リスタートファイルを上書き保存
            try:
                np.savez(restart_filepath,
                         current_t=current_t, dt=dt, step_count_total=step_count_total,
                         R=R, P=P, T_l=T_l, T_g=T_g, Y_g=Y_g,
                         is_gas_phase_only=is_gas_phase_only,
                         mdot_prev_step=mdot_prev_step, T_s_prev_step=T_s_prev_step,
                         Y_eq_prev_step=Y_eq_prev_step,
                         rho_l_avg_prev_step=rho_l_avg_prev_step)
            except Exception as e_npz:
                print(f"Warning: Error saving restart file: {e_npz}")

        # ==============================================================

        # --- Terminal Logging ---
        log_now = step_count_total % max(1, config.TERMINAL_OUTPUT_INTERVAL_STEP) == 0
        if log_now or save_this_step or status != "Running": # Log if saved or interval or finished
             # Calculate max dT/dt from reaction step for ignition check (optional)
             #max_dTdt_current = np.max(dT_dt_react) if 'dT_dt_react' in locals() and len(dT_dt_react)>0 else 0.0
             maxTgas_log = np.max(T_g) if Ng > 0 else np.nan
             Tl_s_cell = T_l[Nl-1] if Nl > 0 else np.nan
             Tg_s_cell = T_g[0] if Ng > 0 else np.nan
             step_stop_time = time.time(); step_duration = step_stop_time - step_start_time
             mdot_log = mdot_prev_step if not is_gas_phase_only else 0.0
             Ts_log = T_s_prev_step if not is_gas_phase_only else np.nan_to_num(Tg_s_cell)
             # <<<<<<< 修正箇所 (2): iter_info を最終回数表示に変更 >>>>>>>
             iter_info = f"Iter:{final_iter_count+1}" # Add 1 because iter_count is 0-based index
             # <<<<<< 修正箇所 (2): conv_warn の定義 >>>>>>>
             conv_warn = "[WARN:IterFail]" if not converged else "" # Define conv_warn based on the flag

             # <<<<<<< 修正箇所 (3): print文に iter_info を含める >>>>>>>
             # より多くの情報を表示、行が長くなる場合は調整
             print(f"t={current_t:.4e}s dt={dt:.2e} Stp:{step_count_total:<6} "
                   f"R={R*1e6:.2f}um P={P/1e6:.3f}MPa "
                   f"Tls={np.nan_to_num(Tl_s_cell):.1f}K Tgs={np.nan_to_num(Tg_s_cell):.1f}K T_s={Ts_log:.1f}K Tmax={np.nan_to_num(maxTgas_log):.1f}K "
                   f"mdot={mdot_log:.2e} {iter_info:<8} Dur:{step_duration:.3f}s {conv_warn}" + " "*5) # 右詰めでスペース確保
             print(f"total_mass_gas={total_mass_gas:.4e}s total_moles_gas={total_moles_gas:.4e} "
                   f"Mw:{gas_props.gas.mean_molecular_weight:.4f} "
                   f"T_avg_final={T_avg_final:.2f}K" + " "*5) # 右詰めでスペース確保

             if not converged: # 追加：収束失敗時は常に警告表示
                 print(f"      WARNING: Interface iteration failed to converge in step {step_count_total}")
        # --- 7. 時間とステップ数を更新 ---
        # このステップの計算がすべて完了したので、時間を進める
        current_t += dt
        step_count_total += 1
 
        # --- 9. Check Termination Conditions ---
        if R < R0 * 0.001 and not is_gas_phase_only: # Terminate if radius is extremely small
             print(f"\n--- Droplet effectively evaporated (R={R:.2e}m) at t={current_t:.4e}s. Stopping. ---")
             status = "Evaporated"
             break
        if step_count_total >= config.MAX_STEPS: status = "Max Steps"; break
        if maxTgas_log >= config.IGNITION_CRITERION_TMAX: status = "Max Temperature"; break
        # 温度上昇率による新しい着火判定を追加
        if max_dTdt_current >= config.IGNITION_CRITERION_DTDT:
            print(f"\n--- Ignition detected by dT/dt criterion at t={current_t:.4e}s. Stopping. ---")
            print(f"    (Max dT/dt = {max_dTdt_current:.2e} K/s >= Threshold = {config.IGNITION_CRITERION_DTDT:.2e} K/s)")
            status = "Ignition (dT/dt)"
            break

        if step_count_total > 0: # 最初のステップは除く
            # 液滴の質量変化 (減少分を正とする)
            delta_m_liquid = mdot_prev_step * (4.0 * np.pi * R_n**2) * dt

            # 実際の気相の質量増加
            # total_mass_gas_at_n は、このステップ開始時の気相総質量を保持する変数 (別途追加が必要)
            # ここでは、簡潔にデバッグ情報を表示することに焦点を当てる
            if config.LOG_LEVEL >= 0:
                print(f"    Mass Check: R={R*1e6:.3f} um, mdot={mdot_prev_step:.3e} kg/m2/s")
                print(f"                Liquid Mass Loss (from mdot): {delta_m_liquid:.3e} kg")

            # 気相の質量変化 (増加分を正とする)
            # total_mass_gas_prev は前のステップの total_mass_gas (保存しておく必要あり)
            # ここでは total_mass_gas (現在のステップで計算されたもの) と、
            # 前のステップの total_mass_gas (例えば total_mass_gas_at_n) を比較
            # total_mass_gas_at_n = total_mass_gas # ループの最後に次のステップのために保存

            # 簡単のため、現在の蒸発量から期待される気相質量増加を計算
            expected_gas_mass_increase = mdot_prev_step * (4 * np.pi * R_n**2) * dt # mdot_prev_step は Step 3 で計算された界面での値

            # 実際の気相質量 (total_mass_gas はStep6で計算済み)
            # この比較は少し複雑なので、まずは total_mass_gas 自体の挙動を追う

            if config.LOG_LEVEL >= 0: # 常に表示するレベル
                print(f"    Mass Check: R={R*1e6:.3f} um, mdot={mdot_prev_step:.3e} kg/m2/s")
                print(f"                Expected Gas Mass Increase (from mdot): {expected_gas_mass_increase:.3e} kg")
                # total_mass_gas_at_n が必要 (このステップ開始時の気相総質量)
                # if 'total_mass_gas_at_n' in locals():
                #     actual_gas_mass_increase = total_mass_gas - total_mass_gas_at_n
                #     print(f"                Actual Gas Mass Increase: {actual_gas_mass_increase:.3e} kg, Discrepancy: {expected_gas_mass_increase - actual_gas_mass_increase:.3e} kg")

        # total_mass_gas_at_n = total_mass_gas # 次のステップのために保存 (ループの最後に置く)
            if config.LOG_LEVEL >= 1:
                print(f"    DEBUG State Before M_gas_at_loop_end: P={P:.6e}, R={R*1e6:.3f}um, T_g[0]={T_g[0]:.3f}, Y_g[fuel_idx,0]={Y_g[fuel_idx,0]:.6e}")

            # ==============================================================
            # <<< 時間ループ最後の状態確認ログ (最新Rに基づくvolumes_gを使用) >>>
            # ==============================================================
            M_gas_at_loop_end = 0.0
            if Ng > 0:
                try:
                    _, _, volumes_g_at_loop_end_calc = grid.gas_grid_fvm(R, config.RMAX, Ng) # Rは最新の半径
                except Exception as e_grid_final:
                    print(f"    ERROR grid gen at loop end for M_gas_at_loop_end (R={R:.3e}): {e_grid_final}")
                    volumes_g_at_loop_end_calc = volumes_g 
                    print(f"    WARNING: Using R_n-based volumes_g for M_gas_at_loop_end due to grid error.")

                for i_mgas_loop_end in range(Ng):
                    density_val_loop_end = gas_props.get_density(T_g[i_mgas_loop_end], P, Y_g[:, i_mgas_loop_end])
                    if not (np.isnan(density_val_loop_end) or density_val_loop_end <= 0):
                        M_gas_at_loop_end += density_val_loop_end * volumes_g_at_loop_end_calc[i_mgas_loop_end]
                    else:
                        print(f"    WARNING: Invalid density for M_gas_at_loop_end at cell {i_mgas_loop_end}, T={T_g[i_mgas_loop_end]:.1f}, P={P:.2e}")

            if config.LOG_LEVEL >= 1:
                time_ended_this_step = current_t + dt
                print(f"    DEBUG Mass: M_gas_at_VERY_END_of_step {step_count_total} (actual_time_ended={time_ended_this_step:.4e}, P={P:.3e}, R_final={R*1e6:.2f}um) = {M_gas_at_loop_end:.6e} kg")
            # ==============================================================


    # --- Finalization ---
    end_time_loop = time.time()
    last_saved_time = saved_times[-1] if saved_times else -1.0
    # current_t はループを抜けたときの時刻
    # 最後に保存された時刻と現在の時刻が十分に異なっていれば最終状態を保存
    # (ループ終了が T_END ピッタリで、かつそれが丁度保存タイミングだった場合を除くため)
    #needs_final_save = abs(current_t - last_saved_time) > 1e-15
    needs_final_save = True

    if needs_final_save:
        # (最終状態の保存コード - 前回の回答のコードを流用)
        # --- NaN Check ---
        is_nan_present = False
        if not is_gas_phase_only and Nl > 0 and np.any(np.isnan(T_l)): is_nan_present = True
        if Ng > 0 and (np.any(np.isnan(T_g)) or np.any(np.isnan(Y_g))): is_nan_present = True
        if np.isnan(R) or np.isnan(P): is_nan_present = True

        if not is_nan_present:
            # --- Append to Python List ---
            saved_times.append(current_t)
            saved_results_list.append({'T_l': T_l.copy(), 'T_g': T_g.copy(), 'Y_g': Y_g.copy(), 'R': R, 'P': P})
            print(f"Saving final computed state at t={current_t:.4e} (Status: {status})")
            # --- Prepare and Append to CSV ---
            try:
                Tls_final = T_l[Nl-1] if Nl > 0 and not is_gas_phase_only else np.nan
                Tgs_final = T_g[0] if Ng > 0 else np.nan
                R_final_log = R if not is_gas_phase_only else np.nan
                mdot_final_log = mdot_prev_step if not is_gas_phase_only else 0.0
                Ts_final_log = T_s_prev_step if not is_gas_phase_only else np.nan_to_num(Tgs_final)
                maxTgas_final = np.max(T_g) if Ng > 0 else np.nan

                final_data = pd.DataFrame({
                    'Time (s)': [current_t],
                    'Radius (m)': [np.nan_to_num(R_final_log)],
                    'Pressure (Pa)': [P],
                    'T_liquid_surf_cell (K)': [np.nan_to_num(Tls_final)],
                    'T_gas_surf_cell (K)': [np.nan_to_num(Tgs_final)],
                    'T_solved_interface (K)': [np.nan_to_num(Ts_final_log)],
                    'Mdot (kg/m2/s)': [mdot_final_log],
                    'MaxGasTemp (K)': [np.nan_to_num(maxTgas_final)]
                })
                final_data.to_csv(csv_filename, mode='a', header=False, index=False, float_format='%.6e')
            except Exception as e_csv_final:
                print(f"Warning: Error appending final state to CSV: {e_csv_final}")

            # 最終時刻の半径方向プロファイルもCSVとして保存
            if config.SAVE_RADIAL_PROFILES:
                print(f"Saving final radial profile CSV at t={current_t:.4e}...")
                try:
                    output.save_radial_profile_to_csv(
                        current_t, output_count_total, R, T_l, T_g, Y_g, gas_props
                    )
                except Exception as e_final_csv:
                    print(f"Warning: Could not save final radial profile CSV. Error: {e_final_csv}")

        else:
            print(f"Warning: NaN detected in final state at t={current_t:.4e}. Not saving final step.")
    # ==============================================================

    if status == "Running": status = "Ended (Time Limit)"
    print("-" * 60); print(f"Simulation loop finished at t = {current_t:.6e} s.")
    print(f"Final Status: {status}"); print(f"Total simulation time: {end_time_loop - start_time_loop:.2f} seconds.")
    print(f"Total steps: {step_count_total}"); print("-" * 60)

    # --- Post Processing ---
    if len(saved_results_list) > 1:
        plotting.plot_results(saved_times, saved_results_list, config.OUTPUT_DIR, nsp, config.NL, config.NG, gas_props) # Pass original NL, NG
    else:
        print("Not enough data points saved for plotting.")

if __name__ == "__main__":
    # Create output dir if needed
    out_dir = config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    # Remove old log file if exists
    csv_path = os.path.join(out_dir, 'time_history_live.csv')
    if os.path.exists(csv_path):
        try: os.remove(csv_path); print(f"Removed old log file: {csv_path}")
        except OSError as e: print(f"Error managing old log file {csv_path}: {e}")

    # Run simulation with error catching
    try:
         run_simulation_split()
    except Exception as main_e:
         print("\n" + "="*20 + " CRITICAL ERROR IN MAIN SCRIPT " + "="*20)
         import traceback
         traceback.print_exc()
         print("="*60)
    finally:
        print("\n--- Finalizing Run ---")
        try:
            # config.py を結果ディレクトリにコピー
            source_config_path = 'config.py'
            # 分かりやすいように別名を付けてコピーする (例: config_run.py)
            destination_config_path = os.path.join(config.OUTPUT_DIR, 'config_run.py')
            
            if os.path.exists(source_config_path):
                shutil.copy2(source_config_path, destination_config_path)
                print(f"Successfully copied config file to: {destination_config_path}")
            else:
                print(f"Warning: Could not find source config file at '{source_config_path}' to copy.")

        except Exception as e_copy:
            print(f"ERROR: Failed to copy config file. Details: {e_copy}")
        
        print("\nProgram finished.")
