# test_initial_interface.py

import numpy as np
import config
import properties
import grid
import interface

def run_interface_test():
    """
    t=0の初期条件における界面計算を単体でテストする。
    """
    print("--- 界面計算 単体テスト開始 ---")

    # --- 1. 初期化 ---
    # シミュレーションの初期化と同様のオブジェクトを準備
    try:
        gas_props = properties.GasProperties(config.MECH_FILE)
        fuel_molar_mass = gas_props.molecular_weights[gas_props.fuel_idx] / 1000.0
        liquid_props = properties.LiquidProperties(config.LIQUID_PROP_FILE, fuel_molar_mass=fuel_molar_mass)
        fugacity_interpolator = properties.FugacityInterpolator(config.FUGACITY_MAP_FILE)
    except Exception as e:
        print(f"ERROR: 初期化中にエラーが発生しました: {e}")
        return

    # --- 2. t=0 の条件を設定 ---
    P = config.P_INIT
    R = config.R0
    Nl = config.NL
    Ng = config.NG
    
    # 温度と組成
    T_l = np.full(Nl, config.T_L_INIT)
    T_g = np.full(Ng, config.T_INF_INIT)
    gas_props.set_state(config.T_INF_INIT, P, config.X_INF_INIT)
    Y_g = np.zeros((gas_props.nsp, Ng))
    Y_g[:, :] = gas_props.gas.Y[:, np.newaxis]
    
    # グリッド
    r_l_centers, _, _ = grid.liquid_grid_fvm(R, Nl)
    r_g_centers, _, _ = grid.gas_grid_fvm(R, config.RMAX, Ng)

    print(f"\nテスト条件:")
    print(f"  P = {P/1e6:.2f} MPa, R = {R*1e6:.1f} um")
    print(f"  T_liquid (surface cell) = {T_l[-1]:.1f} K")
    print(f"  T_gas (cell 0) = {T_g[0]:.1f} K, T_gas (cell 1) = {T_g[1]:.1f} K")

    # --- 3. 界面ソルバーを呼び出し ---
    print("\ninterface.solve_interface_conditions を呼び出します...")
    try:
        mdot, q_gas_out, q_liq_out, v_g_surf, Y_eq, T_s, delta_nu = interface.solve_interface_conditions(
            T_l_node_last=T_l[-1],
            T_g_node0=T_g[0],
            T_g_node1=T_g[1],
            Y_g_node0=Y_g[:, 0],
            Y_g_node1=Y_g[:, 1],
            P=P,
            R=R,
            r_l_node_last_center=r_l_centers[-1],
            r_g_node0_center=r_g_centers[0],
            r_g_node1_center=r_g_centers[1],
            gas_props=gas_props,
            liquid_props=liquid_props,
            Nl=Nl,
            Ng=Ng,
            T_s_previous_step=config.T_L_INIT + 50, # mainと同様の初期推測値
            initial_Y_array=Y_g[:,0],
            fugacity_interpolator=fugacity_interpolator,
            k=0.0, # t=0では乱れはゼロと仮定
            epsilon=0.0
        )

        # --- 4. 結果を表示 ---
        print("\n--- テスト結果 ---")
        print(f"  計算された界面温度 (T_s)      : {T_s:.2f} K")
        print(f"  蒸発質量流束 (mdot)          : {mdot:.4e} kg/m^2/s")
        print(f"  気相からの熱流束 (q_gas_in)  : {-q_gas_out:.4e} W/m^2")
        print(f"  液相への熱流束 (q_liq_out)   : {q_liq_out:.4e} W/m^2")
        print(f"  蒸発潜熱 (mdot * Lv)         : {mdot * liquid_props.get_prop('heat_of_vaporization', T_s):.4e} W/m^2")
        print(f"  界面の燃料質量分率 (Y_f_eq)  : {Y_eq[gas_props.fuel_idx]:.4f}")

    except Exception as e:
        import traceback
        print("\n--- エラー ---")
        traceback.print_exc()

if __name__ == '__main__':
    run_interface_test()