import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# シミュレーションモジュールをインポート
import config
import properties
import grid
import interface
from numerics import harmonic_mean

def verify_initial_heat_flux():
    """
    ステップ2：理論的な初期熱流束とシミュレーションの初期熱流束を比較検証する。
    """
    print("--- 検証2: 初期熱流束 (q_gas) の妥当性確認 ---")

    # --- 1. 初期条件をconfigから取得 ---
    T_inf = config.T_INF_INIT
    T_s_init = config.T_L_INIT  # t=0での液滴表面温度
    R0 = config.R0
    D0 = 2 * R0
    P = config.P_INIT

    # --- 2. 理論値の計算 (Nu=2モデル) ---
    try:
        gas_props = properties.GasProperties(config.MECH_FILE)
        # 膜温度 T_film での物性値を使用
        T_film = (T_inf + T_s_init) / 2.0
        # 周囲気体の組成で状態を設定
        gas_props.set_state(T_film, P, config.X_INF_INIT)
        lambda_g_film = gas_props.gas.thermal_conductivity

        h_theory = 2.0 * lambda_g_film / D0
        q_gas_theory = h_theory * (T_inf - T_s_init)
        
        print(f"[理論値計算]")
        print(f"  膜温度 T_film: {T_film:.1f} K")
        print(f"  気相熱伝導率 @ T_film (λ_g): {lambda_g_film:.4f} W/m-K")
        print(f"  理論熱伝達率 (h_theory): {h_theory:.2f} W/m²-K")
        print(f"  ==> 理論熱流束 (q_gas_theory): {q_gas_theory:.2f} W/m²")

    except Exception as e:
        print(f"理論値の計算中にエラーが発生しました: {e}")
        return

    # --- 3. シミュレーション初期値の計算 ---
    try:
        # シミュレーションと同じ方法でグリッドとプロパティを取得
        _, r_g_nodes, _ = grid.gas_grid_fvm(R0, config.RMAX, config.NG)
        r_g_centers, _, _ = grid.gas_grid_fvm(R0, config.RMAX, config.NG)
        
        # 界面と気相第一セル中心のプロパティ
        gas_props.set_state(T_s_init, P, config.X_INF_INIT) # 界面温度での物性値 (簡易的に周囲組成で)
        lambda_g_s = gas_props.gas.thermal_conductivity
        
        # 論文の初期条件のように、気相側は均一な高温状態とする
        T_g_0 = config.T_INF_INIT
        gas_props.set_state(T_g_0, P, config.X_INF_INIT)
        lambda_g_0 = gas_props.gas.thermal_conductivity

        # 界面での熱伝導率（調和平均）
        lambda_g_face = harmonic_mean(lambda_g_s, lambda_g_0)

        # 温度勾配
        dr_centers_g = r_g_centers[0] - R0
        grad_T_sim = (T_g_0 - T_s_init) / dr_centers_g

        # シミュレーションの熱流束 (q > 0 が液滴向き)
        q_gas_sim = lambda_g_face * grad_T_sim

        print(f"\n[シミュレーション初期値計算]")
        print(f"  気相第一セル中心半径: {r_g_centers[0]*1e6:.2f} µm")
        print(f"  界面と第一セル中心の距離 (dr): {dr_centers_g*1e6:.2f} µm")
        print(f"  界面の熱伝導率 (λ_g_face): {lambda_g_face:.4f} W/m-K")
        print(f"  シミュレーションの温度勾配 (grad_T): {grad_T_sim:.2e} K/m")
        print(f"  ==> シミュレーション熱流束 (q_gas_sim): {q_gas_sim:.2f} W/m²")
        
        # --- 4. 比較 ---
        print("\n[比較結果]")
        diff = ((q_gas_sim - q_gas_theory) / q_gas_theory) * 100
        print(f"  理論値とシミュレーション値の差: {diff:.2f}%")
        if abs(diff) > 20:
            print("  ==> 警告: 理論値とシミュレーションの初期熱流束に大きな乖離があります。")
            print("      気相の熱伝導率の評価、または勾配計算に問題がある可能性があります。")
        else:
            print("  ==> 評価: 初期熱流束の計算は妥当な範囲です。")

    except Exception as e:
        print(f"シミュレーション値の計算中にエラーが発生しました: {e}")

# (verify_initial_heat_flux の定義の下に以下を追加)

def verify_interface_solver(time_step_index=1):
    """
    ステップ3：指定された時刻のデータを用いて界面エネルギーバランスの残差をプロットし、
    シミュレーションのソルバーが見つけた解と比較する。
    """
    print("--- 検証3: 界面エネルギーバランスの検証 ---")

    # --- 1. 検証対象の時刻のデータをCSVから読み込む ---
    try:
        df = pd.read_csv(os.path.join(config.OUTPUT_DIR, 'time_history_live.csv'))
        if time_step_index >= len(df):
            print(f"エラー: 指定された time_step_index ({time_step_index}) はデータ数を超えています。")
            return
        
        sim_data = df.iloc[time_step_index]
        t_sim = sim_data['Time (s)']
        R_sim = sim_data['Radius (m)']
        P_sim = sim_data['Pressure (Pa)']
        Tl_surf_cell_sim = sim_data['T_liquid_surf_cell (K)']
        Tg_surf_cell_sim = sim_data['T_gas_surf_cell (K)']
        Ts_solved_sim = sim_data['T_solved_interface (K)']
        
        print(f"検証データ: 時刻 t = {t_sim:.3e} s")
        print(f"  液滴半径 R = {R_sim*1e6:.2f} µm, 圧力 P = {P_sim/1e6:.3f} MPa")
        print(f"  液滴表面セル温度 T_l[-1] = {Tl_surf_cell_sim:.2f} K")
        print(f"  気相表面セル温度 T_g[0] = {Tg_surf_cell_sim:.2f} K")
        print(f"  シミュレーションが解いた界面温度 T_s = {Ts_solved_sim:.2f} K")

    except FileNotFoundError:
        print(f"エラー: {os.path.join(config.OUTPUT_DIR, 'time_history_live.csv')} が見つかりません。")
        return
    except Exception as e:
        print(f"CSVファイルの読み込み中にエラー: {e}")
        return

    # --- 2. プロパティとグリッドを初期化 ---
    liquid_props = properties.LiquidProperties(config.LIQUID_PROP_FILE)
    gas_props = properties.GasProperties(config.MECH_FILE)
    nsp = gas_props.nsp
    Nl, Ng = config.NL, config.NG

    # 検証時刻のグリッド情報を再生成
    r_l_centers, _, _ = grid.liquid_grid_fvm(R_sim, Nl)
    r_g_centers, _, _ = grid.gas_grid_fvm(R_sim, config.RMAX, Ng)

    # 気相第一セルの組成Y_g[0]をダミーで作成（本来はシミュレーションから取得）
    # ここでは単純化のため、初期の周囲気体組成と仮定する
    gas_props.set_state(Tg_surf_cell_sim, P_sim, config.X_INF_INIT)
    Yg_surf_cell_sim = gas_props.gas.Y

    # --- 3. T_sを変化させてエネルギー残差を計算 ---
    # T_sの推測値の範囲
    T_s_range = np.linspace(Tl_surf_cell_sim - 20, Tg_surf_cell_sim + 20, 200)
    residuals = []

    print("\nエネルギー残差を計算中...")
    for ts_guess in T_s_range:
        # interface.pyの残差計算関数を呼び出す
        # 必要な引数をシミュレーションデータから設定
        res, _, _, _ = interface._interface_energy_residual_full(
            T_s_guess=ts_guess,
            T_l_node_last=Tl_surf_cell_sim,
            T_g_node0=Tg_surf_cell_sim,
            Y_g_node0=Yg_surf_cell_sim,
            P=P_sim,
            R=R_sim,
            r_l_node_last_center=r_l_centers[-1] if Nl > 0 else 0,
            r_g_node0_center=r_g_centers[0] if Ng > 0 else R_sim,
            gas_props=gas_props,
            liquid_props=liquid_props,
            Nl=Nl
        )
        residuals.append(res)
    
    print("計算完了。プロットを生成します...")

    # --- 4. 結果をプロット ---
    plt.figure(figsize=(10, 7))
    plt.plot(T_s_range, residuals, 'b-', label='エネルギー残差 (q_gas_in - q_liq_out - q_evap)')
    
    # ゼロ線
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    
    # シミュレーションが解いたT_sをプロット
    plt.axvline(Ts_solved_sim, color='r', linestyle=':', label=f'シミュレーションの解 T_s = {Ts_solved_sim:.2f} K')
    
    # 残差がゼロになる点を計算してプロット（補間）
    try:
        # 符号が変わる点を探す
        f_interp = np.interp1d(residuals, T_s_range)
        ts_solution_from_residual = f_interp(0)
        plt.axvline(ts_solution_from_residual, color='g', linestyle='-.', label=f'残差ゼロの解 T_s = {ts_solution_from_residual:.2f} K')
    except Exception:
        print("警告: 残差ゼロの点を補間できませんでした（残差の符号が変化しない可能性があります）。")

    plt.title(f'界面エネルギーバランスの検証 (t = {t_sim:.3e} s)')
    plt.xlabel('界面温度の推測値 T_s (K)')
    plt.ylabel('エネルギー残差 (W/m²)')
    plt.legend()
    plt.grid(True, which='both', linestyle=':')
    plt.ylim(np.nanmin(residuals), np.nanmax(residuals)) # y軸範囲を自動調整
    
    plot_filename = os.path.join(config.OUTPUT_DIR, 'verification_interface_residual.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"検証プロットを保存しました: {plot_filename}")

if __name__ == '__main__':
    # 既存の出力ディレクトリが無ければ作成
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 検証1はテキストベースで完了
    
    # 検証2を実行
    verify_initial_heat_flux()

    print("\n" + "="*50 + "\n")

    # 検証3を実行 (CSVの2行目、t>0の最初のステップを使用)
    verify_interface_solver(time_step_index=1)

#if __name__ == '__main__':
#    verify_initial_heat_flux()
#    print("\n" + "="*50 + "\n")