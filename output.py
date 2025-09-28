# droplet_combustion_sim_r14/output.py (新規作成)

import numpy as np
import pandas as pd
import os
import grid as grid_utils
import config
from properties import GasProperties

def save_radial_profile_to_csv(time, step_count, R, T_l, T_g, Y_g, gas_props: GasProperties):
    """
    指定された時刻の半径方向プロファイルをCSVファイルに保存する。
    空間的な補間は行わず、計算格子上のデータを直接出力する。
    """
    if not config.SAVE_RADIAL_PROFILES:
        return

    # --- 1. 出力先ディレクトリの準備 ---
    output_path = os.path.join(config.OUTPUT_DIR, config.RADIAL_CSV_SUBDIR)
    os.makedirs(output_path, exist_ok=True)

    # --- 2. ファイル名の生成 ---
    # 0埋め5桁のステップ数と指数表記の時刻
    filename = f"{step_count:05d}_{time:.4e}.csv"
    filepath = os.path.join(output_path, filename)

    # --- 3. 計算格子の再構築と物理量の結合 ---
    Nl = len(T_l)
    Ng = len(T_g)
    
    # 液相・気相の計算メッシュ（セル中心）を再構築
    r_l_centers, _, _ = grid_utils.liquid_grid_fvm(R, Nl)
    r_g_centers, _, _ = grid_utils.gas_grid_fvm(R, config.RMAX, Ng)

    # 液相と気相の半径座標と温度を一つの配列に結合
    r_native = np.concatenate((r_l_centers, r_g_centers))
    T_native = np.concatenate((T_l, T_g))
    
    # 4. 出力用DataFrameの作成
    data = {
        'Time (s)': np.full_like(r_native, time),
        'Radius (m)': r_native,
        'Radius_normalized (-)': r_native / config.R0,
        'Temperature (K)': T_native,
    }

    # 燃料原子(C+H)の質量分率を計算して追加
    Y_fuel_elemental = np.full_like(r_native, np.nan)
    if Ng > 0:
        Y_fuel_elemental[Nl:] = gas_props.get_elemental_fuel_mass_fraction(Y_g)
    data['Y_fuel_elemental (-)'] = Y_fuel_elemental
    
    # 指定された化学種の質量分率をデータに追加
    for species_name in config.SPECIES_TO_OUTPUT:
        col_name = f'Y_{species_name} (-)'
        # DataFrameの列をNaNで初期化
        species_data = np.full_like(r_native, np.nan)
        try:
            k = gas_props.gas.species_index(species_name)
            # 気相部分にのみデータを格納
            species_data[Nl:] = Y_g[k, :]
        except ValueError:
            # メカニズムに化学種が存在しない場合でもエラーを出さず、NaNのままにする
            if step_count < 2: # 最初の数ステップのみ警告
                 print(f"Warning: Species '{species_name}' not found in mechanism. Skipping for output.")
        
        data[col_name] = species_data
        
    df_output = pd.DataFrame(data)

    # --- 5. CSVファイルへの書き出し ---
    try:
        df_output.to_csv(filepath, index=False, float_format='%.6e')
    except Exception as e:
        print(f"Error: Failed to write CSV file {filepath}. Details: {e}")