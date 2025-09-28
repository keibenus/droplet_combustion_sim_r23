# =========================================
#         plotting.py (r4)
# =========================================
# plotting.py
"""Functions for plotting simulation results (works with FVM cell center data)."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # For formatting axes
import numpy as np
from scipy.interpolate import interp1d
import os
import pandas as pd
import config
import grid # To regenerate grids for plotting
from properties import GasProperties # For type hinting

plt.rcParams.update({'font.size': 12}) # Adjust font size for readability

def plot_results(times, results_list, output_dir, nsp, Nl_init, Ng_init, gas_props: GasProperties):
    """
    Generates and saves plots from a list of state dictionaries (cell center values).
    Uses initial Nl/Ng values (Nl_init, Ng_init) for consistency in array access,
    but handles the transition to gas-only phase (Nl becomes 0).
    """
    print("Generating final plots...")
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    times_array = np.array(times)
    num_times = len(results_list)
    if num_times <= 1: # Need at least initial and one more point
        print("Not enough time points saved for plotting meaningful history.")
        return

    # --- Extract History Data ---
    Tl_surf_hist = np.full(num_times, np.nan) # Use NaN as default
    Tg_surf_hist = np.full(num_times, np.nan)
    T_interface_hist = np.full(num_times, np.nan) # <<< 追加 >>>
    R_hist = np.full(num_times, np.nan)
    P_hist = np.full(num_times, np.nan)
    maxTgas_hist = np.full(num_times, np.nan)
    fuel_idx = gas_props.fuel_idx if gas_props else -1

    # --- Extract data from results_list ---
    # We need T_s (interface temperature) history if available, but it wasn't explicitly saved
    # Let's try to get it from the CSV log file instead, as it's saved there.
    csv_log_file = os.path.join(config.OUTPUT_DIR, 'time_history_live.csv')
    log_data = None
    log_times = np.array([])
    try:
        log_data = pd.read_csv(csv_log_file)
        # Ensure log_times aligns with simulation times (using tolerance)
        log_times = log_data['Time (s)'].values
        T_interface_log = log_data['T_solved_interface (K)'].values
        # Interpolate log data onto simulation save times
        if len(log_times) > 1:
             interp_Ts = interp1d(log_times, T_interface_log, kind='linear', bounds_error=False, fill_value=np.nan)
             T_interface_hist = interp_Ts(times_array)
        else:
             T_interface_hist.fill(np.nan) # Not enough log points to interpolate

    except FileNotFoundError:
        print(f"Warning: Log file '{csv_log_file}' not found. Interface temperature history plot unavailable.")
        T_interface_hist.fill(np.nan)
    except Exception as e:
        print(f"Warning: Error reading or interpolating log file '{csv_log_file}': {e}")
        T_interface_hist.fill(np.nan)


    for i, state_dict in enumerate(results_list):
        # Check if keys exist and arrays have expected length based on initial grid sizes
        # Handle potential transition to gas-only phase where T_l might be empty
        T_l_state = state_dict.get('T_l', np.array([]))
        T_g_state = state_dict.get('T_g', np.array([]))
        R_state = state_dict.get('R', np.nan)
        P_state = state_dict.get('P', np.nan)

        if len(T_l_state) > 0: # Check if liquid phase exists in this state
            Tl_surf_hist[i] = T_l_state[-1]
        #else: Tl_surf_hist[i] = np.nan # Already initialized to NaN

        if len(T_g_state) > 0:
            Tg_surf_hist[i] = T_g_state[0]
            maxTgas_hist[i] = np.max(T_g_state)
        #else: Tg_surf_hist[i] = np.nan; maxTgas_hist[i] = np.nan

        R_hist[i] = R_state
        P_hist[i] = P_state

    # --- Plot 1: Surface/Interface Temperatures vs Time ---
    plt.figure(figsize=(10, 6))
    valid_Tl = ~np.isnan(Tl_surf_hist)
    valid_Tg = ~np.isnan(Tg_surf_hist)
    valid_Ts = ~np.isnan(T_interface_hist)

    if np.any(valid_Tl): plt.plot(times_array[valid_Tl], Tl_surf_hist[valid_Tl], 'b.-', markersize=4, label=f'Liquid Surf Cell (j={Nl_init-1})')
    if np.any(valid_Tg): plt.plot(times_array[valid_Tg], Tg_surf_hist[valid_Tg], 'r.-', markersize=4, label=f'Gas Surf Cell (i=0)')
    if np.any(valid_Ts): plt.plot(times_array[valid_Ts], T_interface_hist[valid_Ts], 'g.--', markersize=4, label='Interface Temp (Solved)')
    if np.any(valid_Tg): plt.plot(times_array[valid_Tg], maxTgas_hist[valid_Tg], 'm:', markersize=4, label='Max Gas Temp')

    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title('Characteristic Temperatures vs Time')
    plt.legend()
    plt.grid(True, which='both', linestyle=':')
    # Determine reasonable Y limits
    all_temps = np.concatenate([Tl_surf_hist, Tg_surf_hist, T_interface_hist, maxTgas_hist])
    min_temp = np.nanmin(all_temps) if not np.all(np.isnan(all_temps)) else config.T_L_INIT
    max_temp = np.nanmax(all_temps) if not np.all(np.isnan(all_temps)) else config.T_INF_INIT
    plt.ylim(bottom=max(0, min_temp - 100), top=max_temp + 200)
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e')) # Scientific notation for time
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_history.png'))
    plt.close()

    # --- Plot 2: Droplet Radius vs Time ---
    valid_R = ~np.isnan(R_hist)
    if np.any(valid_R):
        plt.figure(figsize=(10, 6))
        plt.plot(times_array[valid_R], R_hist[valid_R] * 1e6, 'k.-', markersize=4) # Radius in micrometers
        plt.xlabel('Time (s)')
        plt.ylabel('Droplet Radius (μm)')
        plt.title(f'Droplet Radius vs Time (Initial R0 = {config.R0*1e6:.1f} μm)')
        plt.grid(True, which='both', linestyle=':')
        plt.ylim(bottom=0)
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'droplet_radius.png'))
        plt.close()

    # --- Plot 3: Pressure vs Time ---
    valid_P = ~np.isnan(P_hist)
    if np.any(valid_P):
        plt.figure(figsize=(10, 6))
        plt.plot(times_array[valid_P], P_hist[valid_P] / 1e6, 'k.-', markersize=4) # Pressure in MPa
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (MPa)')
        plt.title(f'Vessel Pressure vs Time (Initial P = {config.P_INIT/1e6:.2f} MPa)')
        plt.grid(True, which='both', linestyle=':')
        # plt.ylim(bottom=0)
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pressure_history.png'))
        plt.close()


    # --- Plot 4: Temperature Profiles at Different Times (from CSV files) ---
    plt.figure(figsize=(12, 7))
    
    # CSVファイルが保存されているディレクトリのパス
    profiles_dir = os.path.join(output_dir, config.RADIAL_CSV_SUBDIR)
    
    plotted_count = 0
    try:
        # ディレクトリ内のCSVファイルリストを取得し、名前順（=時間順）にソート
        all_files = sorted([f for f in os.listdir(profiles_dir) if f.endswith('.csv')])
        
        if not all_files:
            print("Warning: No radial profile CSV files found to plot.")
        else:
            # プロットするファイルの数とインデックスを決定 (最初、最後、およびその間の等間隔な点)
            num_plots = min(len(all_files), 7) # 最大7つのプロファイルをプロット
            plot_indices = np.unique(np.linspace(0, len(all_files) - 1, num_plots, dtype=int))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(plot_indices)))

            print("\nPlotting radial profiles from CSV files:")
            for i, file_index in enumerate(plot_indices):
                filename = all_files[file_index]
                filepath = os.path.join(profiles_dir, filename)
                
                try:
                    # CSVファイルを読み込む
                    df_profile = pd.read_csv(filepath)
                    
                    # データを抽出
                    # ご要望通り、補間せず計算格子上の点をそのままプロットします
                    r_norm = df_profile['Radius_normalized (-)']
                    temp = df_profile['Temperature (K)']
                    time_val = df_profile['Time (s)'].iloc[0] # ファイル内の時間はすべて同じ
                    
                    # プロット実行
                    plt.plot(r_norm, temp, '.-', color=colors[i], label=f't = {time_val:.3e} s', markersize=4, linewidth=1.5)
                    plotted_count += 1
                    print(f"  - Plotted {filename}")

                except Exception as e:
                    print(f"Warning: Could not read or plot {filename}. Error: {e}")

    except FileNotFoundError:
        print(f"Warning: Directory '{profiles_dir}' not found. Skipping profile plots.")
    
    if plotted_count > 0:
        plt.xlabel('Radius / Initial Radius (-)')
        plt.ylabel('Temperature (K)')
        plt.title(f'Temperature Profiles (Rmax/R0 = {config.R_RATIO:.2f})')
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle=':')
        plt.xlim(left=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temperature_profiles.png'))
    plt.close()

    # --- Plot 5: n-heptane Mass Fraction Profile (from CSV files) ---
    plt.figure(figsize=(12, 7))
    
    # 燃料の列名をconfigから動的に生成
    fuel_col_name = f'Y_{config.FUEL_SPECIES_NAME} (-)'
    
    plotted_count_fuel_species = 0
    # profiles_dir, all_files, plot_indices は Plot 4 のブロックで定義済みと仮定
    if 'all_files' in locals() and all_files:
        colors = plt.cm.viridis(np.linspace(0, 1, len(plot_indices)))
        
        for i, file_index in enumerate(plot_indices):
            filepath = os.path.join(profiles_dir, all_files[file_index])
            
            try:
                df_profile = pd.read_csv(filepath)
                if fuel_col_name in df_profile.columns:
                    r_norm = df_profile['Radius_normalized (-)']
                    fuel_frac = df_profile[fuel_col_name]
                    time_val = df_profile['Time (s)'].iloc[0]
                    
                    plt.plot(r_norm, fuel_frac, '.-', color=colors[i], label=f't = {time_val:.3e} s', markersize=4, linewidth=1.5)
                    plotted_count_fuel_species += 1
                elif i == 0:
                    print(f"Warning: Fuel column '{fuel_col_name}' not found in CSV files.")
            except Exception as e:
                print(f"Warning: Could not process {all_files[file_index]} for fuel plot. Error: {e}")

    if plotted_count_fuel_species > 0:
        plt.xlabel('Radius / Initial Radius (-)')
        plt.ylabel(f'{config.FUEL_SPECIES_NAME} Mass Fraction')
        plt.title(f'{config.FUEL_SPECIES_NAME} (Species) Mass Fraction Profiles')
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle=':')
        plt.xlim(left=0)
        plt.ylim(bottom=-0.05, top=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fuel_species_fraction_profiles.png'))
    plt.close()

    # --- Plot 6: Elemental Fuel Mass Fraction (C+H) Profile (from CSV files) ---
    plt.figure(figsize=(12, 7))
    
    elemental_fuel_col_name = 'Y_fuel_elemental (-)'
    plotted_count_elemental = 0
    
    if 'all_files' in locals() and all_files:
        colors = plt.cm.viridis(np.linspace(0, 1, len(plot_indices)))
        
        for i, file_index in enumerate(plot_indices):
            filepath = os.path.join(profiles_dir, all_files[file_index])
            
            try:
                df_profile = pd.read_csv(filepath)
                if elemental_fuel_col_name in df_profile.columns:
                    r_norm = df_profile['Radius_normalized (-)']
                    fuel_frac = df_profile[elemental_fuel_col_name]
                    time_val = df_profile['Time (s)'].iloc[0]
                    
                    plt.plot(r_norm, fuel_frac, '.-', color=colors[i], label=f't = {time_val:.3e} s', markersize=4, linewidth=1.5)
                    plotted_count_elemental += 1
                elif i == 0:
                    print(f"Warning: Elemental fuel column '{elemental_fuel_col_name}' not found in CSV files.")
            except Exception as e:
                print(f"Warning: Could not process {all_files[file_index]} for elemental fuel plot. Error: {e}")

    if plotted_count_elemental > 0:
        plt.xlabel('Radius / Initial Radius (-)')
        plt.ylabel('Elemental Fuel (C+H) Mass Fraction')
        plt.title('Elemental Fuel (C+H) Mass Fraction Profiles (Moriue et al. Style)')
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle=':')
        plt.xlim(left=0)
        plt.ylim(bottom=-0.05, top=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fuel_elemental_fraction_profiles.png'))
    plt.close()

    # --- Plot 7: Other Species Profiles (from CSV files) ---
    # configで指定された化学種の中から、燃料を除いたものをプロット対象とする
    species_to_plot_other = [sp for sp in config.SPECIES_TO_OUTPUT if sp != config.FUEL_SPECIES_NAME]
    
    if species_to_plot_other: # プロット対象の化学種がある場合のみ実行
        plt.figure(figsize=(12, 7))
        
        plot_handles = {} # 凡例をまとめるための辞書
        plotted_count = 0
        
        if all_files: # ファイルが存在する場合のみ実行
            colors = plt.cm.viridis(np.linspace(0, 1, len(plot_indices)))
            
            for i, file_index in enumerate(plot_indices):
                filepath = os.path.join(profiles_dir, all_files[file_index])
                
                try:
                    df_profile = pd.read_csv(filepath)
                    r_norm = df_profile['Radius_normalized (-)']
                    time_val = df_profile['Time (s)'].iloc[0]

                    # 指定された各化学種をプロット
                    for sp_name in species_to_plot_other:
                        col_name = f'Y_{sp_name} (-)'
                        if col_name in df_profile.columns:
                            # 化学種ごとに線のスタイルを変える
                            line_style = '-'
                            if 'o2' in sp_name: line_style = '--'
                            elif 'co2' in sp_name: line_style = ':'
                            elif 'h2o' in sp_name and 'h2o2' not in sp_name: line_style = '-.'
                            
                            # 凡例が重複しないように、各種について初回のみラベルを付ける
                            label = f'{sp_name}' if i == 0 else ""
                            
                            handle, = plt.plot(r_norm, df_profile[col_name], 
                                               linestyle=line_style, color=colors[i],
                                               label=label, linewidth=1.5)
                            
                            if i == 0:
                                plot_handles[sp_name] = handle
                    plotted_count += 1
                except Exception as e:
                    print(f"Warning: Could not process {all_files[file_index]} for other species plot. Error: {e}")

        if plotted_count > 0:
            # 凡例を作成
            # 時間の経過を色で示すテキストを追加
            time_legend_text = f"Color: Early (Purple) -> Late (Yellow)"
            
            # 凡例のハンドルとラベルを取得
            legend_handles = list(plot_handles.values())
            legend_labels = list(plot_handles.keys())
            
            # 凡例をプロットに追加
            plt.legend(legend_handles, legend_labels, title=f"Species Profiles\n{time_legend_text}", fontsize=10)

            plt.xlabel('Radius / Initial Radius (-)')
            plt.ylabel('Mass Fraction')
            plt.title('Key Species Mass Fraction Profiles')
            plt.grid(True, which='both', linestyle=':')
            plt.xlim(left=0)
            plt.ylim(bottom=-0.05, top=1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'other_species_profiles.png'))
        else:
            print("No valid data to plot other species profiles.")
    else:
        print("No other species specified in config.SPECIES_TO_OUTPUT to plot.")
        
    plt.close()
    
    print(f"Plots saved in '{output_dir}' directory.")