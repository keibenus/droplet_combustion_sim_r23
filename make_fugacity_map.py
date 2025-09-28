# -*- coding: utf-8 -*-
"""
【真・最終確定版】流体相フガシティ係数マップ生成コード (thermo==0.4.2 対応)

目的:
thermo v0.4.2の仕様に基づき、eos_mixモジュールを直接使用して、物理的に存在する
流体相（気相、超臨界相など）のフガシティ係数を安定的かつ網羅的に計算し、
CSVファイルに出力する。
"""
import numpy as np
import pandas as pd
from thermo import Mixture, Chemical
from thermo.eos_mix import PRMIX, SRKMIX # 低レベルAPIに戻る
from tqdm import tqdm
import itertools
import sys
import os

def generate_fluid_phase_fugacity_map(
    fuel_name: str,
    gas_composition: dict,
    T_range: tuple,
    P_range: tuple,
    fuel_mass_frac_range: tuple,
    eos_name: str,
    output_filename: str
) -> None:
    # ... (関数の引数説明は省略) ...
    gas_names = list(gas_composition.keys())
    component_names = [fuel_name] + gas_names

    try:
        chemicals = [Chemical(name) for name in component_names]
        Tcs = [c.Tc for c in chemicals]
        Pcs = [c.Pc for c in chemicals]
        omegas = [c.omega for c in chemicals]
    except Exception as e:
        print(f"Error: Could not retrieve properties for one of the chemicals: {component_names}. Details: {e}")
        return

    if eos_name.upper() == 'PR':
        eos_model = PRMIX
    elif eos_name.upper() == 'SRK':
        eos_model = SRKMIX
    else:
        raise ValueError("EOS model must be 'PR' or 'SRK'")

    print(f"Components: {component_names}")
    print(f"EOS: {eos_name.upper()}")

    #T_values = np.linspace(T_range[0], T_range[1], T_range[2])
    #P_values = np.linspace(P_range[0], P_range[1], P_range[2])
    T_values = np.array([260,280,300,320,340,360,380,400,420,440,460,480,500,550,600,650,700,750,800,900,1000])
    P_values = np.array([1,2,3,4,5,7,10,15,20,25,30,35,40,45,50,55,60,70,80,90,100])*1e5
    fuel_mass_frac_values = np.linspace(fuel_mass_frac_range[0], fuel_mass_frac_range[1], fuel_mass_frac_range[2])
    
    param_combinations = list(itertools.product(T_values, P_values, fuel_mass_frac_values))
    results = []
    
    # 周囲気体の平均分子量を計算
    air_mix_for_mw = Mixture(IDs=gas_names, zs=list(gas_composition.values()))
    gas_avg_mw = air_mix_for_mw.MW

    print("\nCalculating Fugacity Coefficients for Fluid-Phase...")
    for T, P, fuel_wf in tqdm(param_combinations, file=sys.stdout):
        
        ws = [fuel_wf] + [ (1.0 - fuel_wf) * (air_mix_for_mw.Chemicals[i].MW * mol_frac / gas_avg_mw) 
                           for i, (name, mol_frac) in enumerate(gas_composition.items()) ]
        
        try:
            # wsからzsへの変換
            zs = Mixture(IDs=component_names, ws=ws).zs
            # eos_mixモジュールを直接使用
            mixture = eos_model(T=T, P=P, Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs)
            
            fluid_phis = None
            
            # --- 診断結果に基づく、v0.4.2で確実に動作するロジック ---
            try:
                # 1. 気相の係数取得を試みる
                fluid_phis = mixture.phis_g
            except AttributeError:
                # 2. 失敗した場合、液体または超臨界相の係数取得を試みる
                try:
                    fluid_phis = mixture.phis_l
                except AttributeError:
                    # 3. 両方失敗した場合、取得できないと判断
                    fluid_phis = None

            if fluid_phis is None:
                tqdm.write(f"Info: No valid fluid phase fugacity could be calculated at T={T:.2f} K, P={P/1e6:.2f} MPa, wf_fuel={fuel_wf:.3f}. Skipping.")
                continue

            res = {'Temperature (K)': T, 'Pressure (Pa)': P}
            for i, name in enumerate(component_names):
                res[f'{name}_mass_fraction'] = ws[i]
                res[f'{name}_mole_fraction'] = zs[i]
                res[f'{name}_fugacity_coeff_Fluid'] = fluid_phis[i]
            results.append(res)

        except Exception as e:
            tqdm.write(f"Warning: An unexpected error occurred at T={T:.2f} K, P={P/1e6:.2f} MPa, wf_fuel={fuel_wf:.3f}. Error: {e}")
            continue

    if not results:
        print("\nNo data was generated. Please check parameter ranges and potential calculation errors.")
        return

    df = pd.DataFrame(results)
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df.to_csv(output_filename, index=False)
    print(f"\nFluid-phase fugacity map has been successfully saved to '{output_filename}'")

if __name__ == '__main__':
    # =========================================================================
    # --- 入力パラメータ設定 ---
    # =========================================================================
    FUEL_NAME = 'n-heptane'
    AIR_COMPOSITION = {'N2': 0.79, 'O2': 0.21}
    TEMPERATURE_RANGE = (270.0, 1000.0, 15)
    PRESSURE_RANGE = (0.1e6, 10.0e6, 20)
    FUEL_MASS_FRACTION_RANGE = (0.0, 1.0, 21)
    EOS = 'SRK'
    OUTPUT_FILE = f'fugacity_map_Fluid_{FUEL_NAME}_{EOS}.csv'
    # =========================================================================
    
    generate_fluid_phase_fugacity_map(
        fuel_name=FUEL_NAME,
        gas_composition=AIR_COMPOSITION,
        T_range=TEMPERATURE_RANGE,
        P_range=PRESSURE_RANGE,
        fuel_mass_frac_range=FUEL_MASS_FRACTION_RANGE,
        eos_name=EOS,
        output_filename=OUTPUT_FILE
    )