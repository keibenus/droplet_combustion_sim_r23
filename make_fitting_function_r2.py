# -*- coding: utf-8 -*-
"""
【誤差解析機能付き】NumPyによる最小二乗法フィッティングコード

目的:
多項式回帰を行い、係数を出力するとともに、モデルの予測誤差を評価する。
元のデータとの誤差が大きい点をリストアップし、モデルの弱点を分析する。
"""
import pandas as pd
import numpy as np
from itertools import combinations_with_replacement

def create_polynomial_features(x, degree, input_column_names):
    """
    入力データから、指定した次数の多項式特徴量（設計行列）を生成する。
    """
    n_samples, n_features = x.shape
    
    def get_powers():
        for d in range(degree + 1):
            for p in combinations_with_replacement(range(n_features), d):
                yield p

    powers = list(get_powers())
    n_output_features = len(powers)
    
    X_poly = np.empty((n_samples, n_output_features), dtype=x.dtype)
    term_names = []
    
    for i, p in enumerate(powers):
        X_poly[:, i] = np.prod(x[:, p], axis=1)
        if not p:
            term_names.append("1 (Intercept)")
        else:
            term_names.append(" * ".join([input_column_names[j] for j in p]))
            
    return X_poly, term_names

def generate_poly_coeffs_with_error_analysis(
    csv_filename: str,
    input_columns: list,
    target_column: str,
    polynomial_degree: int
):
    # ... (関数の引数説明は省略) ...
    print(f"--- Starting Polynomial Fitting with Error Analysis (Degree: {polynomial_degree}) ---")

    # 1. データの読み込みと準備
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: The file '{csv_filename}' was not found.")
        return
    df.dropna(inplace=True)
    X_raw = df[input_columns].values
    y = df[target_column].values

    # 2. データの正規化
    X_min = X_raw.min(axis=0)
    X_max = X_raw.max(axis=0)
    X_norm = (X_raw - X_min) / (X_max - X_min)

    # 3. 設計行列の作成
    term_labels = ['T', 'P', 'Z']
    design_matrix, term_names = create_polynomial_features(X_norm, polynomial_degree, term_labels)
    
    # 4. 正規方程式を解き、係数を求める
    print("\nSolving for coefficients...")
    XTX = design_matrix.T @ design_matrix
    XTy = design_matrix.T @ y
    try:
        coeffs = np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError:
        print("Error: Could not solve. The matrix may be singular. Try a lower degree.")
        return

    # 5. モデル評価 (R2乗値)
    y_pred_raw = design_matrix @ coeffs
    y_pred = np.where(y_pred_raw>0, y_pred_raw, 0)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"Model R-squared: {r2:.8f}")

    # --- ここからが追加機能 ---
    # 6. 誤差解析
    print("\n--- Error Analysis (Worst Fitting Points) ---")
    
    # 元のDataFrameに予測値と相対誤差を追加
    df['predicted_phi'] = y_pred
    
    # ゼロ割を避けるため、分母に微小量を追加
    df['relative_error'] = np.abs((df[target_column] - df['predicted_phi']) / (df[target_column] + 1e-12))
    
    # 相対誤差の大きい順にソート
    df_sorted_by_error = df.sort_values(by='relative_error', ascending=False)
    
    # 表示する列を定義
    display_columns = input_columns + [target_column, 'predicted_phi', 'relative_error']
    
    # 再現率が悪い上位15件を表示
    print("Top 15 points with the largest relative error:")
    # 表示形式を整える
    with pd.option_context('display.max_rows', None, 
                           'display.max_columns', None, 
                           'display.width', 1000,
                           'display.float_format', '{:.6f}'.format):
        print(df_sorted_by_error[display_columns].head(500))

    # --- パラメータと係数の出力は継続 ---
    print("\n" + "="*50)
    print("--- Parameters and Coefficients Summary ---")
    print("="*50)
    print("\n1. Normalization Parameters")
    for i, col in enumerate(input_columns):
        print(f"  {col}: MIN = {X_min[i]:.10e}, MAX = {X_max[i]:.10e}")
    print("\n2. Polynomial Coefficients (for normalized variables T, P, Z)")
    for name, coeff in zip(term_names, coeffs):
        print(f"    Term: {name:<25} | Coeff: {coeff:.15e}")
    
    print("\n--- Process Finished ---")

if __name__ == '__main__':
    # =========================================================================
    # --- パラメータ設定 ---
    # =========================================================================
    CSV_FILE = 'fugacity_map_Fluid_n-heptane_PR.csv'
    
    INPUT_VARS = [
        'Temperature (K)', 
        'Pressure (Pa)', 
        'n-heptane_mole_fraction'
    ]
    
    TARGET_VAR = 'n-heptane_fugacity_coeff_Fluid'
    
    POLY_DEGREE = 6
    
    # =========================================================================
    # --- 実行 ---
    # =========================================================================
    generate_poly_coeffs_with_error_analysis(
        csv_filename=CSV_FILE,
        input_columns=INPUT_VARS,
        target_column=TARGET_VAR,
        polynomial_degree=POLY_DEGREE
    )