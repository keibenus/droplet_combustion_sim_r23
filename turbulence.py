# turbulence.py (新規作成)
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import config

class TurbulenceModel:
    """
    シミュレーション中の周囲の乱れ状態（k, ε）を管理するクラス。
    """
    def __init__(self):
        self.model_type = config.TURBULENCE_MODEL
        if self.model_type == 'CONSTANT':
            self.k_const = config.TURBULENCE_CONSTANT_PARAMS['k']
            self.epsilon_const = config.TURBULENCE_CONSTANT_PARAMS['epsilon']
        
        elif self.model_type == 'FILE':
            try:
                df = pd.read_csv(config.TURBULENCE_FILE_PATH)
                # 必要な列が存在するかチェック
                if not all(col in df.columns for col in ['Time', 'k', 'epsilon']):
                    raise ValueError("CSV must contain 'Time', 'k', and 'epsilon' columns.")

                self.time_data = df['Time'].values
                self.k_data = df['k'].values
                self.epsilon_data = df['epsilon'].values
                
                # 線形補間関数を作成
                # bounds_error=False: 範囲外の時刻が来たら、最初と最後の値で埋める
                self.k_interp = interp1d(self.time_data, self.k_data,
                                         kind='linear', bounds_error=False,
                                         fill_value=(self.k_data[0], self.k_data[-1]))
                self.epsilon_interp = interp1d(self.time_data, self.epsilon_data,
                                               kind='linear', bounds_error=False,
                                               fill_value=(self.epsilon_data[0], self.epsilon_data[-1]))
                print(f"Loaded turbulence data from '{config.TURBULENCE_FILE_PATH}'")
            except FileNotFoundError:
                print(f"ERROR: Turbulence data file not found: {config.TURBULENCE_FILE_PATH}")
                raise
            except Exception as e:
                print(f"ERROR: Failed to process turbulence data file: {e}")
                raise
        

    def get_turbulence_properties(self, t: float):
        """
        指定された時刻tにおける乱流エネルギーk [m^2/s^2] と
        散逸率epsilon [m^2/s^3] を返す。
        """
        if self.model_type == 'NONE':
            return 0.0, 0.0
        elif self.model_type == 'CONSTANT':
            return self.k_const, self.epsilon_const
        elif self.model_type == 'FILE':
            k_t = self.k_interp(t)
            epsilon_t = self.epsilon_interp(t)
            # interp1dはnumpy配列を返すことがあるため、スカラー値に変換
            return k_t.item(), epsilon_t.item()
        else:
            return 0.0, 0.0