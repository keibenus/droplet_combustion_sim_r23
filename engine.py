# =========================================
#            engine.py (NEW)
# =========================================
"""
エンジン圧縮行程における周囲圧力・温度の変化を計算するモジュール。
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import config

class EngineCompressionModel:
    """
    エンジンの圧縮モデルを管理するクラス。
    configファイルの設定に基づき、時刻tにおける周囲圧力・温度を返す。
    """
    def __init__(self):
        self.model_type = config.COMPRESSION_MODEL
        print(f"Initializing EngineCompressionModel with type: '{self.model_type}'")

        if self.model_type == 'ISENTROPIC':
            # 等エントロピー圧縮モデルのパラメータを読み込み
            params = config.COMPRESSION_PARAMS
            self.P_init = config.P_INIT
            self.T_init = config.T_INF_INIT
            self.rpm = params['rpm']
            self.omega_rad_s = self.rpm * 2 * np.pi / 60.0 # rad/s
            self.cr = params['compression_ratio']
            self.L = params['connecting_rod_length']
            self.R_crank = params['crank_radius']
            self.gamma = params['gamma']
            
            # 幾何学的な初期容積比を計算 (V_init は BDC での容積)
            self.V_tdc = 1.0 # TDC容積を1とすると
            self.V_bdc = self.cr * self.V_tdc # BDC容積は圧縮比倍
            self.V_clearance = self.V_tdc # スキッシュ容積
            self.V_stroke = self.V_bdc - self.V_tdc # 行程容積
            
            # 初期クランク角をラジアンに変換
            self.theta_init_rad = np.deg2rad(params['initial_crank_angle'])

        elif self.model_type == 'FILE':
            # ファイルから圧力・温度履歴を読み込み、補間関数を作成
            try:
                df = pd.read_csv(config.COMPRESSION_FILE_PATH)
                self.time_data = df['Time'].values
                self.pressure_data = df['Pressure'].values
                self.temperature_data = df['Temperature'].values
                
                self.p_interp = interp1d(self.time_data, self.pressure_data,
                                         kind='linear', bounds_error=False,
                                         fill_value=(self.pressure_data[0], self.pressure_data[-1]))
                self.t_interp = interp1d(self.time_data, self.temperature_data,
                                         kind='linear', bounds_error=False,
                                         fill_value=(self.temperature_data[0], self.temperature_data[-1]))
                print(f"Loaded compression data from '{config.COMPRESSION_FILE_PATH}'")
            except FileNotFoundError:
                print(f"ERROR: Compression data file not found: {config.COMPRESSION_FILE_PATH}")
                raise
            except Exception as e:
                print(f"ERROR: Failed to process compression data file: {e}")
                raise

    def get_ambient_conditions(self, t: float):
        """
        指定された時刻 t における周囲の圧力 [Pa] と温度 [K] を返す。
        """
        if self.model_type == 'NONE':
            return config.P_INIT, config.T_INF_INIT

        elif self.model_type == 'ISENTROPIC':
            # 時刻 t におけるクランク角を計算
            theta = self.theta_init_rad + self.omega_rad_s * t
            
            # ピストン位置からシリンダー容積を計算
            # s: クランクシャフト中心からピストントップまでの距離
            s = self.R_crank * np.cos(theta) + np.sqrt(self.L**2 - (self.R_crank * np.sin(theta))**2)
            
            # x: ピストンのTDCからの移動距離
            x = self.L + self.R_crank - s
            
            # 現在の容積 V(t)
            # A: シリンダー断面積 (直接は不要なので V_stroke で代用)
            # V(t) = V_clearance + A * x
            # V_stroke = A * 2 * R_crank
            # --> A = V_stroke / (2 * R_crank)
            # V(t) = V_clearance + (V_stroke / (2 * R_crank)) * x
            
            # 容積比で計算する方が簡単
            # V(theta) / V_tdc = 1 + 0.5 * (cr - 1) * [ (L/R) + 1 - cos(theta) - sqrt((L/R)^2 - sin^2(theta)) ]
            # L/R = self.L / self.R_crank
            term1 = (self.L / self.R_crank) + 1 - np.cos(theta)
            term2 = np.sqrt((self.L / self.R_crank)**2 - np.sin(theta)**2)
            V_ratio_to_tdc = 1 + 0.5 * (self.cr - 1) * (term1 - term2)
            
            # V_init は BDC なので V_bdc。V_bdc / V_tdc = cr
            # (V_init / V(t))^gamma = ( (V_bdc/V_tdc) / (V(t)/V_tdc) )^gamma
            V_ratio = self.cr / V_ratio_to_tdc
            
            P_t = self.P_init * (V_ratio)**self.gamma
            T_t = self.T_init * (V_ratio)**(self.gamma - 1)
            
            return P_t, T_t

        elif self.model_type == 'FILE':
            P_t = self.p_interp(t)
            T_t = self.t_interp(t)
            return P_t.item(), T_t.item()

        else:
            raise ValueError(f"Unknown COMPRESSION_MODEL: {self.model_type}")