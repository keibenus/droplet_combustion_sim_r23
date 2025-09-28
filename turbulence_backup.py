# turbulence.py (新規作成)
import numpy as np
import pandas as pd
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
            # (engine.pyと同様にファイル読み込みと補間関数を作成)
            pass

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
            # (補間関数から値を返す)
            pass
        else:
            return 0.0, 0.0