import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

# --- 物理定数と条件 ---
# 油滴の物性
rho_p = 850.0  # 油滴の密度 (kg/m^3)

# 周囲ガスの物性 (800K, 5MPaの空気を想定)
rho_g = 21.8   # ガスの密度 (kg/m^3)
mu_g = 3.7e-5  # ガスの粘性係数 (Pa*s)

# --- 計算関数 ---
def calculate_relaxation_time(d_p, v_rel):
    """高レイノルズ数域を考慮した粒子緩和時間を計算する"""
    if v_rel == 0:
        return np.inf
    # レイノルズ数の計算
    re = rho_g * v_rel * d_p / mu_g
    
    # 抗力係数Cdの計算 (Schiller-Naumannの式)
    if re > 0:
        c_d = (24.0 / re) * (1.0 + 0.15 * re**0.687)
    else:
        c_d = 0
        
    # 緩和時間の計算
    tau_p = (4.0 * rho_p * d_p) / (3.0 * rho_g * v_rel * c_d)
    return tau_p

# --- プロット用のグリッドデータ作成 ---
# 粒子径の範囲 (1um to 300um, 対数スケール)
d_p_range = np.logspace(-6, -3.5, 100)
# 初期相対速度の範囲 (1 m/s to 30 m/s)
v_rel_range = np.linspace(1, 30, 100)

# 2Dグリッドの作成
D, V = np.meshgrid(d_p_range, v_rel_range)

# 各グリッド点での緩和時間を計算
T_p = np.zeros_like(D)
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        T_p[i, j] = calculate_relaxation_time(D[i, j], V[i, j])

# --- 描画 ---
#plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 9))

# 対数スケールでコンター図を作成
levels = np.logspace(-5, -1, 20) # 10us to 100ms
contourf = ax.contourf(D * 1e6, V, T_p * 1e3, levels=levels, norm=LogNorm(), cmap='viridis')

# カラーバーの設定
cbar = fig.colorbar(contourf, ax=ax)
cbar.set_label('Relaxation Time (τp) [ms]', fontsize=14)

# コンターラインの描画
contour_lines = ax.contour(D * 1e6, V, T_p * 1e3, levels=[0.1, 1.5, 8.0], colors='white', linewidths=2)
ax.clabel(contour_lines, inline=True, fontsize=12, fmt='%1.1f ms', colors='white')

# 軸ラベルとタイトルの設定
ax.set_xscale('log')
ax.set_xlabel('Oil Diameter (d) [µm]', fontsize=14)
ax.set_ylabel('Initial Velocity (v_rel) [m/s]', fontsize=14)
ax.set_title('Oil Relaxation time and forrow to flow', fontsize=18, pad=20)

# 注釈の追加
ax.text(5, 25, 'enable follow to fine eddy\n(τp << 1.5ms)', color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))
ax.text(80, 15, 'unenable follow to macro frow\n(τp > 8.0ms)', color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))
ax.text(40, 5, 'unenable follow to eddy, \nenable macro flow\n(1.5ms < τp < 8.0ms)', color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))


ax.grid(True, which="both", ls="--", color='gray', alpha=0.5)

plt.savefig('relaxation_time.png')

plt.show()