# =========================================
#        properties.py (r4)
# =========================================
# properties.py
"""
Handles liquid and gas phase properties.
Includes reading liquid data from CSV and interfacing with Cantera for gas,
with optional Redlich-Kwong corrections for density.
Implements diffusion coefficient options.
"""
import numpy as np
import pandas as pd
import cantera as ct
from scipy.interpolate import interp1d
from scipy.optimize import newton # Using Newton solver for cubic EOS
from scipy.interpolate import RegularGridInterpolator
import warnings
import config

# --- Suppress potential ComplexWarning from numpy roots ---
# warnings.filterwarnings('ignore', category=np.ComplexWarning)

class LiquidProperties:
    """Reads and interpolates liquid properties from a CSV file."""
    def __init__(self, filename=config.LIQUID_PROP_FILE, fuel_molar_mass=0.1002):
        try:
            self.df = pd.read_csv(filename)
            self.df.sort_values(by='temperature', inplace=True)
            if self.df.isnull().values.any():
                print(f"Warning: NaN values found in {filename}. Filling NAs.")
                self.df.fillna(method='ffill', inplace=True) # Forward fill first
                self.df.fillna(method='bfill', inplace=True) # Then backward fill
        except FileNotFoundError:
            raise FileNotFoundError(f"Liquid property file '{filename}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error reading liquid property file '{filename}': {e}")

        self.fuel_molar_mass = fuel_molar_mass # 燃料モル質量 [kg/mol] をインスタンス変数として保持

        self._create_interpolators()
        self.T_min = self.df['temperature'].min()
        self.T_max = self.df['temperature'].max()
        print(f"Liquid properties loaded. Temp range: [{self.T_min:.1f} K - {self.T_max:.1f} K]")

    def _create_interpolators(self):
        self.interpolators = {}
        for col in self.df.columns:
            if col != 'temperature':
                # Check if data is monotonic before interpolating
                if not self.df[col].is_monotonic_increasing and not self.df[col].is_monotonic_decreasing:
                     # Allow non-monotonic for viscosity etc. but maybe warn
                     # print(f"Note: Liquid property '{col}' is not monotonic.")
                     pass
                self.interpolators[col] = interp1d(
                    self.df['temperature'], self.df[col],
                    kind='linear', # Linear interpolation is generally safer
                    fill_value="extrapolate", # Extrapolate outside the range
                    bounds_error=False
                )

    def get_properties(self, T):
        """Returns a dictionary of liquid properties at temperature T (K). T can be scalar or array."""
        props = {}
        # Ensure T is a NumPy array for consistent handling, even if scalar
        T_arr = np.asarray(T)
        T_clipped = np.clip(T_arr, self.T_min, self.T_max)

        is_scalar_input = T_arr.ndim == 0

        # --- Check for clipping (slightly modified to handle arrays) ---
        was_clipped = False
        if is_scalar_input:
            if T != T_clipped:
                was_clipped = True
        else: # Array input
            if np.any(T != T_clipped):
                was_clipped = True

        if was_clipped and config.LOG_LEVEL >= 1:
            # Optional: Add a single warning for array clipping if desired
            # print(f"Warning: Liq T input clipped outside range [{self.T_min:.1f}-{self.T_max:.1f}K]...")
            pass

        # --- Interpolate properties ---
        for prop_name, interpolator in self.interpolators.items():
            try:
                # interpolator handles array input directly
                props[prop_name] = interpolator(T_clipped)
            except ValueError:
                # Return NaNs matching input shape
                props[prop_name] = np.full_like(T_arr, np.nan)

        # --- Calculate enthalpy (handles array T) ---
        # Ensure specific_heat is array for calculation consistency
        cp_val = props.get('specific_heat', np.nan)
        cp_arr = np.asarray(cp_val) # Convert potential scalar from props to array

        if 'specific_heat' in props and not np.any(np.isnan(cp_arr)):
            h_ref_interp = self.interpolators.get('enthalpy_ref')
            if h_ref_interp:
                 props['enthalpy'] = h_ref_interp(T_clipped)
            else:
                # Approximate enthalpy integration (works for arrays too)
                T_ref = self.T_min
                # Ensure cp_ref is scalar for the calculation
                cp_ref_scalar = float(self.interpolators['specific_heat'](T_ref))
                H_ref_approx = cp_ref_scalar * T_ref
                # Use the array cp_arr here
                props['enthalpy'] = H_ref_approx + 0.5 * (cp_arr + cp_ref_scalar) * (T_clipped - T_ref)
        else:
            props['enthalpy'] = np.full_like(T_arr, np.nan)

        # --- Ensure scalar output if scalar input ---
        if is_scalar_input:
            for key, val in props.items():
                 if isinstance(val, np.ndarray) and val.size == 1:
                     props[key] = val.item() # Convert length-1 array to scalar

        return props


    def get_prop(self, prop_name, T):
        """Returns a specific liquid property at temperature T (K). T can be scalar or array."""
        # Ensure T is array for consistent processing
        T_arr = np.asarray(T)
        T_clipped = np.clip(T_arr, self.T_min, self.T_max)
        is_scalar_input = T_arr.ndim == 0

        # --- Clipping Warning (same logic as get_properties) ---
        was_clipped = False
        if is_scalar_input:
            if T != T_clipped: was_clipped = True
        else:
            if np.any(T != T_clipped): was_clipped = True

        if was_clipped and config.LOG_LEVEL >= 1:
            # print(f"Warning: Liq T input clipped...")
            pass

        if prop_name == 'enthalpy':
            # get_properties handles array/scalar T and returns appropriate type
            props = self.get_properties(T)
            # Return nan with correct shape if key missing
            return props.get('enthalpy', np.full_like(T_arr, np.nan))

        if prop_name in self.interpolators:
            try:
                interpolated_value = self.interpolators[prop_name](T_clipped)
                # Return array if input was array, scalar if input was scalar
                return interpolated_value.item() if is_scalar_input else interpolated_value
            except ValueError:
                 # Return nan with correct shape
                 return np.full_like(T_arr, np.nan).item() if is_scalar_input else np.full_like(T_arr, np.nan)
        else:
            valid_props = list(self.interpolators.keys())
            raise ValueError(f"Property '{prop_name}' not found in liquid data. Available: {valid_props}")

    def get_molar_volume(self, T):
        """
        指定された温度における液体のモル体積 [m^3/mol] を計算する。
        """
        # 液体密度 [kg/m^3] を取得
        rho_l = self.get_prop('density', T)
        
        if np.isnan(rho_l) or rho_l <= 0:
            print(f"Warning: Invalid liquid density ({rho_l}) at T={T:.1f}K. Cannot calculate molar volume.")
            return np.nan
            
        # モル体積 v_L = M_fuel / rho_l
        return self.fuel_molar_mass / rho_l

class GasProperties:
    """
    Wrapper around Cantera Gas object. Implements Redlich-Kwong EOS for density.
    Other properties are currently calculated using Cantera's models (typically ideal gas based).
    """
    def __init__(self, mech_file=config.MECH_FILE, use_rk=config.USE_RK_EOS):
        try:
            with warnings.catch_warnings():
                 warnings.filterwarnings("ignore", category=UserWarning, message="Found additional thermo entry for species")
                 self.gas = ct.Solution(mech_file)
            print(f"Successfully loaded Cantera Solution from: {mech_file}")
        except Exception as e:
             raise RuntimeError(f"Failed to initialize Cantera Solution from {mech_file}: {e}")

        self.nsp = self.gas.n_species
        self.use_rk = use_rk
        try:
            self.fuel_idx = self.gas.species_index(config.FUEL_SPECIES_NAME)
            print(f"Found fuel '{config.FUEL_SPECIES_NAME}' at index {self.fuel_idx}.")
        except ValueError:
             raise ValueError(f"Fuel species '{config.FUEL_SPECIES_NAME}' not found in mechanism '{mech_file}'")
        try:
            self.o2_idx = self.gas.species_index('o2')
            self.n2_idx = self.gas.species_index('n2')
        except ValueError:
             print(f"Warning: 'o2' or 'n2' not found in mechanism. Check species names.")
             self.o2_idx = -1
             self.n2_idx = -1

        self.species_names = self.gas.species_names
        self.molecular_weights = self.gas.molecular_weights # kg/kmol

        try:
             self.gas.TPX = config.T_INF_INIT, config.P_INIT, config.X_INF_INIT
             self.X_amb_init = self.gas.X.copy()
             self.X_amb_init = np.maximum(self.X_amb_init, 0.0)
             self.X_amb_init /= np.sum(self.X_amb_init)
             self.X_amb_non_fuel_sum_init = np.sum(self.X_amb_init) - (self.X_amb_init[self.fuel_idx] if self.fuel_idx >=0 else 0.0)
             if self.X_amb_non_fuel_sum_init < 1e-9:
                  self.X_amb_non_fuel_sum_init = 1.0
             print("Initial ambient mole fractions calculated and stored.")
        except Exception as e:
            print(f"CRITICAL ERROR calculating initial ambient mole fractions: {e}")
            self.X_amb_init = np.zeros(self.nsp)
            fallback_idx = self.n2_idx if self.n2_idx >= 0 else 0
            self.X_amb_init[fallback_idx] = 1.0
            self.X_amb_non_fuel_sum_init = 1.0
            print("Warning: Using default ambient mole fractions (pure N2 or first species).")

        if self.use_rk:
            print("Redlich-Kwong EOS enabled for density calculation.")
            self._init_rk_parameters()
        else:
             print("Using Cantera's default EOS for density calculation.")
             self.Tc = None; self.Pc = None; self.omega = None
        
        # ### <<< 追加: キャッシュ用変数の初期化 >>> ###
        self._cached_T = -1.0  # ありえない温度で初期化
        self._cached_P = -1.0  # ありえない圧力で初期化
        self._cached_Y = np.full(self.nsp, -1.0) # ありえない組成で初期化
        self._cached_X_str = "" # Xを文字列としてキャッシュする場合 (今回はYを直接比較)
        self._state_is_set_by_Y = False # 最後にYでセットされたかXでセットされたか
        # ### <<< 追加ここまで >>> ###

        # Store last valid state for fallback (これは従来通り)
        self.last_T = config.T_INF_INIT
        self.last_P = config.P_INIT
        self.last_Y = np.zeros(self.nsp)
        self.last_Y[self.n2_idx if self.n2_idx>=0 else 0] = 1.0

    def _init_rk_parameters(self):
        """Load critical properties from config."""
        self.Tc = np.zeros(self.nsp)
        self.Pc = np.zeros(self.nsp)
        self.omega = np.zeros(self.nsp)
        loaded_count = 0
        print("Loading RK parameters from config...")
        for i, name in enumerate(self.species_names):
            # Use case-insensitive matching? Config might use different case.
            name_lower = name.lower()
            rk_param_found = False
            for cfg_name, params in config.RK_PARAMS.items():
                if cfg_name.lower() == name_lower:
                    self.Tc[i] = params['Tc']
                    self.Pc[i] = params['Pc']
                    self.omega[i] = params['omega']
                    loaded_count += 1
                    rk_param_found = True
                    break # Found match for this species
            if not rk_param_found:
                # Set defaults (implies ideal gas for this species in RK calcs)
                self.Tc[i] = 0.0
                self.Pc[i] = 1.0e5 # Assign a dummy pressure > 0
                self.omega[i] = 0.0

        print(f"  Loaded RK parameters for {loaded_count}/{self.nsp} species from config.")
        if self.Tc[self.fuel_idx] <= 0.0:
             print(f"WARNING: RK parameters for Fuel '{config.FUEL_SPECIES_NAME}' missing/invalid in config! RK density inaccurate.")
        # Add checks for O2/N2 if needed, e.g.,
        # if self.o2_idx >= 0 and self.Tc[self.o2_idx] <= 0.0: print("Warning: RK parameters for O2 missing.")
        # if self.n2_idx >= 0 and self.Tc[self.n2_idx] <= 0.0: print("Warning: RK parameters for N2 missing.")


    def set_state(self, T, P, Y_or_X):
        """
        Safely set the thermodynamic state of the Cantera object using T, P, and
        either mass fractions (Y) or mole fractions (X).
        Optimized to avoid redundant Cantera calls if state is unchanged.
        Returns True if successful, False otherwise.
        """
        T_safe = np.clip(T, 200.0, 6000.0)
        P_safe = max(P, 100.0)

        # ### <<< 修正: キャッシュ確認と状態更新ロジック >>> ###
        is_Y_input = isinstance(Y_or_X, np.ndarray) and Y_or_X.ndim == 1 and Y_or_X.size == self.nsp

        # キャッシュと比較 (Yで設定する場合)
        if is_Y_input:
            if (abs(T_safe - self._cached_T) < 1e-9 and # 温度がほぼ同じ
                abs(P_safe - self._cached_P) < 1e-9 and # 圧力がほぼ同じ
                self._state_is_set_by_Y and             # 前回もYで設定され
                np.allclose(Y_or_X, self._cached_Y, rtol=1e-9, atol=1e-12)): # Yがほぼ同じ
                # print("DEBUG: set_state skipped (Y cache hit)") # デバッグ用
                # Canteraの内部状態が既に設定されていると仮定し、last_T/P/Yのみ更新してTrueを返す
                # (ただし、Canteraのself.gasオブジェクトの状態が外部から変更されていないという前提)
                # より安全には、キャッシュヒットでもself.gas.TPYを呼ぶが、それは最適化にならない。
                # ここでは、キャッシュが有効ならCanteraコールをスキップする。
                # fallback用の last_T, last_P, last_Y はCanteraが実際に更新されたときのみ更新する方針とする
                return True # 既にこの状態なので何もしない
        # (Xで設定する場合のキャッシュ比較は、X_or_Yがdictの場合の処理内で行うか、別途実装)
        # 今回は主にYで呼ばれるケースを最適化する

        try:
            if abs(T - T_safe) > 1e-1 and config.LOG_LEVEL >= 1:
                pass
            if abs(P - P_safe) > 1e-1 and config.LOG_LEVEL >= 1:
                pass

            if isinstance(Y_or_X, dict): # Mole fractions (X)
                # Xでのキャッシュ比較 (簡易版: 文字列比較)
                # current_X_str = str(sorted(Y_or_X.items())) # dictをソートして文字列化
                # if (abs(T_safe - self._cached_T) < 1e-9 and
                #     abs(P_safe - self._cached_P) < 1e-9 and
                #     not self._state_is_set_by_Y and
                #     current_X_str == self._cached_X_str):
                #     # print("DEBUG: set_state skipped (X cache hit)")
                #     return True
                
                self.gas.TPX = T_safe, P_safe, Y_or_X
                # キャッシュ更新
                self._cached_T = T_safe
                self._cached_P = P_safe
                self._cached_Y = self.gas.Y.copy() # Xで設定してもYをキャッシュ
                # self._cached_X_str = current_X_str
                self._state_is_set_by_Y = False

            elif is_Y_input: # Mass fractions (Y)
                Y_clean = np.maximum(Y_or_X, 0)
                sum_Y = np.sum(Y_clean)
                if abs(sum_Y - 1.0) > 1e-4:
                    if sum_Y > 1e-6:
                        Y_clean /= sum_Y
                    else:
                        if config.LOG_LEVEL >= 1: print(f"Warning: Mass fractions sum to {sum_Y:.1e} at T={T:.1f}, P={P:.1e}. Resetting to N2.")
                        Y_clean = np.zeros_like(Y_or_X)
                        idx_fill = self.n2_idx if self.n2_idx >= 0 else (0 if self.nsp > 0 else -1)
                        if idx_fill != -1: Y_clean[idx_fill] = 1.0
                
                self.gas.TPY = T_safe, P_safe, Y_clean
                # キャッシュ更新
                self._cached_T = T_safe
                self._cached_P = P_safe
                self._cached_Y = Y_clean.copy() # 設定に使ったY_cleanを保存
                self._state_is_set_by_Y = True
            else:
                if config.LOG_LEVEL >= 0: print(f"Error: Invalid composition format provided to set_state.")
                return False

            # Store last successfully set state for fallback
            self.last_T = self.gas.T
            self.last_P = self.gas.P
            self.last_Y = self.gas.Y.copy()
            return True
        
        except (ct.CanteraError, ValueError) as e:
            if config.LOG_LEVEL >= 1:
                 print(f"Warning: Cantera Error setting state T={T:.1f}K, P={P:.1e}Pa, Y_sum={np.sum(Y_or_X) if is_Y_input else 'X_dict'}. Err: {e}")
                 # エラー時はキャッシュを無効化（次の呼び出しで必ず再設定するように）
                 self._cached_T = -1.0
                 self._cached_P = -1.0
                 self._cached_Y.fill(-1.0)
            return False
        # ### <<< 修正ここまで >>> ###

    def _rk_eos_residual(self, Z, A_mix, B_mix):
        """Residual form of the RK cubic equation for the solver Z^3 - Z^2 + (A-B-B^2)Z - AB = 0."""
        return Z**3 - Z**2 + (A_mix - B_mix - B_mix**2) * Z - A_mix * B_mix

    def _solve_rk_eos(self, T, P, X):
        """Solves the Redlich-Kwong EOS for compressibility factor Z using Newton's method."""
        R_univ = config.R_UNIVERSAL # J/(mol K)

        # 1. Calculate pure component parameters a_i, b_i
        valid_rk_mask = (self.Tc > 0) & (self.Pc > 0)
        Tci = np.where(valid_rk_mask, self.Tc, 1.0)
        Pci = np.where(valid_rk_mask, self.Pc, 1.0e5)
        a_i = np.where(valid_rk_mask, 0.42748 * (R_univ**2 * Tci**2.5) / Pci, 0.0) # Includes T^-0.5
        b_i = np.where(valid_rk_mask, 0.08664 * (R_univ * Tci) / Pci, 0.0)

        # 2. Apply mixing rules (Van der Waals one-fluid using geometric mean for a)
        b_mix = np.sum(X * b_i)
        a_mix_sqrt = 0.0
        # Use sqrt(a_i/sqrt(T)) in mixing rule to get a_mix / sqrt(T)
        a_i_T05 = np.sqrt(a_i / max(T, 1e-2)) # sqrt(a_i(T)) ~ a_i_const / T^0.25
        for i in range(self.nsp):
             for j in range(self.nsp):
                 a_mix_sqrt += X[i] * X[j] * np.sqrt(a_i_T05[i] * a_i_T05[j])
        # a_mix calculation: a_mix represents the 'a' parameter *at temperature T*
        # The 'a' in the dimensionless A_mix = aP/(RT)^2 should be temperature dependent a_i(T)
        # a_i(T) = 0.42748 * R^2 * Tc^2.5 / (Pc * sqrt(T))
        # a_mix(T) = sum(xi*xj*sqrt(ai(T)*aj(T)))
        a_i_T = np.where(valid_rk_mask, 0.42748 * (R_univ**2 * Tci**2.5) / (Pci * np.sqrt(max(T,1e-2))), 0.0)
        a_mix = np.sum( X[i] * X[j] * np.sqrt(a_i_T[i]*a_i_T[j]) for i in range(self.nsp) for j in range(self.nsp) )


        # 3. Dimensionless parameters for the cubic equation
        T_safe = max(T, 1e-2)
        A_mix = a_mix * P / (R_univ**2 * T_safe**2) # a_mix is a(T)
        B_mix = b_mix * P / (R_univ * T_safe)

        # 4. Solve the cubic equation for Z
        Z = 1.0 # Default to ideal gas
        try:
            Z_solution = newton(self._rk_eos_residual, x0=1.0, args=(A_mix, B_mix),
                                tol=config.RK_SOLVER_TOL, maxiter=config.MAX_ITER_RK)
            # Check if the solution is physical (Z > B_mix for gas)
            if Z_solution > B_mix:
                 Z = Z_solution
            else: # Try numpy roots as fallback
                 coeffs = [1.0, -1.0, A_mix - B_mix - B_mix**2, -A_mix * B_mix]
                 roots = np.roots(coeffs)
                 real_roots = roots[np.isreal(roots)].real
                 physical_roots = real_roots[real_roots > B_mix]
                 if len(physical_roots) > 0:
                     Z = np.max(physical_roots) # Vapor root (largest real root > B)
                     if config.LOG_LEVEL >= 2: print(f"      DEBUG: RK Newton failed, used numpy root Z={Z:.4f}")
                 else:
                     if config.LOG_LEVEL >= 1: print(f"Warning: No physical RK root found (Z>{B_mix:.3f}) at T={T:.1f}, P={P:.1e}. Using Z=1.")
                     Z = 1.0 # Fallback

        except (RuntimeError, OverflowError): # Solver failed to converge or overflow
            if config.LOG_LEVEL >= 1: print(f"Warning: RK EOS solver failed at T={T:.1f}, P={P:.1e}. Using Z=1.")
            Z = 1.0 # Fallback to ideal gas

        # Final safety check
        if np.isnan(Z) or Z <= 0:
             if config.LOG_LEVEL >= 1: print(f"Warning: RK Z is NaN or non-positive ({Z}). Using Z=1.")
             Z = 1.0

        return Z

    def get_density(self, T, P, Y):
        if not self.set_state(T, P, Y): # set_state内でキャッシュ確認が行われる
            # Fallback to last known valid state for density calculation attempt
            if hasattr(self, 'last_T') and self.last_T > 0 : # last_Tが有効な値か確認
                if config.LOG_LEVEL >=1: print(f"Info: Using last valid state for density T={self.last_T:.1f} P={self.last_P:.1e}")
                # last_T, P, Y を使って再度set_stateを試みる (無限ループにならないように注意)
                # ここでは、既にset_state内で fallback用の last_T/P/Y が gas オブジェクトに設定されているはず
                # なので、ここでは gas オブジェクトの現在の状態を使う
            else:
                 if config.LOG_LEVEL >=0: print("Error: Cannot set state in get_density, no valid fallback available in gas object.")
                 return np.nan
        
        # self.gas は set_state (成功時またはfallback時) によって状態が設定されているはず
        T_use, P_use = self.gas.T, self.gas.P

        # Proceed with density calculation
        if self.use_rk and np.any(self.Tc > 0):
            try:
                X = self.gas.X # Get mole fractions from the set state
                # Ensure mole fractions are valid
                if np.any(X < -1e-9):
                     X = np.maximum(X, 0.0); X /= np.sum(X)

                Z = self._solve_rk_eos(T_use, P_use, X) # Use potentially clipped T/P
                mean_mw_kmol = self.gas.mean_molecular_weight # kg/kmol
                R_kmol = config.R_UNIVERSAL * 1000.0 # J/(kmol*K)
                density_rk = P_use * mean_mw_kmol / (Z * R_kmol * T_use)

                if np.isnan(density_rk) or density_rk <= 1e-6:
                    if config.LOG_LEVEL >= 1: print(f"Warning: RK density calc resulted in {density_rk:.2e} at T={T_use:.1f}. Using Cantera default.")
                    return self.gas.density # Fallback to Cantera default
                return density_rk
            except Exception as e:
                print(f"Error during RK density calculation at T={T_use:.1f}, P={P_use:.1e}: {e}. Falling back to Cantera default.")
                return self.gas.density # Fallback
        else:
            # Use Cantera's default density (ideal gas or other loaded EOS)
            return self.gas.density

    # --- Other Properties (get_cp_mass, get_enthalpy_mass, etc.) ---
    # これらのメソッドは、最初に self.set_state(T, P, Y) を呼び出す構造はそのままです。
    # set_state 内部のキャッシュ機構により、不要なCanteraの更新が抑制されることを期待します。

    def get_cp_mass(self, T, P, Y):
        if not self.set_state(T, P, Y): return np.nan
        return self.gas.cp_mass

    def get_enthalpy_mass(self, T, P, Y):
         if not self.set_state(T, P, Y): return np.nan
         return self.gas.enthalpy_mass

    # ... (他の get_... メソッドも同様に、冒頭の set_state 呼び出しはそのまま) ...
    def get_int_energy_mass(self, T, P, Y):
         if not self.set_state(T, P, Y): return np.nan
         return self.gas.int_energy_mass # J/kg

    def get_cv_mass(self, T, P, Y):
         if not self.set_state(T, P, Y): return np.nan
         return self.gas.cv_mass # J/kg/K

    def get_partial_enthalpies_mass(self, T, P, Y):
         if not self.set_state(T, P, Y): return np.full(self.nsp, np.nan)
         Hi_molar = self.gas.partial_molar_enthalpies # J/kmol
         Mi_kg_kmol = self.molecular_weights # kg/kmol
         Mi_safe = np.maximum(Mi_kg_kmol, 1e-6) # Avoid division by zero
         return Hi_molar / Mi_safe # J/kg

    def get_thermal_conductivity(self, T, P, Y):
         if not self.set_state(T, P, Y): return np.nan
         lambda_val = self.gas.thermal_conductivity # W/m/K
         if np.isnan(lambda_val) or lambda_val < 0:
              if config.LOG_LEVEL >= 1: print(f"Warning: Invalid thermal conductivity ({lambda_val:.2e}) from Cantera at T={T:.1f}. Returning small positive.")
              return 1e-4 # Return small positive value
         return lambda_val

    def get_viscosity(self, T, P, Y):
         if not self.set_state(T, P, Y): return np.nan
         visc = self.gas.viscosity # Pa*s
         if np.isnan(visc) or visc < 0:
              if config.LOG_LEVEL >= 1: print(f"Warning: Invalid viscosity ({visc:.2e}) from Cantera at T={T:.1f}. Returning small positive.")
              return 1e-6 # Return small positive value
         return visc

    def get_diffusion_coeffs(self, T, P, Y, option=config.DIFFUSION_OPTION):
         if not self.set_state(T, P, Y): return np.full(self.nsp, np.nan)

         if option == 'constant':
             return np.full(self.nsp, config.DIFFUSION_CONSTANT_VALUE)

         rho = self.gas.density
         cp_mass = self.gas.cp_mass
         lambda_val = self.gas.thermal_conductivity

         if np.isnan(rho) or np.isnan(cp_mass) or np.isnan(lambda_val) or rho <=0 or cp_mass <=0 or lambda_val <=0:
             print(f"Warning: NaN/Invalid properties in get_diffusion_coeffs (T={T:.1f}, rho={rho:.2e}, cp={cp_mass:.2e}, lam={lambda_val:.2e}).")
             return np.full(self.nsp, 1e-9)

         if option == 'Le=1':
             D_eff = lambda_val / (rho * cp_mass)
             return np.full(self.nsp, max(D_eff, 1e-12))
         elif option == 'mixture_averaged':
             try:
                 Dk_mix_mole = self.gas.mix_diff_coeffs
                 if np.any(np.isnan(Dk_mix_mole)) or np.any(Dk_mix_mole < 0):
                      if config.LOG_LEVEL >= 1: print(f"Warning: Invalid mixture diff coeffs from Cantera at T={T:.1f}. Using Le=1 fallback.")
                      D_eff = lambda_val / (rho * cp_mass)
                      return np.full(self.nsp, max(D_eff, 1e-12))
                 return np.maximum(Dk_mix_mole, 1e-12)
             except Exception as e:
                 print(f"Warning: Cantera failed to get mixture_averaged diff coeffs: {e}. Using Le=1 fallback.")
                 D_eff = lambda_val / (rho * cp_mass)
                 return np.full(self.nsp, max(D_eff, 1e-12))
         else:
             raise ValueError(f"Unknown diffusion option: {option}")
    
    def get_effective_transport_properties(self, T, P, Y, k, epsilon, R, r_g_centers, delta_nu, option=config.TURBULENCE_MODEL):
        """
        分子輸送物性値と乱流輸送物性値を計算し、実効物性値を返す。
        """
        # 1. Canteraから分子輸送物性値を取得
        if not self.set_state(T, P, Y):
            return np.nan, np.nan, np.full(self.nsp, np.nan) # エラー時はNaNを返す
        
        lambda_mol = self.gas.thermal_conductivity
        rho = self.gas.density
        cp = self.gas.cp_mass
        Dk_mol = self.get_diffusion_coeffs(T, P, Y) # 既存の拡散係数取得メソッドを流用

        # 2. 乱流輸送物性値の計算
        if option == 'CONSTANT' or option == 'FILE':
            nu_t_inf = 0.0
            
            if k > 1e-9 and epsilon > 1e-9 and rho > 1e-9:
                nu_t_inf = config.CMU * k**2 / epsilon
            
            #    各計算格子点での減衰関数と乱流物性値を計算 (ベクトル化)
            #    r_g_centers は配列なので、この計算は一度に全ての格子点に対して行われる
            y = r_g_centers - R
            
            # delta_nu が非常に小さい（流れが速い）場合に exp の引数が大きくなりすぎるのを防ぐ
            delta_nu_safe = max(delta_nu, 1e-9)
            damping_func = 1.0 - np.exp(-y * config.C_MUT_DAMP_FACTOR / delta_nu_safe)
            nu_t_local = nu_t_inf * damping_func

            lambda_t = rho * nu_t_local * cp / config.TURBULENT_PRANDTL_NUMBER
            D_t = nu_t_local / config.TURBULENT_SCHMIDT_NUMBER
            mu_t = rho * nu_t_local

        elif option == 'NONE':
            lambda_t = 0.0
            D_t = 0.0
            mu_t = 0.0
        else:
            raise ValueError(f"Unknown TURBULENCE_MODEL option: {option}")
        
        # 3. 実効物性値 = 分子物性値 + 乱流物性値
        lambda_eff = lambda_mol + lambda_t
        Dk_eff = Dk_mol + D_t

        # 乱流粘性も返す（界面でのNu, Sh数計算で利用）
        mu_mol = self.gas.viscosity
        mu_eff = mu_mol + mu_t

        return lambda_eff, mu_eff, Dk_eff

class FugacityInterpolator:
    """
    フガシティー係数のCSVマップを読み込み、多次元補間を行うクラス。
    """
    def __init__(self, filename='fugacity_map_Fluid_n-heptane_SRK.csv'):
        try:
            df = pd.read_csv(filename)
            print(f"Fugacity map '{filename}' loaded successfully.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Fugacity map file '{filename}' not found.")

        # 補間で利用する入力変数とターゲット変数
        self.input_cols = ['Temperature (K)', 'Pressure (Pa)', 'n-heptane_mole_fraction']
        self.target_col = 'n-heptane_fugacity_coeff_Fluid'

        # グリッドデータの作成
        self.points = [df[col].unique() for col in self.input_cols]
        
        # データがグリッド形式であることを確認
        for i, col in enumerate(self.input_cols):
            if not np.all(np.isclose(np.diff(self.points[i]), np.diff(self.points[i])[0])):
                # データが等間隔でない場合、ソートして使う
                self.points[i] = np.sort(self.points[i])

        # グリッドデータに整形
        mesh_T, mesh_P, mesh_X = np.meshgrid(*self.points, indexing='ij')
        
        # dfをピボットしてグリッド形式のデータを作成
        df_pivot = df.pivot_table(index=['Temperature (K)', 'Pressure (Pa)'], 
                                  columns='n-heptane_mole_fraction', 
                                  values=self.target_col)
        
        # グリッドデータにNaNが含まれている場合は補間する
        df_pivot.interpolate(axis=1, method='linear', inplace=True)
        df_pivot.fillna(method='ffill', inplace=True)
        df_pivot.fillna(method='bfill', inplace=True)
        
        values = df_pivot.values.reshape(len(self.points[0]), len(self.points[1]), len(self.points[2]))

        # 補間器の作成
        self.interpolator = RegularGridInterpolator(self.points, values,
                                                    method="linear",
                                                    bounds_error=False,
                                                    fill_value=None) # 範囲外は最近傍点で埋める
        print("Fugacity coefficient interpolator has been created.")

    def get_phi(self, T, P, X_fuel):
        """
        与えられた T, P, 燃料モル分率 からフガシティー係数を返す。
        """
        point = np.array([T, P, X_fuel])
        # 補間器の範囲内にクリップして安全に評価
        clipped_point = np.array([
            np.clip(point[0], self.points[0].min(), self.points[0].max()),
            np.clip(point[1], self.points[1].min(), self.points[1].max()),
            np.clip(point[2], self.points[2].min(), self.points[2].max())
        ])
        
        phi = self.interpolator(clipped_point)[0]
        return phi