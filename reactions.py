# =========================================
#        reactions.py (r4)
# =========================================
# reactions.py
"""Functions for calculating chemical reaction rates with cutoff option."""

import numpy as np
import cantera as ct
import config # Import config to access cutoff settings
from properties import GasProperties # For type hinting

def calculate_reaction_rates(gas_props: GasProperties, T_g: np.ndarray, P: float, Y_g: np.ndarray):
    """
    Calculates reaction rates [kg/m^3/s] based on REACTION_TYPE in config.
    """
    if config.REACTION_TYPE == 'detailed':
        return calculate_cantera_rates(gas_props, T_g, P, Y_g)
    elif config.REACTION_TYPE == 'overall':
        # Need density for overall rate calculation
        rho_g = np.zeros_like(T_g)
        for i in range(len(T_g)):
             rho_g[i] = gas_props.get_density(T_g[i], P, Y_g[:, i])
        return calculate_overall_rates(rho_g, T_g, Y_g, gas_props, P)
    elif config.REACTION_TYPE == 'none':
        return np.zeros_like(Y_g) # Return zeros if reactions are off
    else:
        raise ValueError(f"Unknown REACTION_TYPE: {config.REACTION_TYPE}")


def calculate_cantera_rates(gas_props: GasProperties, T_g: np.ndarray, P: float, Y_g: np.ndarray):
    """
    Calculates reaction rates using Cantera for detailed kinetics.
    Includes optional cutoff based on temperature and fuel mole fraction.
    Returns wdot [kg/m^3/s].
    """
    Ng = T_g.shape[0]
    nsp = gas_props.nsp
    ###nsp = gas_props.n_species
    wdot = np.zeros((nsp, Ng))
    fuel_idx = gas_props.fuel_idx
    ###fuel_idx = gas_props.species_index(config.FUEL_SPECIES_NAME) 

    # Temporary array for mole fractions if needed for cutoff
    X_g_i = np.zeros(nsp)

    points_reacted = 0 # Counter for debug
    for i in range(Ng):
        calculate_reactions = True
        fuel_mole_frac = -1.0 # Initialize

        # --- Reaction Cutoff Check ---
        if config.ENABLE_REACTION_CUTOFF:
            # Check basic conditions first
            if T_g[i] < config.REACTION_CALC_MIN_TEMP or np.isnan(T_g[i]) or np.isnan(P):
                 calculate_reactions = False
            else:
                 # Set state safely to get mole fractions for cutoff check
                 if gas_props.set_state(T_g[i], P, Y_g[:, i]):
                     X_g_i = gas_props.gas.X
                     fuel_mole_frac = X_g_i[fuel_idx] if fuel_idx >= 0 else -1.0
                 else:
                      # If state cannot be set, assume no reaction for safety
                      fuel_mole_frac = -1.0
                      if config.LOG_LEVEL >= 1: print(f"Warning: Could not set state for reaction cutoff check at node {i}")

                 if fuel_mole_frac < config.REACTION_CALC_MIN_FUEL_MOL_FRAC:
                     calculate_reactions = False
        # --- End Cutoff Check ---

        if calculate_reactions:
            points_reacted += 1
            # Set state again (might be redundant if set_state was called above)
            if gas_props.set_state(T_g[i], P, Y_g[:, i]):
                try:
                    # Get net production rates [kmol/m^3/s] and convert to mass rates [kg/m^3/s]
                    wdot[:, i] = gas_props.gas.net_production_rates * gas_props.molecular_weights
                except (ct.CanteraError, ValueError) as e:
                     if config.LOG_LEVEL >= 1: print(f"Warning: Cantera failed rate calc at point {i}. T={T_g[i]:.1f}, P={P:.1e}. Err: {e}")
                     wdot[:, i] = 0.0
            else:
                # If state setting failed here (should be rare if it passed above), set rates to zero
                wdot[:, i] = 0.0
        else:
            # Set rates to zero if cutoff conditions met
            wdot[:, i] = 0.0

    if config.LOG_LEVEL >= 2 and Ng > 0:
        print(f"      DEBUG Reactions: Calculated detailed rates for {points_reacted}/{Ng} points.", end='\r', flush=True)

    return wdot


def calculate_overall_rates(rho_g: np.ndarray, T_g: np.ndarray, Y_g: np.ndarray, gas_props: GasProperties, P: float):
    """
    Calculates reaction rates using the specified overall Arrhenius model.
    Includes optional cutoff based on temperature and fuel mole fraction.
    NOTE: Assumes n-heptane combustion C7H16 + 11 O2 -> 7 CO2 + 8 H2O.
    Returns wdot [kg/m^3/s].
    """
    Ng = T_g.shape[0]
    nsp = gas_props.nsp
    wdot_mass = np.zeros((nsp, Ng))

    try:
        fuel_idx = gas_props.fuel_idx
        ox_idx = gas_props.o2_idx
        co2_idx = gas_props.gas.species_index('CO2')
        h2o_idx = gas_props.gas.species_index('H2O')
        if fuel_idx < 0 or ox_idx < 0 or co2_idx < 0 or h2o_idx < 0:
             raise ValueError("Required species (fuel, O2, CO2, H2O) not found for overall reaction.")

        products = {co2_idx: 7.0, h2o_idx: 8.0} # Map index to stoichiometric coeff
        nu_fuel = -1.0; nu_ox = -11.0 # Stoichiometry for n-heptane

    except ValueError as e:
        print(f"CRITICAL ERROR finding species indices for overall reaction: {e}")
        return wdot_mass # Return zeros

    Mf = gas_props.molecular_weights[fuel_idx]
    Mo = gas_props.molecular_weights[ox_idx]
    Mp = gas_props.molecular_weights
    points_reacted = 0

    for i in range(Ng):
        calculate_reactions = True
        fuel_mole_frac = -1.0
        # --- Reaction Cutoff Check ---
        if config.ENABLE_REACTION_CUTOFF:
             if T_g[i] < config.REACTION_CALC_MIN_TEMP or np.isnan(T_g[i]) or np.isnan(P):
                 calculate_reactions = False
             else:
                 if gas_props.set_state(T_g[i], P, Y_g[:, i]): # Check if state is valid
                     X_g_i = gas_props.gas.X
                     fuel_mole_frac = X_g_i[fuel_idx] if fuel_idx >= 0 else -1.0
                 else:
                     fuel_mole_frac = -1.0

                 if fuel_mole_frac < config.REACTION_CALC_MIN_FUEL_MOL_FRAC:
                     calculate_reactions = False
        # --- End Cutoff Check ---

        if calculate_reactions:
            Yf_i = Y_g[fuel_idx, i]
            Yo_i = Y_g[ox_idx, i]
            T_i = T_g[i]
            rho_i = rho_g[i]
            T_threshold = 298.0; Y_threshold = 1e-10 # Rate calc thresholds

            if T_i > T_threshold and Yf_i > Y_threshold and Yo_i > Y_threshold and rho_i > 1e-6:
                points_reacted += 1
                try:
                    # Molar concentrations [kmol/m^3]
                    Cf_i = rho_i * Yf_i / Mf
                    Co_i = rho_i * Yo_i / Mo
                    # Rate expression omega_mol = B * [Fuel] * [Ox] * exp(-Ea/RT) [kmol/m^3/s]
                    # Note: Config B_SI is in m^3/(mol*s), need to convert to m^3/(kmol*s) or adjust concentration units
                    # Let's use config B_SI [m^3/mol/s] and concentrations in mol/m^3
                    Cf_mol_m3 = Cf_i * 1000.0 # mol/m^3
                    Co_mol_m3 = Co_i * 1000.0 # mol/m^3

                    exp_term = np.exp(-config.OVERALL_E_SI / (config.R_UNIVERSAL * T_i))
                    omega_mol_fuel_m3_s = (config.OVERALL_B_SI * Cf_mol_m3 * Co_mol_m3 * exp_term) # mol/m^3/s

                    # Convert to mass rates [kg/m^3/s] = omega_mol * MW_kg_mol
                    wdot_mass[fuel_idx, i] = omega_mol_fuel_m3_s * nu_fuel * (Mf / 1000.0) # kg/m^3/s
                    wdot_mass[ox_idx, i]   = omega_mol_fuel_m3_s * nu_ox   * (Mo / 1000.0) # kg/m^3/s
                    for idx, nu_prod in products.items():
                        wdot_mass[idx, i] = omega_mol_fuel_m3_s * nu_prod * (Mp[idx] / 1000.0) # kg/m^3/s

                except OverflowError:
                    # print(f"Warning: Overflow calculating overall rate at T={T_i:.1f}K. Setting rate to 0.")
                    wdot_mass[:, i] = 0.0 # Set rate to 0 if exp overflows
            else:
                wdot_mass[:, i] = 0.0 # No reaction if below thresholds
        else:
             wdot_mass[:, i] = 0.0 # No reaction if cutoff

    if config.LOG_LEVEL >= 2 and Ng > 0:
        print(f"      DEBUG Overall Reactions: Calculated for {points_reacted}/{Ng} points.", end='\r', flush=True)

    return wdot_mass