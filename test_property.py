import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ユーザー提供のシミュレーションモジュールをインポート
import config
from properties import LiquidProperties

def verify_liquid_properties_english():
    """
    Compares n-heptane liquid properties from three sources. (English Version)
    1. User-provided CSV file
    2. NIST reference data (key points)
    """
    print("--- Starting Liquid Property Verification Program (English Version) ---")

    # --- 1. Load data from user's CSV ---
    try:
        user_props = LiquidProperties(config.LIQUID_PROP_FILE)
        print(f"Loaded user property file: '{config.LIQUID_PROP_FILE}'")
    except Exception as e:
        print(f"ERROR: Failed to load {config.LIQUID_PROP_FILE}: {e}")
        return

    # --- 2. Define NIST reference data ---
    # Source: NIST Chemistry WebBook (https://webbook.nist.gov/chemistry/)
    # n-Heptane Liquid properties
    nist_data = {
        'density': {
            'T': [298.15, 323.15, 373.15, 423.15, 473.15], # K
            'val': [679.5, 658.3, 617.7, 565.0, 505.8]  # kg/m^3
        },
        'specific_heat': {
            'T': [298.15, 323.15, 373.15, 423.15, 473.15], # K
            'val': [2248, 2351, 2580, 2873, 3262]  # J/kg-K
        },
        'thermal_conductivity': {
            'T': [300, 350, 400, 450, 500], # K
            'val': [0.138, 0.124, 0.110, 0.097, 0.085] # W/m-K
        },
        'vapor_pressure': {
            'T': [300, 350, 400, 450, 500], # K
            'val': [6111, 46115, 203800, 683400, 1845000] # Pa
        }
    }
    print("Loaded NIST reference data points.")


    # --- 3. Compare and Plot ---
    temp_range = np.linspace(280, 530, 100) # Temperature range for comparison

    # Define properties to plot with English labels
    properties_to_plot = {
        'density': 'Density (kg/m³)',
        'specific_heat': 'Specific Heat (J/kg-K)',
        'thermal_conductivity': 'Thermal Conductivity (W/m-K)',
        'vapor_pressure': 'Vapor Pressure (Pa)'
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('n-Heptane Liquid Property Verification (vs. NIST)', fontsize=18)

    print("\n--- Comparison of values at 300 K ---")
    print(f"{'Property':<30} | {'User CSV':<15} | {'NIST Ref.':<15}")
    print("-"*65)

    for i, (prop_key, prop_label) in enumerate(properties_to_plot.items()):
        ax = axes[i]
        
        # User CSV values
        user_values = user_props.get_prop(prop_key, temp_range)
        ax.plot(temp_range, user_values, 'b-', label='User CSV Data', linewidth=2)
        user_val_300k = user_props.get_prop(prop_key, 300.0)

        # NIST reference values
        if prop_key in nist_data:
            nist_T = nist_data[prop_key]['T']
            nist_val = nist_data[prop_key]['val']
            ax.plot(nist_T, nist_val, 'ro', label='NIST Reference', markersize=8, fillstyle='none')
        
        # Print comparison result at 300 K
        nist_val_300k = nist_data.get(prop_key, {}).get('val', ["N/A"])[0]
        print(f"{prop_label:<30} | {user_val_300k:<15.3f} | {nist_val_300k:<15.3f}")

        ax.set_title(prop_label)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel(prop_label)
        ax.grid(True, which='both', linestyle=':')
        ax.legend()
        if prop_key == 'vapor_pressure':
            ax.set_yscale('log') # Vapor pressure is better viewed on a log scale

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = os.path.join(config.OUTPUT_DIR, 'verification_liquid_properties_english.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"\nProperty comparison plot saved to: {plot_filename}")


if __name__ == '__main__':
    # Create output directory if it doesn't exist
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        
    verify_liquid_properties_english()