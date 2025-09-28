# =========================================
#            grid.py (r4)
# =========================================
# grid.py
"""Functions for generating liquid and gas phase grids for FVM."""

import numpy as np
import config

def liquid_grid_fvm(R, Nl):
    """
    Generates the liquid phase grid for Finite Volume Method (FVM).
    Returns cell center locations (r_centers), face locations (r_nodes),
    and cell volumes (volumes).
    Cell 0 is at the center, Cell Nl-1 is adjacent to the surface.
    Faces: Nl+1 locations (node 0 at r=0, node Nl at r=R)
    Centers: Nl locations
    Volumes: Nl volumes
    Grid is uniform in radius.
    """
    if Nl <= 0:
        return np.array([]), np.array([]), np.array([])
    if Nl == 1: # Single cell covering the whole sphere
        r_nodes = np.array([0.0, R]) # Face locations
        r_centers = np.array([0.5 * R]) # Cell center (approx)
        volumes = np.array([(4.0/3.0) * np.pi * R**3])
        return r_centers, r_nodes, volumes

    # Generate Nl+1 face locations uniformly from 0 to R
    r_nodes = np.linspace(0, R, Nl + 1)

    # Calculate cell center locations (arithmetic mean between faces)
    r_centers = 0.5 * (r_nodes[:-1] + r_nodes[1:])

    # Calculate cell volumes (volume of spherical shell)
    # Ensure r_nodes are strictly increasing for volume calculation
    r_nodes_safe = np.maximum.accumulate(r_nodes)
    volumes = (4.0 / 3.0) * np.pi * (r_nodes_safe[1:]**3 - r_nodes_safe[:-1]**3)
    volumes = np.maximum(volumes, 1e-30) # Ensure non-negative volumes

    # FVM cell center for the first cell (j=0) is between face 0 and face 1
    # The 'center' node is conceptually at r=0, but the first *cell* is defined by faces 0 and 1.

    return r_centers, r_nodes, volumes

def gas_grid_fvm(R, rmax, Ng):
    """
    Generates the gas phase grid for FVM using a geometric progression ('geometric_series')
    or a power law ('power_law').
    Cell 0 is adjacent to the droplet surface (r=R), Cell Ng-1 is adjacent to rmax.
    Returns cell center locations (r_centers), face locations (r_nodes),
    and cell volumes (volumes).
    Faces: Ng+1 locations (node 0 at r=R, node Ng at r=rmax)
    Centers: Ng locations
    Volumes: Ng volumes
    """
    if Ng <= 0:
        return np.array([]), np.array([]), np.array([])
    if Ng == 1: # Single cell covering the whole gas domain
        r_nodes = np.array([R, rmax]) # Face locations
        r_centers = np.array([0.5 * (R + rmax)]) # Cell center (approx)
        volumes = np.array([(4.0/3.0) * np.pi * (rmax**3 - R**3)])
        return r_centers, r_nodes, volumes

    # Generate Ng+1 face locations
    r_nodes = np.zeros(Ng + 1)

    # Prevent numerical issues if R is extremely small or rmax=R
    R_safe = max(R, 1e-15)
    rmax_safe = max(rmax, R_safe + 1e-15)
    ratio = rmax_safe / R_safe

    if config.GRID_TYPE == 'geometric_series':
        if ratio <= 1.0:
            # Handle edge case where R is very close to rmax
             g_nodes_norm = np.linspace(0, 1, Ng + 1)
             r_nodes = R + (rmax - R) * g_nodes_norm # Fallback to linear
        else:
             growth_factor = ratio**(1.0 / Ng)
             r_nodes = R_safe * growth_factor**np.arange(Ng + 1)

    elif config.GRID_TYPE == 'power_law':
        xi = config.XI_GAS_GRID
        g_nodes_norm = np.linspace(0, 1, Ng + 1) # Normalized coordinate [0, 1]
        r_nodes = R_safe + (rmax_safe - R_safe) * g_nodes_norm**xi
    else:
        raise ValueError(f"Unknown GRID_TYPE: {config.GRID_TYPE}")

    # Ensure boundaries are exact
    r_nodes[0] = R
    r_nodes[-1] = rmax

    # Calculate cell center locations (Arithmetic mean)
    r_centers = 0.5 * (r_nodes[:-1] + r_nodes[1:])

    # Calculate cell volumes (volume of spherical shell)
    # Ensure r_nodes are strictly increasing
    r_nodes_safe = np.maximum.accumulate(r_nodes)
    volumes = (4.0 / 3.0) * np.pi * (r_nodes_safe[1:]**3 - r_nodes_safe[:-1]**3)
    volumes = np.maximum(volumes, 1e-30) # Ensure non-negative volumes

    return r_centers, r_nodes, volumes

# Helper function to get face areas (needed for flux calculations)
def face_areas(r_nodes):
    """Calculates the area of the faces defined by r_nodes."""
    if len(r_nodes) < 1:
        return np.array([])
    return 4.0 * np.pi * r_nodes**2