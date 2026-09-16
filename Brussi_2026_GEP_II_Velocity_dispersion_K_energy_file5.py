# =============================================================================
# --- START OF FILE Brussi_2026_GEP_II_Velocity_dispersion_K_energy_file5 ---
#
# Copyright (C) 2026 Andrea Brussi - https://andreabrussi.it
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://gnu.org>.
#
# =============================================================================


import numpy as np
from scipy.special import gamma # Import required for the exact Euler gamma function

def sersic_3d_density(m_kpc, I_e, R_e_kpc, n_sersic):
    """
    Computes the deprojected 3D density profile for a generic Sersic index 'n'
    using the exact Prugniel & Simien (1997) analytical normalization.
    """
    if m_kpc <= 0:
        return 0.0

    # Structural coefficients tied to Sersic index n
    b_n = 2.0 * n_sersic - 1.0/3.0 + 0.00913 / n_sersic
    p_n = 1.0 - 0.6097 / n_sersic + 0.05463 / (n_sersic**2)

    nu = m_kpc / R_e_kpc

    # Prugniel-Simien 3D deprojection profile shape
    density_shape = (nu ** (-p_n)) * np.exp(-b_n * (nu ** (1.0 / n_sersic)))

    # UNIT CONVERSION: Convert R_e from kpc to pc to guarantee rho_0 is in L_sun / pc^3
    R_e_pc = R_e_kpc * 1000.0

    # Exact Prugniel-Simien 3D central normalization factor (rho_0)
    numerator = b_n ** (n_sersic * (3.0 - p_n))
    denominator = 2.0 * np.pi * R_e_pc * n_sersic * gamma(n_sersic * (3.0 - p_n))
    rho_0 = I_e * (numerator / denominator)

    # Returns 3D luminosity density in L_sun / pc^3
    return rho_0 * density_shape

def calculate_dispersion_spheroid_energy(I_e, R_e_kpc, r_max_kpc, n_sersic, ellipticity, sigma_kms, upsilon_star):
    """
    Integrates a generic 3D deprojected Sersic spheroid for velocity dispersion kinetic energy.
    Supports both Ellipticals and Bulges with customizable structural index 'n'.
    """
    MSUN_TO_KG = 1.98847e30
    KMS_TO_MS = 1000.0

    if ellipticity < 0 or ellipticity >= 0.8:
        raise ValueError("Ellipticity (1 - b/a) must be between 0 (spherical) and 0.8.")
    if n_sersic <= 0:
        raise ValueError("Sersic index 'n' must be strictly positive.")

    sigma_ms = sigma_kms * KMS_TO_MS

    # Numerical Integration over ellipsoidal shells
    steps = 4000
    m_array = np.linspace(1e-4, r_max_kpc, steps)
    dm_kpc = m_array[1] - m_array[0]
    integrated_mass_msun = 0.0

    for m in m_array:
        # Get local 3D luminosity density (in L_sun / pc^3)
        rho_light = sersic_3d_density(m, I_e, R_e_kpc, n_sersic)
        rho_mass = rho_light * upsilon_star  # M_sun / pc^3
    
        m_pc = m * 1000.0
        dm_pc = dm_kpc * 1000.0
        # ellipsoidal volume element: dV = 4 * pi * (1 - epsilon) * m^2 * dm
        dV_shell_pc3 = 4.0 * np.pi * (1.0 - ellipticity) * (m_pc**2) * dm_pc
        integrated_mass_msun += rho_mass * dV_shell_pc3

    total_mass_kg = integrated_mass_msun * MSUN_TO_KG
    # K_disp = 3/2 * M * sigma^2 (Isotropic velocity ellipsoid)
    k_disp_joule = 1.5 * total_mass_kg * (sigma_ms ** 2)

    return {
        "integrated_mass_msun": integrated_mass_msun,
        "total_mass_kg": total_mass_kg,
        "kinetic_energy_joule": k_disp_joule
    }

if __name__ == "__main__":
    print("-" * 65)
    print("ENERGY INVENTORY: VELOCITY DISPERSION SERSIC SPHEROID CALCULATOR")
    print("-" * 65)
    print("Select target: [E]lliptical Galaxy body OR [B]ulge component of a disk")
    target = input("Selection [E/B]: ").strip().upper()
    while target not in ['E', 'B']:
        target = input("Invalid. Enter 'E' or 'B': ").strip().upper()

    try:
        print("\n--- Spheroid Photometric & Kinematic Parameters ---")
        i_e = float(input("Enter effective surface intensity I_e in L_sun/pc^2: "))
        
        # Coherence check for R_e (> 0)
        while True:
            try:
                r_e = float(input("Enter effective radius (R_e) in kpc: "))
                if r_e > 0:
                    break
                print("   [!] COHERENCE ERROR: Effective radius must be > 0.")
            except ValueError:
                print("   [!] Please enter a valid numerical value.")
    
        # Coherence check for r_max (> 0)
        while True:
            try:
                if target == 'E':
                    r_max = float(input("Enter maximum observable semi-major axis (a_max) in kpc: "))
                else:
                    r_max = float(input("Enter bulge outer boundary radius (R_in where disk starts) in kpc: "))
                if r_max > 0:
                    break
                print("   [!] COHERENCE ERROR: Maximum radius must be > 0.")
            except ValueError:
                print("   [!] Please enter a valid numerical value.")
        
        n_val = float(input("Enter Sersic index 'n' (e.g., 4.0 for de Vaucouleurs, 1.0 for Exponential/Pseudo-bulge): "))
        ell = float(input("Enter structural ellipticity epsilon (1 - b/a) [e.g., 0.0 for spherical, 0.3 for E3-like]: "))
        sigma_g = float(input("Enter 1D velocity dispersion sigma in km/s: "))
        upsilon = float(input("Enter stellar mass-to-light ratio M/L (Upsilon): "))
    
        res = calculate_dispersion_spheroid_energy(i_e, r_e, r_max, n_val, ell, sigma_g, upsilon)
    
        label = "ELLIPTICAL GALACTIC BODY" if target == 'E' else "GALACTIC BULGE COMPONENT"
        print("\n" + "=" * 65)
        print("                        CALCULATION RESULTS                     ")
        print("=" * 65)
        print(f"Target Structure       : {label}")
        print(f"Sersic Index (n)       : {n_val}")
        print(f"Ellipticity (epsilon)  : {ell:.2f} (b/a = {1.0 - ell:.2f})")
        print(f"Velocity Dispersion    : {sigma_g} km/s")
        print("-" * 65)
        print(f"Total Spheroid Mass    : {res['integrated_mass_msun']:.4e} M_sun ({res['total_mass_kg']:.4e} kg)")
        print(f"Total Spheroid K_Energy: {res['kinetic_energy_joule']:.4e} Joules")
        print("=" * 65)
    except Exception as e:
        print(f"\nError: {e}")

# =============================================================================
# --- END OF FILE ---
# =============================================================================
