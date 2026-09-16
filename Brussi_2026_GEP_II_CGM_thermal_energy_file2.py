# =============================================================================
# --- START OF FILE Brussi_2026_GEP_II_CGM_thermal_energy_file2 ---
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
from scipy.integrate import quad

# =============================================================================
# ENERGY INVENTORY: HOT CGM THERMAL ENERGY & MASS CALCULATOR
# =============================================================================
if __name__ == "__main__":
    print("-" * 65)
    print("ENERGY INVENTORY: HOT CGM THERMAL ENERGY & MASS CALCULATOR")
    print("-" * 65)

    # Astrophysical & Physical Constants (SI Units)
    K_B = 1.380649e-23             # Boltzmann constant in J/K
    M_P = 1.6726219236e-27         # Proton mass in kg
    MU = 0.6                       # Mean molecular weight (fully ionized gas)
    KPC_TO_M = 3.08567758e19       # 1 kpc in meters
    G_CM3_TO_KG_M3 = 1000.0        # Conversion from g/cm^3 to kg/m^3
    KG_TO_MSUN = 1.0 / 1.98847e30  # Conversion from kg to Solar Masses

    try:
        # Input: Galaxy Morphology
        g_type = input("Enter galaxy type - [D]isk, [E]lliptical: ").strip().lower()
        while g_type not in ["d", "e"]:
            g_type = input("Invalid input. Please enter 'D' or 'E': ").strip().lower()

        # Input: Spherical Halo boundary
        while True:
            try:
                R_H = float(input("Enter spherical Halo boundary radius (R_H) in kpc: "))
                if R_H > 0:
                    break
                print("   [!] COHERENCE ERROR: Halo radius must be > 0.")
            except ValueError:
                print("   [!] Please enter a valid numerical value.")

        R_H_m = R_H * KPC_TO_M

        print("\n--- Kinematic Parameters ---")
        if g_type == "d":
            # Disk Galaxies: Temperature linked to v_flat
            while True:
                try:
                    v_flat = float(input("Enter flat rotation velocity (v_flat) in km/s: "))
                    if v_flat > 0:
                        break
                    print("   [!] COHERENCE ERROR: Velocity must be > 0.")
                except ValueError:
                    print("   [!] Please enter a valid numerical value.")
            
            # Convert velocity to m/s
            v_flat_m_s = v_flat * 1000.0

        else:
            # Elliptical Galaxies: Temperature linked to stellar velocity dispersion sigma_g
            while True:
                try:
                    sigma_g = float(input("Enter 1D stellar velocity dispersion (sigma_g) in km/s: "))
                    if sigma_g > 0:
                        break
                    print("   [!] COHERENCE ERROR: Velocity dispersion must be > 0.")
                except ValueError:
                    print("   [!] Please enter a valid numerical value.")
            
            # Convert velocity dispersion to m/s
            sigma_g_m_s = sigma_g * 1000.0

        print("\n--- Isothermal Beta-Model Parameters ---")
        # Central gas mass density rho_0_gas
        while True:
            try:
                rho_0_gas_g_cm3 = float(input("Enter central gas density (rho_0_gas) in g/cm^3 (e.g., 1e-26): "))
                if rho_0_gas_g_cm3 > 0:
                    break
                print("   [!] COHERENCE ERROR: Central density must be > 0.")
            except ValueError:
                print("   [!] Please enter a valid numerical value.")

        rho_0_gas_kg_m3 = rho_0_gas_g_cm3 * G_CM3_TO_KG_M3

        # Core radius r_c
        while True:
            try:
                r_c = float(input("Enter gas core radius (r_c) in kpc: "))
                if r_c > 0:
                    break
                print("   [!] COHERENCE ERROR: Core radius must be > 0.")
            except ValueError:
                print("   [!] Please enter a valid numerical value.")

        r_c_m = r_c * KPC_TO_M

        # Beta parameter
        while True:
            try:
                beta_input = input("Enter gas slope parameter (beta) [default 0.5]: ").strip()
                beta = float(beta_input) if beta_input else 0.5
                if beta > 0:
                    break
                print("   [!] COHERENCE ERROR: Beta must be > 0.")
            except ValueError:
                print("   [!] Please enter a valid numerical value.")

        # --- AFTER ALL INPUTS: Calculate Virial Temperature ---
        if g_type == "d":
            T_vir = (MU * M_P * (v_flat_m_s**2)) / (2.0 * K_B)
        else:
            T_vir = (MU * M_P * (sigma_g_m_s**2)) / (beta * K_B)

        # 1D Integrand for Mass Integration: 4 * pi * r^2 * rho_gas(r)
        def mass_integrand(r):
            rho = rho_0_gas_kg_m3 * (1.0 + (r / r_c_m)**2)**(-1.5 * beta)
            return 4.0 * np.pi * (r**2) * rho

        # Integrate to find the total gas mass in kg
        mass_result, _ = quad(mass_integrand, 0.0, R_H_m)
        total_mass_msun = mass_result * KG_TO_MSUN

        # Calculate Total Thermal Energy from the Mass and Temperature
        # E_th = (3/2) * (k_B * T / (mu * m_p)) * M_gas
        total_energy_joule = 1.5 * (K_B * T_vir / (MU * M_P)) * mass_result

        # --- RESULTS ---
        print("\n" + "=" * 65)
        print("                    CALCULATION RESULTS                    ")
        print("=" * 65)
        print(f"VIRIAL TEMPERATURE (T_vir)   : {T_vir:.4e} K")
        print(f"TOTAL HOT CGM BARYON MASS    : {total_mass_msun:.4e} M_sun ({mass_result:.4e} kg)")
        print(f"TOTAL CGM THERMAL ENERGY     : {total_energy_joule:.4e} Joules")
        print("=" * 65)

    except Exception as e:
        print(f"\nAn error occurred: {e}")

# =============================================================================
# --- END OF FILE ---
# =============================================================================
