# =============================================================================
# --- START OF FILE Brussi_2026_GEP_II_Ordered_rotational_K_energy_file3.py ---
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

# =============================================================================
# ENERGY INVENTORY: ROTATIONAL BARYONIC KINETIC ENERGY CALCULATOR
# =============================================================================
if __name__ == "__main__":
    print("-" * 65)
    print("ENERGY INVENTORY: ROTATIONAL BARYONIC KINETIC ENERGY & THERMAL CALCULATOR")
    print("-" * 65)

    # Astrophysical Constants & Conversion Factors to SI
    MSUN_TO_KG = 1.98847e30        # 1 Solar Mass in kg
    KMS_TO_MS = 1000.0             # 1 km/s in m/s
    PC2_TO_M2 = 9.52140614e32      # 1 pc^2 in m^2
    KPC_TO_M = 3.08567758e19       # 1 kpc in meters
    K_B = 1.380649e-23             # Boltzmann constant in J/K
    M_P = 1.6726219236e-27         # Proton mass in kg
    MU_HI = 1.0                    # Mean molecular weight for neutral atomic hydrogen
    T_HI = 100.0                   # Characteristic temperature of cold HI (CNM) in Kelvin

    try:
        # Input: Galaxy Morphology
        g_type = input("Enter galaxy type - [D]isk Pure, [B]arred Disk: ").strip().lower()
        while g_type not in ["d", "b"]:
            g_type = input("Invalid input. Please enter 'D' or 'B': ").strip().lower()

        # Input: Kinematics
        while True:
            try:
                vel_flat = float(input("Enter flat rotation velocity v_flat in km/s: "))
                if vel_flat > 0:
                    break
                print("   [!] COHERENCE ERROR: v_flat must be strictly positive (> 0).")
            except ValueError:
                print("   [!] Please enter a valid numerical value.")
        v_flat_ms = vel_flat * KMS_TO_MS

        print("\n--- Geometric Radial Boundaries ---")

        # R_in: Inner boundary
        while True:
            R_in = float(input("Enter inner boundary radius (R_in) in kpc: "))
            if R_in > 0:
                break
            print("   [!] COHERENCE ERROR: Inner radius must be > 0.")

        # R_out_bar: Bar structures (only if type B)
        if g_type == "b":
            while True:
                R_out_bar = float(input("Enter bar structures outer radius (R_out_bar) in kpc: "))
                if R_out_bar > R_in:
                    break
                print(f"   [!] COHERENCE ERROR: Must be > R_in ({R_in} kpc).")
            bar_angle_deg = float(input("Enter total angle of bar structures (degrees): "))
            bar_density = float(input("Enter bar surface density (M_sun/pc^2): "))
            R_start_disk = R_out_bar
        else:
            R_out_bar = R_in
            bar_angle_deg, bar_density = 0.0, 0.0
            R_start_disk = R_in

        # h: Disk scale length
        h_scale = float(input("Enter stellar disk scale length (h) in kpc: "))
        
        # R_opt: Optical radius
        while True:
            R_opt = float(input("Enter optical disk outer radius (R_opt) in kpc: "))
            if R_opt > R_start_disk:
                break
            print(f"   [!] COHERENCE ERROR: Must be > previous boundary ({R_start_disk} kpc).")

        print("\n--- Stellar Disk & Gas Parameters ---")
        i_0 = float(input("Enter central intensity I_0 (L_sun/pc^2): "))
        upsilon = float(input("Enter mass-to-light ratio Upsilon_*: "))
        
        eta_input = input("Enter gas correction factor (eta) [default 1.4]: ").strip()
        gas_eta = float(eta_input) if eta_input else 1.4

        # Fixed parameter for gas anchor
        gas_sigma_Ropt = 4.0 

        # --- CALCULATIONS ---

        # 1. Bar Component
        if g_type == "b":
            bar_area_m2 = 0.5 * np.radians(bar_angle_deg) * ((R_out_bar * KPC_TO_M)**2 - (R_in * KPC_TO_M)**2)
            sigma_bar_si = bar_density * (MSUN_TO_KG / PC2_TO_M2)
            bar_mass_kg = bar_area_m2 * sigma_bar_si
            bar_mass_msun = bar_mass_kg / MSUN_TO_KG
            bar_ke_joule = 0.5 * bar_mass_kg * (v_flat_ms**2)
        else:
            bar_mass_kg, bar_mass_msun, bar_ke_joule = 0.0, 0.0, 0.0

        # 2. Stellar Component
        sigma_0_kpc2 = (i_0 * upsilon) * 1e6
        def freeman_mass(R, h):
            return -2.0 * np.pi * sigma_0_kpc2 * h * (R + h) * np.exp(-R / h)

        stellar_mass_msun = freeman_mass(R_opt, h_scale) - freeman_mass(R_start_disk, h_scale)
        stellar_mass_kg = stellar_mass_msun * MSUN_TO_KG
        stellar_ke_joule = 0.5 * stellar_mass_kg * (v_flat_ms**2)

        # 3. Gaseous Component
        sigma_0_gas = np.exp(1.0 / 0.36)
        R_s = R_opt / np.log(sigma_0_gas / gas_sigma_Ropt)
        gas_mass_msun = 2.0 * np.pi * gas_eta * (sigma_0_gas * 1e6) * R_s * (R_opt + R_s) * np.exp(-R_opt / R_s)
        gas_mass_kg = gas_mass_msun * MSUN_TO_KG
        gas_ke_joule = 0.5 * gas_mass_kg * (v_flat_ms**2)
        
        # --- NEW: Gaseous Thermal Energy (HI CNM phase) ---
        gas_th_joule = 1.5 * (gas_mass_kg / (MU_HI * M_P)) * K_B * T_HI

        # Totals
        total_mass_kg = bar_mass_kg + stellar_mass_kg + gas_mass_kg
        total_mass_msun = total_mass_kg / MSUN_TO_KG
        total_ke_joule = bar_ke_joule + stellar_ke_joule + gas_ke_joule

        # --- RESULTS ---
        print("\n" + "=" * 65)
        print("                    CALCULATION RESULTS                    ")
        print("=" * 65)
        if g_type == "b":
            print(f"BAR STRUCTURES MASS       : {bar_mass_msun:.4e} M_sun ({bar_mass_kg:.4e} kg)")
            print(f"BAR KINETIC ENERGY        : {bar_ke_joule:.4e} Joules")
        
        print(f"STELLAR DISK MASS         : {stellar_mass_msun:.4e} M_sun ({stellar_mass_kg:.4e} kg)")
        print(f"STELLAR KINETIC ENERGY    : {stellar_ke_joule:.4e} Joules")
        
        print(f"GASEOUS ENVELOPE MASS     : {gas_mass_msun:.4e} M_sun ({gas_mass_kg:.4e} kg)")
        print(f"GASEOUS KINETIC ENERGY    : {gas_ke_joule:.4e} Joules")
        print(f"GASEOUS THERMAL ENERGY    : {gas_th_joule:.4e} Joules") # Stampato qui
        
        print("-" * 65)
        print(f"TOTAL ROTATIONAL MASS     : {total_mass_msun:.4e} M_sun ({total_mass_kg:.4e} kg)")
        print(f"TOTAL ROTATIONAL K_ENERGY : {total_ke_joule:.4e} Joules")
        print("=" * 65)

    except Exception as e:
        print(f"\nAn error occurred: {e}")

# =============================================================================
# --- END OF FILE ---
# =============================================================================
