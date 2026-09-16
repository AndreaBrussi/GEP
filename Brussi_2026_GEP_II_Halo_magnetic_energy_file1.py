# =============================================================================
# --- START OF FILE Brussi_2026_GEP_II_Halo_magnetic_energy_file1.py ---
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
from scipy.integrate import dblquad, quad

# =============================================================================
# ENERGY INVENTORY: TOTAL HALO MAGNETIC ENERGY CALCULATOR
# =============================================================================
if __name__ == "__main__":
    print("-" * 65)
    print("ENERGY INVENTORY: TOTAL HALO MAGNETIC ENERGY CALCULATOR")
    print("-" * 65)

    # Astrophysical Constants & Conversion Factors to SI
    MU_0 = 4.0 * np.pi * 1e-7      # Vacuum permeability in H/m (or T*m/A)
    KPC_TO_M = 3.08567758e19       # 1 kpc in meters
    UG_TO_T = 1e-10                # 1 microGauss in Tesla

    try:
        # Input: Galaxy Morphology
        g_type = input("Enter galaxy type - [D]isk Pure, [B]arred Disk, [E]lliptical: ").strip().lower()
        while g_type not in ["d", "b", "e"]:
            g_type = input("Invalid input. Please enter 'D', 'B', or 'E': ").strip().lower()

        # Input: Spherical Halo boundary
        while True:
            try:
                R_H = float(input("Enter spherical Halo boundary radius (R_H) in kpc: "))
                if R_H > 0:
                    break
                print("   [!] COHERENCE ERROR: Halo radius must be > 0.")
            except ValueError:
                print("   [!] Please enter a valid numerical value.")

        # Spherical Halo Volume calculations
        R_H_m = R_H * KPC_TO_M
        V_halo_m3 = (4.0 / 3.0) * np.pi * (R_H_m**3)

        print("\n--- Magnetic Field Profile Parameters ---")

        if g_type in ["d", "b"]:
            # Disk systems
            while True:
                B_0 = float(input("Enter central magnetic field strength (B_0) in microGauss: "))
                if B_0 > 0:
                    break
                print("   [!] COHERENCE ERROR: Central field must be > 0.")

            if g_type == "b":
                f_bar = float(input("Enter bar magnetic enhancement factor (F_bar): "))
                B_0_eff = B_0 * f_bar
            else:
                B_0_eff = B_0

            R_B = float(input("Enter radial magnetic scale length (R_B) in kpc: "))
            
            # Distinct prompt for vertical scale height depending on morphology (z_B vs z_B,bar)
            if g_type == "b":
                z_B_eff = float(input("Enter barred vertical magnetic scale height (z_B,bar) in kpc: "))
            else:
                z_B_eff = float(input("Enter vertical magnetic scale height (z_B) in kpc: "))

            # Convert to SI for integration
            R_B_m = R_B * KPC_TO_M
            z_B_m = z_B_eff * KPC_TO_M
            B_0_T = B_0_eff * UG_TO_T

            # Integration integrand: r^2 * sin(theta) * u_B(r, theta)
            # Cylindrical mapping: R = r * sin(theta), z = r * cos(theta)
            def integrand_disk(theta, r):
                R = r * np.sin(theta)
                z = r * np.cos(theta)
                B = B_0_T * np.exp(-R / R_B_m) * np.exp(-np.abs(z) / z_B_m)
                u_B = (B**2) / (2.0 * MU_0)
                return (r**2) * np.sin(theta) * u_B

            # Double numerical integration over theta [0, pi] and r [0, R_H]
            result, _ = dblquad(integrand_disk, 0.0, R_H_m, lambda r: 0.0, lambda r: np.pi)
            total_energy_joule = 2.0 * np.pi * result

        else:
            # Elliptical systems
            while True:
                B_0 = float(input("Enter central magnetic field strength (B_0) in microGauss: "))
                if B_0 > 0:
                    break
                print("   [!] COHERENCE ERROR: Central field must be > 0.")
            
            r_c = float(input("Enter gas core radius (r_c) in kpc: "))
            
            alpha_input = input("Enter magnetic scaling index (alpha_B) [default 0.5]: ").strip()
            alpha_B = float(alpha_input) if alpha_input else 0.5

            # Convert to SI for integration
            r_c_m = r_c * KPC_TO_M
            B_0_T = B_0 * UG_TO_T

            # 1D integration integrand over r
            def integrand_ell(r):
                B = B_0_T * (1.0 + (r / r_c_m)**2)**(-alpha_B)
                u_B = (B**2) / (2.0 * MU_0)
                return (r**2) * u_B

            result, _ = quad(integrand_ell, 0.0, R_H_m)
            total_energy_joule = 4.0 * np.pi * result

        # Derive RMS magnetic field strength inside the Halo volume
        B_rms_T = np.sqrt((2.0 * MU_0 * total_energy_joule) / V_halo_m3)
        B_rms_uG = B_rms_T / UG_TO_T

        # --- RESULTS ---
        print("\n" + "=" * 65)
        print("                    CALCULATION RESULTS                    ")
        print("=" * 65)
        print(f"TOTAL HALO MAGNETIC ENERGY: {total_energy_joule:.4e} Joules")
        print(f"RMS HALO MAGNETIC FIELD   : {B_rms_uG:.4e} microGauss")
        print("=" * 65)

    except Exception as e:
        print(f"\nAn error occurred: {e}")

# =============================================================================
# --- END OF FILE ---
# =============================================================================
