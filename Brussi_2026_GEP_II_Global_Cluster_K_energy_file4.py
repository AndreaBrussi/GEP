# =============================================================================
# --- START OF FILE Brussi_2026_GEP_II_Global_Cluster_K_energy_file4.py ---
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

def calculate_gc_kinetic_energy(galaxy_type, num_clusters, avg_cluster_mass_msun, velocity_kms):
    """
    Calculates the total kinetic energy of a globular cluster (GC) system.
    All intermediate calculations and outputs are in SI units (kg, meters, Joules).

    Parameters:
    ----
    galaxy_type : str
        'disk' for spiral/disk galaxies, 'elliptical' for elliptical galaxies.
    num_clusters : int or float
        Total number of clusters in the galaxy (N).
    avg_cluster_mass_msun : float
        Average mass of a single cluster in solar masses (M_sun).
    velocity_kms : float
        If 'disk': Circular velocity V_c in km/s.
        If 'elliptical': 1D stellar velocity dispersion sigma_g in km/s.

    Returns:
    ----
    dict
        Dictionary containing the total system mass in Msun and kg, and total kinetic energy in Joules.
    """
    MSUN_TO_KG = 1.98847e30
    KMS_TO_MS = 1000.0

    m_c_kg = avg_cluster_mass_msun * MSUN_TO_KG
    M_tot_kg = num_clusters * m_c_kg
    M_tot_msun = num_clusters * avg_cluster_mass_msun
    velocity_ms = velocity_kms * KMS_TO_MS

    if galaxy_type == "disk":
        alpha = 0.5
    elif galaxy_type == "elliptical":
        alpha = 1.5
    else:
        raise ValueError("Invalid galaxy type internally mapped.")

    E_k_joule = alpha * num_clusters * m_c_kg * (velocity_ms ** 2)

    return {
        "total_mass_msun": M_tot_msun,
        "total_mass_kg": M_tot_kg,
        "kinetic_energy_joule": E_k_joule
    }


# ====
# INTERACTIVE COMMAND-LINE INTERFACE
# ====
if __name__ == "__main__":
    print("-" * 65)
    print("ENERGY INVENTORY: GLOBULAR CLUSTER KINETIC ENERGY CALCULATOR")
    print("-" * 65)
    print("Please input the parameters for your target galaxy simulation:\n")

    try:
        user_choice = input("Enter galaxy type - D for Disk / E for Elliptical: ").strip().upper()
        while user_choice not in ["D", "E"]:
            user_choice = input("Invalid input. Please enter 'D' or 'E': ").strip().upper()

        if user_choice == "D":
            g_type = "disk"
            g_name = "DISK"
        else:
            g_type = "elliptical"
            g_name = "ELLIPTICAL"

        num_gcs = float(input("Enter total number of clusters (N) [e.g., 200]: "))

        avg_mass = float(input("Enter average cluster mass in Solar Masses (M_sun) [default is 200000]: ") or 200000)

        if g_type == "disk":
            vel = float(input("Enter circular velocity V_c in km/s [e.g., 220]: "))
            vel_name = "Flat Rotation Velocity (v_flat)"
        else:
            vel = float(input("Enter 1D velocity dispersion sigma_g in km/s [e.g., 250]: "))
            vel_name = "Velocity Dispersion (sigma_g)"

        results = calculate_gc_kinetic_energy(
            galaxy_type=g_type,
            num_clusters=num_gcs,
            avg_cluster_mass_msun=avg_mass,
            velocity_kms=vel
        )

        print("\n" + "=" * 65)
        print("                    CALCULATION RESULTS                    ")
        print("=" * 65)
        print(f"Galaxy Profile Regime : {g_name}")
        print(f"Total Cluster Count   : {num_gcs:,.0f}")
        print(f"Average Cluster Mass  : {avg_mass:,.0f} M_sun")
        print(f"Kinematic Velocity    : {vel} km/s ({vel_name})")
        print("-" * 65)
        print(f"Total GC Mass         : {results['total_mass_msun']:.4e} M_sun ({results['total_mass_kg']:.4e} kg)")
        print(f"TOTAL GC KINETIC ENERGY  : {results['kinetic_energy_joule']:.4e} Joules")
        print("=" * 65)

    except ValueError:
        print("\nError: Please ensure all numeric inputs are formatted properly.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

# =============================================================================
# --- END OF FILE ---
# =============================================================================
