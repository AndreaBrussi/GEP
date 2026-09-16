# =============================================================================
# --- START OF FILE Brussi_2026_GEP_II_Total_energy_profile_file6.py ---
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
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import gamma, gammainc, ellipe


if __name__ == "__main__":
    print("-" * 75)
    print("GEP INVENTORY: CUMULATIVE ENERGY AND MASS PROFILE GENERATOR (_file6.py)")
    print("-" * 75)

    # -------------------------------------------------------------------------
    # Physical constants
    # -------------------------------------------------------------------------
    K_B = 1.380649e-23
    M_P = 1.6726219236e-27
    MU = 0.6
    MU_HI = 1.0
    T_HI = 100.0
    KPC_TO_M = 3.08567758e19
    MSUN_TO_KG = 1.98847e30
    KG_TO_MSUN = 1.0 / MSUN_TO_KG
    KMS_TO_MS = 1000.0
    PC2_TO_M2 = 9.52140614e32
    G_CM3_TO_KG_M3 = 1000.0
    C_LIGHT = 299792458.0
    L_SUN_TO_W = 3.828e26

    # Isotropic background energy densities
    u_CMB = 4.19e-14
    u_EBL = 1.60e-15
    u_iso_tot = u_CMB + u_EBL

    # Stellar neutrino luminosity correction:
    # L_esc = L_bol + L_nu ~= 1.025 L_bol
    F_NU = 0.025

    # -------------------------------------------------------------------------
    # Sérsic / Prugniel-Simien 3D luminosity density
    # Units:
    #   m_kpc, R_e_kpc in kpc
    #   I_e in L_sun / pc^2
    # Output:
    #   rho_light in L_sun / pc^3
    # -------------------------------------------------------------------------
    def sersic_3d_density(m_kpc, I_e, R_e_kpc, n_sersic):
        if m_kpc <= 0.0:
            return 0.0

        b_n = 2.0 * n_sersic - 1.0 / 3.0 + 0.00913 / n_sersic
        p_n = 1.0 - 0.6097 / n_sersic + 0.05463 / (n_sersic**2)

        nu = m_kpc / R_e_kpc
        density_shape = (nu ** (-p_n)) * np.exp(-b_n * (nu ** (1.0 / n_sersic)))

        R_e_pc = R_e_kpc * 1000.0

        numerator = b_n ** (n_sersic * (3.0 - p_n))
        denominator = (
            2.0
            * np.pi
            * R_e_pc
            * n_sersic
            * gamma(n_sersic * (3.0 - p_n))
        )

        rho_0 = I_e * (numerator / denominator)

        return rho_0 * density_shape

    # -------------------------------------------------------------------------
    # Spheroidal mass profile from the 3D Sérsic luminosity density
    # Output:
    #   mass in M_sun
    # -------------------------------------------------------------------------
    def get_sersic_mass_profile_msun(r_kpc, I_e, R_e, n, eps, upsilon):
        if r_kpc <= 0.0:
            return 0.0

        def integrand(m):
            rho_light = sersic_3d_density(m, I_e, R_e, n)
            rho_mass = rho_light * upsilon  # M_sun / pc^3
            m_pc = m * 1000.0
            dm_pc_dm_kpc = 1000.0
            return 4.0 * np.pi * (1.0 - eps) * (m_pc**2) * dm_pc_dm_kpc * rho_mass

        mass_msun, _ = quad(integrand, 1e-4, r_kpc, limit=200)
        return mass_msun

    # -------------------------------------------------------------------------
    # Mean escape path for a source at radius r_source inside a sphere
    # of boundary radius R_boundary.
    # This is the GEP I spherical local mean path.
    # Units are arbitrary but consistent; output has same units as input.
    # -------------------------------------------------------------------------
    def local_t_sphere(r_source, R_boundary):
        if R_boundary <= 0.0:
            return 0.0

        if r_source <= 0.0:
            return R_boundary

        if r_source >= R_boundary:
            return 0.5 * R_boundary

        term = (R_boundary**2 - r_source**2) / (4.0 * r_source)
        return 0.5 * R_boundary + term * np.log((R_boundary + r_source) / (R_boundary - r_source))

    # -------------------------------------------------------------------------
    # Mean escape path for a source at disk radius r_source in a thin disk
    # embedded inside a spherical control volume of radius R_boundary.
    # This follows the GEP I thin-disk angular averaging.
    # -------------------------------------------------------------------------
    def local_t_disk(r_source, R_boundary):
        if R_boundary <= 0.0:
            return 0.0

        if r_source <= 0.0:
            return R_boundary

        if r_source >= R_boundary:
            return 0.5 * R_boundary

        def inner_integrand(theta_prime):
            val = R_boundary**2 - r_source**2 * (np.cos(theta_prime) ** 2)
            if val <= 0.0:
                return 0.0

            sqrt_val = np.sqrt(val)
            k = (r_source * np.sin(theta_prime)) / sqrt_val
            k = max(0.0, min(1.0, k))

            return sqrt_val * ellipe(k) * np.sin(theta_prime)

        res, _ = quad(inner_integrand, 0.0, np.pi, limit=200)
        return res / np.pi

    # -------------------------------------------------------------------------
    # Disk luminosity fraction enclosed within r for an exponential disk.
    # The total disk luminosity is normalized within R_opt.
    # -------------------------------------------------------------------------
    def disk_luminosity_fraction(r, h_scale, R_opt):
        r_lim = min(max(r, 0.0), R_opt)

        denom = h_scale**2 * (
            1.0 - np.exp(-R_opt / h_scale) * (1.0 + R_opt / h_scale)
        )

        if denom <= 0.0:
            return 0.0

        numer = h_scale**2 * (
            1.0 - np.exp(-r_lim / h_scale) * (1.0 + r_lim / h_scale)
        )

        return max(0.0, min(1.0, numer / denom))

    # -------------------------------------------------------------------------
    # Resident escaping energy profile for a thin exponential disk.
    #
    # Here the running radius r is treated as the local control-volume boundary.
    # The source luminosity is accumulated only from sources enclosed within r.
    # The escape path is computed from each source position to the same running
    # boundary r, consistently with the cumulative-profile interpretation.
    # -------------------------------------------------------------------------
    def disk_resident_escape_energy(r, h_scale, R_opt, L_disk_si):
        if r <= 0.0:
            return 0.0

        r_src_lim = min(r, R_opt)

        if r_src_lim <= 0.0:
            return 0.0

        denom_total = h_scale**2 * (
            1.0 - np.exp(-R_opt / h_scale) * (1.0 + R_opt / h_scale)
        )

        if denom_total <= 0.0:
            return 0.0

        def integrand(x):
            return local_t_disk(x, r) * np.exp(-x / h_scale) * x

        weighted_path_integral, _ = quad(integrand, 0.0, r_src_lim, limit=200)

        mean_weighted_path_m = (weighted_path_integral / denom_total) * KPC_TO_M
        return (L_disk_si / C_LIGHT) * mean_weighted_path_m

    # -------------------------------------------------------------------------
    # Resident escaping energy profile for a spheroidal component.
    #
    # This is a spherical-path approximation for the cumulative profile, using the
    # same running-radius control-volume logic. Ellipticity enters the luminosity
    # normalization and mass/light integration through the spheroidal volume
    # factor. For the cumulative escaping profile, this avoids using a fixed global
    # coefficient as if it were local at all radii.
    # -------------------------------------------------------------------------
    def spheroid_resident_escape_energy(r, I_e, R_e, n_sersic, eps, L_sph_si, r_src_max):
        if r <= 0.0 or r_src_max <= 0.0:
            return 0.0

        r_src_lim = min(r, r_src_max)

        if r_src_lim <= 0.0:
            return 0.0

        def luminosity_weight(m):
            rho_light = sersic_3d_density(m, I_e, R_e, n_sersic)
            m_pc = m * 1000.0
            dm_pc_dm_kpc = 1000.0
            return 4.0 * np.pi * (1.0 - eps) * (m_pc**2) * dm_pc_dm_kpc * rho_light

        denom_total, _ = quad(luminosity_weight, 1e-4, r_src_max, limit=200)

        if denom_total <= 0.0:
            return 0.0

        def weighted_path_integrand(m):
            return local_t_sphere(m, r) * luminosity_weight(m)

        weighted_path_integral, _ = quad(weighted_path_integrand, 1e-4, r_src_lim, limit=200)

        mean_weighted_path_m = (weighted_path_integral / denom_total) * KPC_TO_M
        return (L_sph_si / C_LIGHT) * mean_weighted_path_m

    # -------------------------------------------------------------------------
    # Optional GEP I elliptical coefficient interpolation.
    # This is retained only as a diagnostic/reference for the final total value,
    # not as the main local cumulative calculation.
    # -------------------------------------------------------------------------
    def interpolate_nage1_oblate_coefficient(eps):
        table_eps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        table_coeff = np.array([0.8740, 0.8834, 0.8923, 0.9010, 0.9097, 0.9184, 0.9272, 0.9360])
        eps_clipped = max(table_eps.min(), min(table_eps.max(), eps))
        return float(np.interp(eps_clipped, table_eps, table_coeff))

    try:
        # ---------------------------------------------------------------------
        # User inputs
        # ---------------------------------------------------------------------
        g_type = input("Enter galaxy type - [D]isk Pure, [B]arred Disk, [E]lliptical: ").strip().lower()
        while g_type not in ["d", "b", "e"]:
            g_type = input("Invalid input. Please enter 'D', 'B', or 'E': ").strip().lower()

        R_H_def = 100.0
        R_H = float(input(f"Enter spherical Halo boundary radius (R_H) in kpc [default {R_H_def}]: ") or R_H_def)
        R_H_m = R_H * KPC_TO_M

        if g_type in ["d", "b"]:
            v_flat_def = 220.0
            v_flat = float(input(f"Enter flat rotation velocity v_flat in km/s [default {v_flat_def}]: ") or v_flat_def)
            v_flat_ms = v_flat * KMS_TO_MS

            R_in_def = 3.0
            R_in = float(input(f"Enter inner disk boundary R_in in kpc [default {R_in_def}]: ") or R_in_def)

            if g_type == "b":
                R_out_bar_def = 7.0
                R_out_bar = float(input(f"Enter bar outer boundary R_out_bar in kpc [default {R_out_bar_def}]: ") or R_out_bar_def)

                bar_angle_def = 90.0
                bar_angle = float(input(f"Enter bar angle in degrees [default {bar_angle_def}]: ") or bar_angle_def)

                bar_density_def = 1000.0
                bar_density = float(input(f"Enter bar surface density in M_sun/pc^2 [default {bar_density_def}]: ") or bar_density_def)

                R_start_disk = R_out_bar
            else:
                R_out_bar = R_in
                bar_angle = 0.0
                bar_density = 0.0
                R_start_disk = R_in

            h_def = 3.0
            h_scale = float(input(f"Enter stellar disk scale length (h) in kpc [default {h_def}]: ") or h_def)

            R_opt_def = 15.0
            R_opt = float(input(f"Enter optical disk outer radius (R_opt) in kpc [default {R_opt_def}]: ") or R_opt_def)

            I_0_def = 1000.0
            I_0 = float(input(f"Enter disk central intensity I_0 in L_sun/pc^2 [default {I_0_def}]: ") or I_0_def)

            upsilon_def = 1.2
            upsilon = float(input(f"Enter disk mass-to-light ratio Upsilon_* [default {upsilon_def}]: ") or upsilon_def)

            gas_eta_def = 1.4
            gas_eta = float(input(f"Enter gas correction factor (eta) [default {gas_eta_def}]: ") or gas_eta_def)

            has_bulge = input("Include central bulge component? [Y/n]: ").strip().lower() != "n"

            if has_bulge:
                I_e_b_def = 500.0
                I_e_b = float(input(f"   Enter bulge effective intensity I_e in L_sun/pc^2 [default {I_e_b_def}]: ") or I_e_b_def)

                R_e_b_def = 1.0
                R_e_b = float(input(f"   Enter bulge effective radius R_e in kpc [default {R_e_b_def}]: ") or R_e_b_def)

                n_b_def = 2.0
                n_b = float(input(f"   Enter bulge Sersic index n [default {n_b_def}]: ") or n_b_def)

                ell_b_def = 0.2
                ell_b = float(input(f"   Enter bulge ellipticity epsilon [default {ell_b_def}]: ") or ell_b_def)

                sigma_b_def = 120.0
                sigma_b = float(input(f"   Enter bulge velocity dispersion sigma in km/s [default {sigma_b_def}]: ") or sigma_b_def)

                upsilon_b_def = 2.0
                upsilon_b = float(input(f"   Enter bulge mass-to-light ratio Upsilon_b [default {upsilon_b_def}]: ") or upsilon_b_def)
            else:
                I_e_b = 0.0
                R_e_b = 0.0
                n_b = 0.0
                ell_b = 0.0
                sigma_b = 0.0
                upsilon_b = 0.0

            rho_0_cgm_def = 1e-26
            rho_0_cgm = float(input(f"Enter CGM central gas density in g/cm^3 [default {rho_0_cgm_def:.1e}]: ") or rho_0_cgm_def)

            r_c_cgm_def = 2.0
            r_c_cgm = float(input(f"Enter CGM gas core radius r_c in kpc [default {r_c_cgm_def}]: ") or r_c_cgm_def)

            beta_cgm_def = 0.5
            beta_cgm = float(input(f"Enter CGM slope parameter beta [default {beta_cgm_def}]: ") or beta_cgm_def)

            T_vir = (MU * M_P * (v_flat_ms**2)) / (2.0 * K_B)
            rho_0_cgm_kg_m3 = rho_0_cgm * G_CM3_TO_KG_M3
            r_c_cgm_m = r_c_cgm * KPC_TO_M

            L_bol_def = 4e10
            L_bol = float(input(f"Enter bolometric luminosity L_bol in L_sun [default {L_bol_def:.1e}]: ") or L_bol_def)

        else:
            sigma_g_def = 250.0
            sigma_g = float(input(f"Enter stellar velocity dispersion sigma_g in km/s [default {sigma_g_def}]: ") or sigma_g_def)
            sigma_g_ms = sigma_g * KMS_TO_MS

            I_e_def = 100.0
            I_e = float(input(f"Enter spheroid effective intensity I_e in L_sun/pc^2 [default {I_e_def}]: ") or I_e_def)

            R_e_def = 5.0
            R_e = float(input(f"Enter spheroid effective radius R_e in kpc [default {R_e_def}]: ") or R_e_def)

            n_val_def = 4.0
            n_val = float(input(f"Enter Sersic index n [default {n_val_def}]: ") or n_val_def)

            ell_def = 0.3
            ell = float(input(f"Enter spheroid ellipticity epsilon [default {ell_def}]: ") or ell_def)

            upsilon_def = 6.0
            upsilon = float(input(f"Enter spheroid mass-to-light ratio Upsilon [default {upsilon_def}]: ") or upsilon_def)

            rho_0_cgm_def = 1e-25
            rho_0_cgm = float(input(f"Enter CGM central gas density in g/cm^3 [default {rho_0_cgm_def:.1e}]: ") or rho_0_cgm_def)

            r_c_cgm_def = 2.0
            r_c_cgm = float(input(f"Enter CGM gas core radius r_c in kpc [default {r_c_cgm_def}]: ") or r_c_cgm_def)

            beta_cgm_def = 0.6
            beta_cgm = float(input(f"Enter CGM slope parameter beta [default {beta_cgm_def}]: ") or beta_cgm_def)

            T_vir = (MU * M_P * (sigma_g_ms**2)) / (beta_cgm * K_B)
            rho_0_cgm_kg_m3 = rho_0_cgm * G_CM3_TO_KG_M3
            r_c_cgm_m = r_c_cgm * KPC_TO_M

            L_bol_def = 8e10
            L_bol = float(input(f"Enter bolometric luminosity L_bol in L_sun [default {L_bol_def:.1e}]: ") or L_bol_def)

            C_morph_reference = interpolate_nage1_oblate_coefficient(ell)

        # ---------------------------------------------------------------------
        # Radial grid
        # ---------------------------------------------------------------------
        print("\nComputing profiles over the Halo radial range...")

        r_vals = np.logspace(-1, np.log10(R_H), 100)

        E_iso_profile = []
        E_cgm_profile = []
        E_kin_profile = []
        E_esc_profile = []

        # ---------------------------------------------------------------------
        # Disk/barred-disk auxiliary quantities
        # ---------------------------------------------------------------------
        if g_type in ["d", "b"]:
            sigma_0_gas = np.exp(1.0 / 0.36)
            gas_sigma_Ropt = 4.0
            R_s = R_opt / np.log(sigma_0_gas / gas_sigma_Ropt)

            # Stellar disk mass surface density normalization in M_sun/kpc^2
            sigma_0_kpc2 = (I_0 * upsilon) * 1e6

            # Optical luminosity weights for distributing L_esc between disk and bulge
            L_disk_opt = 2.0 * np.pi * I_0 * ((h_scale * 1000.0) ** 2)

            if has_bulge:
                b_nb = 2.0 * n_b - 1.0 / 3.0 + 0.00913 / n_b
                L_bulge_opt = (
                    I_e_b
                    * 2.0
                    * np.pi
                    * ((R_e_b * 1000.0) ** 2)
                    * (n_b * gamma(2.0 * n_b))
                    / (b_nb ** (2.0 * n_b))
                )

                w_bulge = L_bulge_opt / (L_disk_opt + L_bulge_opt)
                w_disk = 1.0 - w_bulge
            else:
                L_bulge_opt = 0.0
                w_bulge = 0.0
                w_disk = 1.0

            def freeman_mass_antiderivative(R):
                return (
                    -2.0
                    * np.pi
                    * sigma_0_kpc2
                    * h_scale
                    * (R + h_scale)
                    * np.exp(-R / h_scale)
                )

            def get_disk_kinetic_energy(r):
                # Bar kinetic energy
                if g_type == "b" and r > R_in:
                    r_bar_lim = min(r, R_out_bar)

                    if r_bar_lim > R_in:
                        bar_area_m2 = (
                            0.5
                            * np.radians(bar_angle)
                            * (((r_bar_lim * KPC_TO_M) ** 2) - ((R_in * KPC_TO_M) ** 2))
                        )
                        sigma_bar_si = bar_density * (MSUN_TO_KG / PC2_TO_M2)
                        m_bar = bar_area_m2 * sigma_bar_si
                        k_bar = 0.5 * m_bar * (v_flat_ms**2)
                    else:
                        k_bar = 0.0
                else:
                    k_bar = 0.0

                # Stellar disk kinetic energy
                if r > R_start_disk:
                    r_star_lim = min(r, R_opt)

                    if r_star_lim > R_start_disk:
                        m_star_msun = (
                            freeman_mass_antiderivative(r_star_lim)
                            - freeman_mass_antiderivative(R_start_disk)
                        )
                        k_star = 0.5 * (m_star_msun * MSUN_TO_KG) * (v_flat_ms**2)
                    else:
                        k_star = 0.0
                else:
                    k_star = 0.0

                # Outer gas kinetic energy
                if r > R_opt:
                    gas_mass_msun = (
                        2.0
                        * np.pi
                        * gas_eta
                        * (sigma_0_gas * 1e6)
                        * R_s
                        * (
                            (R_opt + R_s) * np.exp(-R_opt / R_s)
                            - (r + R_s) * np.exp(-r / R_s)
                        )
                    )
                    k_gas = 0.5 * (gas_mass_msun * MSUN_TO_KG) * (v_flat_ms**2)
                else:
                    k_gas = 0.0

                # Bulge random kinetic energy
                if has_bulge and r > 1e-4:
                    r_bulge_lim = min(r, R_in)
                    m_bulge_msun = get_sersic_mass_profile_msun(
                        r_bulge_lim, I_e_b, R_e_b, n_b, ell_b, upsilon_b
                    )
                    k_bulge = 1.5 * (m_bulge_msun * MSUN_TO_KG) * ((sigma_b * KMS_TO_MS) ** 2)
                else:
                    k_bulge = 0.0

                return k_bar + k_star + k_gas + k_bulge

        # ---------------------------------------------------------------------
        # Luminosity used for resident escaping energy
        # ---------------------------------------------------------------------
        L_bol_si = L_bol * L_SUN_TO_W
        L_esc_si = L_bol_si * (1.0 + F_NU)

        if g_type in ["d", "b"]:
            L_disk_esc_si = w_disk * L_esc_si
            L_bulge_esc_si = w_bulge * L_esc_si
        else:
            L_sph_esc_si = L_esc_si

        # ---------------------------------------------------------------------
        # Main profile loop
        # ---------------------------------------------------------------------
        for r in r_vals:
            r_m = r * KPC_TO_M

            # Isotropic background cumulative energy
            V_enc_m3 = (4.0 / 3.0) * np.pi * (r_m**3)
            E_iso = u_iso_tot * V_enc_m3
            E_iso_profile.append(E_iso)

            # CGM cumulative thermal energy
            def cgm_integrand(x):
                return (x**2) * (1.0 + (x / r_c_cgm_m) ** 2) ** (-1.5 * beta_cgm)

            cgm_mass_int, _ = quad(cgm_integrand, 0.0, r_m, limit=200)
            mass_cgm_kg = 4.0 * np.pi * rho_0_cgm_kg_m3 * cgm_mass_int
            E_cgm = 1.5 * (K_B * T_vir / (MU * M_P)) * mass_cgm_kg
            E_cgm_profile.append(E_cgm)

            # Kinetic cumulative energy
            if g_type in ["d", "b"]:
                E_kin = get_disk_kinetic_energy(r)
            else:
                m_spheroid_msun = get_sersic_mass_profile_msun(
                    r, I_e, R_e, n_val, ell, upsilon
                )
                E_kin = 1.5 * (m_spheroid_msun * MSUN_TO_KG) * (sigma_g_ms**2)

            E_kin_profile.append(E_kin)

            # Resident escaping energy cumulative profile
            if g_type in ["d", "b"]:
                E_esc_disk = disk_resident_escape_energy(
                    r, h_scale, R_opt, L_disk_esc_si
                )

                if has_bulge:
                    E_esc_bulge = spheroid_resident_escape_energy(
                        r,
                        I_e_b,
                        R_e_b,
                        n_b,
                        ell_b,
                        L_bulge_esc_si,
                        R_in,
                    )
                else:
                    E_esc_bulge = 0.0

                E_esc = E_esc_disk + E_esc_bulge

            else:
                E_esc = spheroid_resident_escape_energy(
                    r,
                    I_e,
                    R_e,
                    n_val,
                    ell,
                    L_sph_esc_si,
                    R_H,
                )

            E_esc_profile.append(E_esc)

        # ---------------------------------------------------------------------
        # Total masses
        # ---------------------------------------------------------------------
        print("\nEvaluating total system masses...")

        if g_type in ["d", "b"]:
            mass_cgm_total_msun = (
                E_cgm_profile[-1]
                / (1.5 * (K_B * T_vir / (MU * M_P)))
                / MSUN_TO_KG
            )

            m_star_total_msun = (
                freeman_mass_antiderivative(R_opt)
                - freeman_mass_antiderivative(R_start_disk)
            )

            m_gas_total_msun = (
                2.0
                * np.pi
                * gas_eta
                * (sigma_0_gas * 1e6)
                * R_s
                * (
                    (R_opt + R_s) * np.exp(-R_opt / R_s)
                    - (R_H + R_s) * np.exp(-R_H / R_s)
                )
            )

            if has_bulge:
                m_bulge_total_msun = get_sersic_mass_profile_msun(
                    R_in, I_e_b, R_e_b, n_b, ell_b, upsilon_b
                )
            else:
                m_bulge_total_msun = 0.0

            total_baryonic_mass_msun = (
                m_star_total_msun
                + m_gas_total_msun
                + m_bulge_total_msun
                + mass_cgm_total_msun
            )

        else:
            m_spheroid_total_msun = get_sersic_mass_profile_msun(
                R_H, I_e, R_e, n_val, ell, upsilon
            )

            m_cgm_total_msun = (
                E_cgm_profile[-1]
                / (1.5 * (K_B * T_vir / (MU * M_P)))
                / MSUN_TO_KG
            )

            total_baryonic_mass_msun = m_spheroid_total_msun + m_cgm_total_msun

        # ---------------------------------------------------------------------
        # Convert profiles to arrays and compute total profile
        # ---------------------------------------------------------------------
        E_iso_profile = np.array(E_iso_profile)
        E_cgm_profile = np.array(E_cgm_profile)
        E_kin_profile = np.array(E_kin_profile)
        E_esc_profile = np.array(E_esc_profile)

        E_total_profile = (
            E_iso_profile
            + E_cgm_profile
            + E_kin_profile
            + E_esc_profile
        )

        total_baryonic_mass_kg = total_baryonic_mass_msun * MSUN_TO_KG

        # ---------------------------------------------------------------------
        # Compute local energy densities and pressures (for the right panel)
        # ---------------------------------------------------------------------
        print("Computing local volumetric energy densities and pressures...")
        r_m_vals = r_vals * KPC_TO_M

        # 1. Isotropic Background (constant)
        u_bkg_local = u_CMB + u_EBL
        P_bkg_profile = np.ones_like(r_vals) * (1.0 / 3.0) * u_bkg_local

        # 2. CGM Hot Gas
        P_cgm_profile = []
        for r in r_vals:
            rho_gas_local = rho_0_cgm_kg_m3 * (1.0 + (r / r_c_cgm)**2) ** (-1.5 * beta_cgm)
            if g_type in ["d", "b"]:
                # Consistent with Eq 49: U_CGM = 3/8 * rho_gas * v_flat^2, P_CGM = 2/3 * U_CGM = 1/4 * rho_gas * v_flat^2
                P_local = 0.25 * rho_gas_local * (v_flat_ms**2)
            else:
                # Consistent with Eq 50: U_CGM = 3/2 * rho_gas * sigma_g^2 / beta, P_CGM = 2/3 * U_CGM = rho_gas * sigma_g^2 / beta
                P_local = rho_gas_local * (sigma_g_ms**2) / beta_cgm
            P_cgm_profile.append(P_local)
        P_cgm_profile = np.array(P_cgm_profile)

        # 3. Escaping Radiation (GEP I) - Numerical derivative from cumulative
        dE_esc_dr = np.gradient(E_esc_profile, r_m_vals)
        U_esc_profile = dE_esc_dr / (4.0 * np.pi * (r_m_vals**2))
        U_esc_profile = np.maximum(1e-35, U_esc_profile)  # avoid log of zero
        P_esc_profile = (1.0 / 3.0) * U_esc_profile

        # 4. Kinetic/Dispersive Pressure - Numerical derivative from cumulative
        dE_kin_dr = np.gradient(E_kin_profile, r_m_vals)
        U_kin_profile = dE_kin_dr / (4.0 * np.pi * (r_m_vals**2))
        U_kin_profile = np.maximum(1e-35, U_kin_profile)  # avoid log of zero
        P_kin_profile = (2.0 / 3.0) * U_kin_profile

        # Smooth out the numerical step-discontinuity artifact at the disk boundary (R_in)
        if len(P_kin_profile) > 4:
            P_kin_profile = np.convolve(P_kin_profile, np.ones(3)/3.0, mode='same')
            P_kin_profile[0] = P_kin_profile[1]
            P_kin_profile[-1] = P_kin_profile[-2]

        # Total local pressure support
        P_total_profile = P_bkg_profile + P_cgm_profile + P_esc_profile + P_kin_profile

        # ---------------------------------------------------------------------
        # Final output
        # ---------------------------------------------------------------------
        print("\n" + "=" * 65)
        print("                    FINAL INTEGRATED RESULTS                    ")
        print("=" * 65)
        print(f"Total Baryonic Mass (Stars + Gas)  : {total_baryonic_mass_msun:.4e} M_sun ({total_baryonic_mass_kg:.4e} kg)")
        print(f"Total Hot CGM Virial Temperature   : {T_vir:.4e} K")
        print("-" * 65)
        print(f"Total Isotropic Background Energy  : {E_iso_profile[-1]:.4e} J")
        print(f"Total CGM Hot Gas Thermal Energy   : {E_cgm_profile[-1]:.4e} J")
        print(f"Total Baryonic Kinetic Energy      : {E_kin_profile[-1]:.4e} J")
        print(f"Total Escaping Energy (GEP I + nu): {E_esc_profile[-1]:.4e} J")
        print(f"GRAND TOTAL ACCUMULATED ENERGY     : {E_total_profile[-1]:.4e} J")
        print("-" * 65)
        print("            LOCAL PRESSURE SUPPORT AT HALO BOUNDARY (R_H)       ")
        print("-" * 65)
        print(f"Isotropic Background Pressure      : {P_bkg_profile[-1]:.4e} Pa")
        print(f"CGM Hot Gas Thermal Pressure       : {P_cgm_profile[-1]:.4e} Pa")
        print(f"Baryonic Kinetic/Dynamical Pressure: {P_kin_profile[-1]:.4e} Pa")
        print(f"Escaping Radiation Pressure        : {P_esc_profile[-1]:.4e} Pa")
        print(f"TOTAL LOCAL PRESSURE SUPPORT       : {P_total_profile[-1]:.4e} Pa")

        if g_type == "e":
            print("-" * 65)
            print(f"GEP I oblate reference coefficient: {C_morph_reference:.4f}")

        print("=" * 65)

        # ---------------------------------------------------------------------
        # Plot (Dual-panel Layout: Left = Energy, Right = Pressure)
        # ---------------------------------------------------------------------
        plt.style.use(
            "seaborn-v0_8-whitegrid"
            if "seaborn-v0_8-whitegrid" in plt.style.available
            else "default"
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

        # --- LEFT PANEL: Cumulative Energies ---
        ax1.loglog(
            r_vals,
            E_kin_profile,
            label="Baryonic Kinetic Energy ($K$)",
            color="royalblue",
            linewidth=2.0,
        )

        ax1.loglog(
            r_vals,
            E_cgm_profile,
            label="Hot CGM Thermal Energy ($E_{\\rm CGM}$)",
            color="crimson",
            linewidth=2.0,
        )

        ax1.loglog(
            r_vals,
            E_iso_profile,
            label="Isotropic Backgrounds ($E_u$)",
            color="forestgreen",
            linestyle="--",
            linewidth=1.8,
        )

        ax1.loglog(
            r_vals,
            E_esc_profile,
            label="Resident Escaping Energy ($E_{\\rm esc}$, GEP I + $\\nu$)",
            color="orange",
            linestyle="-.",
            linewidth=1.8,
        )

        ax1.loglog(
            r_vals,
            E_total_profile,
            label="TOTAL Cumulative Energy ($E_{\\rm tot}$)",
            color="black",
            linewidth=2.5,
        )

        ax1.set_xlabel(r"$\lg(r)$ (kpc)", fontsize=11)
        ax1.set_ylabel(r"$\lg(E(<r))$ (Joules)", fontsize=11)
        ax1.set_xlim(0.1, R_H)
        ax1.grid(True, which="both", ls="-", alpha=0.5)
        ax1.legend(
            frameon=True,
            facecolor="white",
            edgecolor="lightgray",
            fontsize=10,
            loc="lower right",
        )

        # --- RIGHT PANEL: Local Pressures ---
        ax2.loglog(
            r_vals,
            P_kin_profile,
            label="Kinetic Dynamical Pressure ($P_{\\rm kin}$)",
            color="royalblue",
            linewidth=2.0,
        )

        ax2.loglog(
            r_vals,
            P_cgm_profile,
            label="CGM Gas Thermal Pressure ($P_{\\rm CGM}$)",
            color="crimson",
            linewidth=2.0,
        )

        ax2.loglog(
            r_vals,
            P_bkg_profile,
            label="Isotropic Background Pressure ($P_{\\rm bkg}$)",
            color="forestgreen",
            linestyle="--",
            linewidth=1.8,
        )

        ax2.loglog(
            r_vals,
            P_esc_profile,
            label="Radiation Pressure ($P_{\\rm esc}$, GEP I + $\\nu$)",
            color="orange",
            linestyle="-.",
            linewidth=1.8,
        )

        ax2.loglog(
            r_vals,
            P_total_profile,
            label="TOTAL Hydrostatic Pressure ($P_{\\rm tot}$)",
            color="black",
            linewidth=2.5,
        )

        ax2.set_xlabel(r"$\lg(r)$ (kpc)", fontsize=11)
        ax2.set_ylabel(r"$\lg(P(r))$ (Pascals)", fontsize=11)
        ax2.set_xlim(0.1, R_H)
        # Optimized scale for local pressures in standard halos (from 1e-20 to 1e-8 Pa)
        ax2.set_ylim(1e-20, 1e-6)
        ax2.grid(True, which="both", ls="-", alpha=0.5)
        ax2.legend(
            frameon=True,
            facecolor="white",
            edgecolor="lightgray",
            fontsize=10,
            loc="lower left",
        )

        # Main overall title
        if g_type == "d":
            title_label = "Pure Disk"
        elif g_type == "b":
            title_label = "Barred Disk"
        else:
            title_label = "Elliptical"

        fig.suptitle(
            f"GEP II Diagnostics: Standard {title_label} Reference Case",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        
        
        # --- Add vertical dashed lines for physical boundaries ---
        if g_type in ["d", "b"]:
            for ax in [ax1, ax2]:
                # Draw vertical lines
                ax.axvline(R_in, color="gray", linestyle=":", alpha=0.6, linewidth=1.2)
                ax.axvline(R_opt, color="gray", linestyle=":", alpha=0.6, linewidth=1.2)
                
                # Dynamically compute Y position near the top (95% of the vertical log-scale span)
                ymin, ymax = ax.get_ylim()
                y_pos = 10**(np.log10(ymin) + 0.95 * (np.log10(ymax) - np.log10(ymin)))
                
                
                # Place text labels near the top
                ax.text(R_in * 1.15, y_pos, r"$R_{\rm in}$", color="gray", fontsize=10)
                ax.text(R_opt * 1.15, y_pos, r"$R_{\rm opt}$", color="gray", fontsize=10)
        else:
            for ax in [ax1, ax2]:
                # Spheroid effective radius (R_e)
                ax.axvline(R_e, color="gray", linestyle=":", alpha=0.6, linewidth=1.2)
                
                # Dynamically compute Y position near the top (95% of the vertical log-scale span)
                ymin, ymax = ax.get_ylim()
                y_pos = 10**(np.log10(ymin) + 0.95 * (np.log10(ymax) - np.log10(ymin)))
                
                # Place text label near the top
                ax.text(R_e * 1.15, y_pos, r"$R_e$", color="gray", fontsize=10)

    except Exception as e:
        print(f"\nAn error occurred during calculations or plotting: {e}")

# =============================================================================
# --- END OF FILE ---
# =============================================================================
