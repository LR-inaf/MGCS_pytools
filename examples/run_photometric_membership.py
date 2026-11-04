"""Example: Spatial Photometric Decontamination with Voronoi CVT cells.

This script demonstrates statistical membership determination using concentric
annular spatial groupings, differential reddening correction, and Centroidal
Voronoi Tessellation (CVT) cells.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mgcs_pytools.statistical_membership import compute_statistical_membership
from mgcs_pytools.utils import plotting as utplot


def main():
    np.random.seed(42)

    # 1. Synthesize Mock Star Catalogs
    # Cluster stars: concentrated spatially, with tight CMD fiducial
    n_cluster = 600
    ra_cluster = np.random.normal(150.0, 0.03, n_cluster)
    dec_cluster = np.random.normal(2.0, 0.03, n_cluster)
    mag1_cluster = np.random.uniform(16.0, 21.0, n_cluster)
    # Synthetic Main Sequence: color = 0.2 * (mag1 - 16) + 0.6
    color_cluster = 0.2 * (mag1_cluster - 16.0) + 0.6 + np.random.normal(0, 0.03, n_cluster)
    mag2_cluster = mag1_cluster - color_cluster

    df_cluster = pd.DataFrame(
        {
            "ra": ra_cluster,
            "dec": dec_cluster,
            "mag_b": mag1_cluster,
            "mag_v": mag2_cluster,
            "mag_b_err": np.full(n_cluster, 0.02),
            "mag_v_err": np.full(n_cluster, 0.02),
        }
    )

    # Parallel Field stars: uniformly spread spatially and widely in CMD
    n_field = 1200
    df_field = pd.DataFrame(
        {
            "ra": np.random.uniform(149.8, 150.2, n_field),
            "dec": np.random.uniform(1.8, 2.2, n_field),
            "mag_b": np.random.uniform(15.5, 21.5, n_field),
            "mag_v": np.random.uniform(14.5, 21.0, n_field),
        }
    )

    # 2. Setup Differential Reddening Parameters
    dr_params = {
        "rband1": 3.1,
        "rband2": 2.5,
        "TO_mag": 17.5,
        "TO_color": 0.9,
        "nref": 25,
        "ord_step": 0.4,
    }

    # Rectangular ROI in the rotated (abscissa, ordinate) reference frame
    roi = np.array(
        [
            [-5.0, -5.0],
            [5.0, -5.0],
            [5.0, 5.0],
            [-5.0, 5.0],
        ]
    )

    print("Running statistical decontamination pipeline...")
    df_output = compute_statistical_membership(
        df_cluster=df_cluster,
        df_field=df_field,
        target_mag_col=["mag_b", "mag_v"],
        target_mag_col_error=["mag_b_err", "mag_v_err"],
        field_mag_col=["mag_b", "mag_v"],
        dr_params=dr_params,
        parallel_fov=0.16,
        racol="ra",
        deccol="dec",
        roi=roi,
        minstars=150,
        min_star_per_cell=5,
        membership_iter=50,
        membership_method="analytical",
        plot_dred=False,
        plot_voronoi=False,
    )

    print(
        f"Membership calculation finished for {len(df_output)} stars. "
        f"Mean probability: {df_output['membership'].mean():.3f}"
    )

    # 3. Diagnostic Plots
    fig_overview = utplot.plot_membership_overview(
        btarget=df_output["mag_b_drcorr"],
        vtarget=df_output["mag_v_drcorr"],
        bfield=df_field["mag_b"],
        vfield=df_field["mag_v"],
        membership=df_output["membership"],
    )
    fig_overview.savefig("membership_overview_demo.png", dpi=100)
    plt.close(fig_overview)

    fig_snapshot = utplot.plot_decontamination_snapshot(
        df_output["mag_b_drcorr"],
        df_output["mag_v_drcorr"],
        df_field["mag_b"],
        df_field["mag_v"],
        df_output["membership"],
        member_threshold=0.7,
    )
    fig_snapshot.savefig("membership_snapshot_demo.png", dpi=100)
    plt.close(fig_snapshot)

    print("Photometric membership analysis completed successfully! Demo plots saved.")


if __name__ == "__main__":
    main()
