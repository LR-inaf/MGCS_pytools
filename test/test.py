import os
import pandas as pd
import numpy as np

import statistical_membership.membership as stat_mem


def main():
    CATALOG_PATH = os.path.abspath("/home/linux-machine/work/NGC6749/catalogs")
    OUTPUT_PATH = os.path.abspath(
        "/home/linux-machine/work/NGC6749/pyMGCS_tools/output"
    )

    # v1cat = "/home/linux-machine/NGC6749/NGC6749.CAT.v1"

    df_cat_v1 = pd.read_csv(os.path.join(CATALOG_PATH, "NGC6749.CAT.v1"), sep="\s+")
    df_cat_v1 = df_cat_v1[df_cat_v1["oktot(54)"] == 1]
    df_cat_v1["membership"] = 0.0

    cluster_mag = ["m6061c(6)", "m8141c(7)"]
    cluster_mag_corr = [col + "_rdcorr" for col in cluster_mag]
    field_mag = ["m6062c(8)", "m8142c(9)"]

    df_field = df_cat_v1[df_cat_v1[field_mag[0]] != 0.0]
    df_cluster = df_cat_v1[df_cat_v1[cluster_mag[0]] != 0.0]

    fov_wfc3 = 163.0**2  # arcsec squared
    fov_acs = 202.0**2  # arcsec squared

    # reload(stat_mem)

    # setting up the parameters for the reddening correction
    dr_params = {
        "rband1": 0.903,
        "rband2": 0.597,
        "TO_mag": 22,
        "TO_color": 1.75,
        "nref": 20,
        "xcol": "x(3)",
        "ycol": "y(4)",
        "band1": cluster_mag[0],
        "band2": cluster_mag[1],
        "ord_step": 0.2,
    }

    # ord_max = -0.4  # 0.40
    # ord_min = -2.0  # -0.05
    # roi = np.array([[1.0, ord_max], [2.1, ord_max], [0.3, ord_min], [-0.2, ord_min]])
    # roi = np.array([[-0.8, ord_max], [0, ord_max], [-4, ord_min], [-5, ord_min]])
    # compute ememberhsip and reddening correction
    df_cluster_corr = stat_mem.do_statistical_membership(
        df_cluster,
        df_field,
        field_mag,
        dr_params,
        racol="ra(1)",
        deccol="dec(2)",
        fov_ratio=fov_acs / fov_wfc3,
        minstars=1000,
        memebership_iter=100,
        member_threshold=0.9,
        plot_dred=True,
        plot_voronoi=True,
    )


if __name__ == "__main__":
    main()
