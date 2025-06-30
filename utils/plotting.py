import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator

from scipy.spatial import voronoi_plot_2d

import numpy as np

from shapely import plotting

from astropy.coordinates import SkyCoord
import astropy.units as u


def plot_cmd(ax, df, mag_col, color="black", cmap=None, inverty=True):
    ax.scatter(
        df[mag_col[0]] - df[mag_col[1]],
        df[mag_col[0]],
        s=1,
        lw=0.3,
        c=color,
        cmap=cmap,
        alpha=0.3,
    )

    if inverty:
        ax.invert_yaxis()


def plot_cmd_comparison(
    cluster_df,
    field_df,
    mag_col_cluster,
    mag_col_field,
    title,
    xlim=None,
    ylim=None,
    inverty=True,
):
    fig, [[ax, ax2], [ax3, ax4]] = plt.subplots(
        2, 2, layout="constrained", figsize=(12, 12)
    )

    ax.set_title("CMD Target")
    plot_cmd(ax, cluster_df, mag_col_cluster, color="black", inverty=inverty)
    ax.set_xlabel("$(m_{\mathrm{F606W}} - m_{\mathrm{F814W}})$")
    ax.set_ylabel("$m_{\mathrm{F606W}}$")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax2.set_title("CMD Field")
    plot_cmd(ax2, field_df, mag_col_field, color="black", inverty=inverty)
    ax2.set_xlabel("$(m_{\mathrm{F606W}} - m_{\mathrm{F814W}})$")
    ax2.set_ylabel("$m_{\mathrm{F606W}}$")
    ax2.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())

    ax3.set_title("CMD Target with membership")
    plot_cmd(
        ax3,
        cluster_df,
        mag_col_cluster,
        color=cluster_df["membership"],
        cmap="jet_r",
        inverty=inverty,
    )
    _ = ax3.figure.colorbar(
        cm.ScalarMappable(cmap="jet_r"),
        ax=ax3,
        orientation="horizontal",
        location="top",
        shrink=0.85,
        aspect=30,
        pad=-0.15,
    )
    ax3.set_xlabel("$(m_{\mathrm{F606W}} - m_{\mathrm{F814W}})$")
    ax3.set_ylabel("$m_{\mathrm{F606W}}$")
    ax3.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())

    cluster_df["membership"].plot(
        ax=ax4,
        kind="hist",
        density=True,
        bins=30,
        xlabel="Membership",
        ylabel="Density",
        label="",
        align="mid",
    )

    ax4.set_title("Membership distribution")

    p = np.percentile(
        cluster_df["membership"],
        [16, 50, 84],
    )
    ax4.axvline(p[0], color="red", linestyle="--", label=f"{p[0]:.2f}: 16th percentile")
    ax4.axvline(
        p[1],
        color="green",
        linestyle="--",
        label=f"{p[1]:.2f}: 50th percentile",
    )
    ax4.axvline(
        p[2],
        color="blue",
        linestyle="--",
        label=f"{p[2]:.2f}: 84th percentile",
    )
    ax4.legend()

    return fig


def show_cmds_with_voronoi(
    cvor,
    cpoints,
    xc,
    yc,
    xfl,
    yfl,
    padx=0.05,
    pady=0.05,
):

    fig = plt.figure(layout="constrained")
    ax = fig.add_subplot(111)
    ax.set_title("Field stars CMD (Voronoi Grid)")

    voronoi_plot_2d(
        cvor,
        ax=ax,
        show_vertices=False,
        show_points=False,
        line_colors="orange",
        line_aplha=0.5,
        line_width=0.5,
    )

    ax.scatter(xc, yc, s=1, lw=0.5, alpha=0.5, c="red", zorder=2)

    ax.scatter(xfl, yfl, s=1, lw=0.5, alpha=0.5, c="black", zorder=2)

    ax.set(
        xlim=[cpoints[:, 0].min() - padx, cpoints[:, 0].max() + padx],
        ylim=[cpoints[:, 1].min() - pady, cpoints[:, 1].max() + pady],
    )
    ax.invert_yaxis()
    plt.show()
    return fig


def spatial_ecdf(df, racol, deccol, rej_thr):

    # get the center of the dataframe
    center_ra = df[racol].min() + (df[racol].max() - df[racol].min()) / 2
    center_dec = df[deccol].min() + (df[deccol].max() - df[deccol].min()) / 2
    center = SkyCoord(center_ra, center_dec, unit=u.deg)
    distances = center.separation(
        SkyCoord(
            df[racol],
            df[deccol],
            unit="deg",
        )
    ).arcsec

    uniform = np.vstack(
        (
            np.random.uniform(low=df[racol].min(), high=df[racol].max(), size=10000),
            np.random.uniform(low=df[deccol].min(), high=df[deccol].max(), size=10000),
        )
    ).T
    uniform_dist = (
        np.sqrt(
            [
                (uniform[:, 0] - center.ra.deg) ** 2
                + (uniform[:, 1] - center.dec.deg) ** 2
            ]
        )[0]
        * 3600
    )

    fig, ax = plt.subplots(
        layout="tight",
        subplot_kw={"xlabel": "Distance from center (arcsec)", "ylabel": "ECDF"},
    )
    ax.set_title(f"Spatial ECDF at {int(rej_thr*100)}% membership")
    ax.ecdf(
        distances[df["membership"] > rej_thr],
        label="cluster",
    )
    ax.ecdf(
        distances[df["membership"] < rej_thr],
        label="rejection",
    )
    ax.ecdf(uniform_dist, color="red", ls="--", label="uniform decontamination")
    ax.legend()
    return fig


def _plot_grid_with_counts(ax, regions, pts_in_regions):

    max_number = max([len(pts) for _, pts in pts_in_regions])
    cmap = mpl.colormaps["viridis"]
    colors = [cmap(len(pts) / max_number) for _, pts in pts_in_regions]

    for poli, color in zip(regions, colors):
        plotting.plot_polygon(
            poli,
            ax=ax,
            facecolor=color,
            edgecolor="black",
            add_points=False,
            linewidth=0.5,
        )

    cbar = plt.colorbar(
        cm.ScalarMappable(cmap=cmap),
        ax=ax,
        orientation="vertical",
        label="Number of stars",
    )
    cbar.set_ticks(cbar.get_ticks())
    cbar.set_ticklabels([int(el * max_number) for el in cbar.get_ticks()])


def plot_cmd_and_vorgrid(df, col_name, regions, pts_in_regions):
    fig, ax = plt.subplots(layout="constrained")
    ax.set(aspect="auto")
    _plot_grid_with_counts(ax, regions, pts_in_regions)
    plot_cmd(ax, df, col_name, color="red")
    return fig


def plot_spatial_membership(df, racol, deccol, member_threshold):
    fig, [ax_rej, ax_all] = plt.subplots(
        1,
        2,
        layout="tight",
        figsize=(10, 8),
        subplot_kw={"xlabel": "RA", "ylabel": "DEC", "aspect": "equal"},
    )

    rej_df = df[df["membership"] < member_threshold].copy()
    member_df = df[df["membership"] > member_threshold].copy()

    ax_rej.scatter(rej_df[racol], rej_df[deccol], s=1, lw=0.5, alpha=0.5, c="black")
    ax_rej.set_title("Rejected stars")

    ax_all.scatter(
        member_df[racol], member_df[deccol], s=1, lw=0.5, alpha=0.5, c="black"
    )

    ax_all.set_title("Member stars")

    return fig


def plot_cmd_membership_overview(
    df,
    df_field,
    member_threshold,
    mag_cols,
    field_mag_cols,
    inverty=True,
    xlim=None,
    ylim=None,
):
    fig, [[ax_cluster, ax_decon], [ax_rej, ax_field]] = plt.subplots(
        2,
        2,
        layout="constrained",
        figsize=(12, 12),
        subplot_kw={
            "xlabel": "$(m_{\mathrm{F606W}} - m_{\mathrm{F814W}})$",
            "ylabel": "$m_{\mathrm{F606W}}$",
        },
        sharey=True,
        sharex=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    fig.suptitle(f"Decontamination at {member_threshold * 100:.0f}% membership")
    ax_cluster.set_title("Cluster + Field", y=0.9)
    ax_decon.set_title(f"Cluster", y=0.9)
    ax_rej.set_title("Rejection", y=0.9)
    ax_field.set_title("Parallel field", y=0.9)

    plot_cmd(ax_cluster, df, mag_cols, color="black", inverty=inverty)
    plot_cmd(
        ax_decon,
        df[df["membership"] > member_threshold],
        mag_cols,
        color="black",
        inverty=inverty,
    )
    plot_cmd(
        ax_rej,
        df[df["membership"] < member_threshold],
        mag_cols,
        color="black",
        inverty=inverty,
    )
    plot_cmd(ax_field, df_field, field_mag_cols, color="black", inverty=inverty)

    _ = ax_cluster.set(
        ylim=ax_cluster.get_ylim()[::-1],
        xticks=ax_cluster.get_xticks(),
        xticklabels=[f"{el:.1f}" for el in ax_cluster.get_xticks()],
    )

    if xlim is not None:
        ax_cluster.set_xlim(xlim)
    if ylim is not None:
        ax_cluster.set_ylim(ylim)

    return fig


def plot_reddening_map(ra, dec, ra0, dec0, delta_E_B_V):
    fig, ax = plt.subplots(layout="constrained", figsize=(12, 10))
    hex = ax.hexbin(
        ra - ra0,
        dec - dec0,
        C=delta_E_B_V,
        # cmap="magma_r",
        cmap="RdBu_r",
        vmax=delta_E_B_V.max(),
        vmin=-delta_E_B_V.max(),
        gridsize=100,
        reduce_C_function=np.median,
    )
    ax.tick_params(
        axis="x",
        direction="in",
        which="major",
        top="True",
        length=10,
        pad=10,
        # labelsize=12,
    )
    ax.tick_params(
        axis="y",
        direction="in",
        which="major",
        right="True",
        length=10,
        pad=10,
        # labelsize=12,
    )
    ax.tick_params(axis="x", direction="in", which="minor", top="True", length=5)
    ax.tick_params(axis="y", direction="in", which="minor", right="True", length=5)
    ax.xaxis.set_minor_locator(MultipleLocator(0.00500))
    ax.xaxis.set_major_locator(MultipleLocator(0.01000))
    ax.yaxis.set_minor_locator(MultipleLocator(0.00500))
    ax.yaxis.set_major_locator(MultipleLocator(0.01000))
    cbar = fig.colorbar(hex, ax=ax)
    cbar.set_label("$\Delta$ E(B-V) [mag]")
    cbar.ax.tick_params(length=8, pad=10)
    plt.ticklabel_format(style="plain", useOffset=False, axis="x")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    return fig
