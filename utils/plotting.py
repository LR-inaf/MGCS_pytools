import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator

from scipy.spatial import voronoi_plot_2d

import numpy as np

from shapely import plotting

from corner import corner


def plot_cmd(
    b,
    v,
    ax=None,
    ms=1,
    lw=0.3,
    color="black",
    cmap=None,
    alpha=0.3,
    inverty=True,
    **kwargs,
) -> mpl.figure.Figure:

    if ax is None:
        _, ax = plt.subplots(layout="constrained")

    ax.scatter(
        b - v,
        b,
        s=ms,
        lw=lw,
        c=color,
        cmap=cmap,
        alpha=alpha,
    )

    if inverty:
        ax.invert_yaxis()

    ax.set(**kwargs)

    return ax.get_figure()


def plot_membership_overview(
    btarget,
    vtarget,
    bfield,
    vfield,
    membership,
    blabel="$m_{\mathrm{B}}$",
    vlabel="$m_{\mathrm{V}}$",
    xlim=None,
    ylim=None,
):
    fig, [[ax_cmd_tg, ax_cmd_fl], [ax_cmd_tg_memb, ax_memb_dist]] = plt.subplots(
        2, 2, layout="constrained", figsize=(12, 12)
    )

    plot_cmd(btarget, vtarget, ax_cmd_tg)
    ax_cmd_tg.set(title="Target", xlabel=f"{blabel} - {vlabel}", ylabel=blabel)

    if xlim is not None:
        ax_cmd_tg.set_xlim(xlim)
    if ylim is not None:
        ax_cmd_tg.set_ylim(ylim)

    plot_cmd(bfield, vfield, ax_cmd_fl)
    ax_cmd_fl.set(title="Field", xlabel=f"{blabel} - {vlabel}", ylabel=blabel)
    ax_cmd_fl.set(xlim=ax_cmd_tg.get_xlim(), ylim=ax_cmd_tg.get_ylim())

    plot_cmd(btarget, vtarget, ax_cmd_tg_memb, color=membership, cmap="jet_r")
    ax_cmd_tg_memb.set(
        title="Target with membership", xlabel=f"{blabel} - {vlabel}", ylabel=blabel
    )
    ax_cmd_tg_memb.set(xlim=ax_cmd_tg.get_xlim(), ylim=ax_cmd_tg.get_ylim())

    _ = ax_cmd_tg_memb.figure.colorbar(
        cm.ScalarMappable(cmap="jet_r"),
        ax=ax_cmd_tg_memb,
        orientation="horizontal",
        location="top",
        shrink=0.85,
        aspect=30,
        pad=-0.15,
    )

    ax_memb_dist.hist(membership, bins=30, density=True, align="mid")
    ax_memb_dist.set(
        title="Membership distribution", xlabel="Membership", ylabel="Density"
    )

    p = np.percentile(
        membership,
        [16, 50, 84],
    )
    ax_memb_dist.axvline(
        p[0], color="red", linestyle="--", label=f"{p[0]:.2f}: 16th percentile"
    )
    ax_memb_dist.axvline(
        p[1],
        color="green",
        linestyle="--",
        label=f"{p[1]:.2f}: 50th percentile",
    )
    ax_memb_dist.axvline(
        p[2],
        color="blue",
        linestyle="--",
        label=f"{p[2]:.2f}: 84th percentile",
    )
    ax_memb_dist.legend()

    return fig


def show_cmds_with_voronoi(
    cvor,
    cpoints,
    b,
    v,
    padx=0.05,
    pady=0.05,
):

    fig, ax = plt.subplots(layout="constrained")

    voronoi_plot_2d(
        cvor,
        ax=ax,
        show_vertices=False,
        show_points=False,
        line_colors="orange",
        line_aplha=0.5,
        line_width=0.5,
    )

    ax.set(
        xlim=[cpoints[:, 0].min() - padx, cpoints[:, 0].max() + padx],
        ylim=[cpoints[:, 1].min() - pady, cpoints[:, 1].max() + pady],
    )
    _ = plot_cmd(b, v, ax=ax)
    return fig


def _plot_grid_with_counts(ax, regions, pts_in_regions, cmap="viridis"):

    max_number = max([len(pts) for _, pts in pts_in_regions])
    cmap = mpl.colormaps[cmap]
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


def plot_cmd_and_vorgrid(b, v, regions, pts_in_regions, cmap="viridis"):
    fig, ax = plt.subplots(layout="constrained")
    ax.set(aspect="auto")
    _plot_grid_with_counts(ax, regions, pts_in_regions, cmap=cmap)
    plot_cmd(ax, b, v, ax=ax, color="red")
    return fig


def plot_spatial_membership(ra, dec, membership, member_threshold):
    fig, [ax_rej, ax_memb] = plt.subplots(
        1,
        2,
        layout="tight",
        figsize=(10, 8),
        subplot_kw={"xlabel": "RA", "ylabel": "DEC", "aspect": "equal"},
    )

    ra_rej = ra[membership < member_threshold]
    dec_rej = dec[membership < member_threshold]
    ra_member = ra[membership > member_threshold]
    dec_member = dec[membership > member_threshold]

    ax_rej.scatter(ra_rej, dec_rej, s=1, lw=0.5, alpha=0.5, c="black")
    ax_rej.set_title("Rejected stars")

    ax_memb.scatter(ra_member, dec_member, s=1, lw=0.5, alpha=0.5, c="black")
    ax_memb.set_title("Member stars")

    return fig


def plot_decontamination_snapshot(
    btarget,
    vtarget,
    bfield,
    vfield,
    membership,
    member_threshold,
    blabel="$m_{\mathrm{B}}$",
    vlabel="$m_{\mathrm{V}}$",
    xlim=None,
    ylim=None,
):
    fig, [[ax_cluster, ax_decon], [ax_rej, ax_field]] = plt.subplots(
        2,
        2,
        layout="constrained",
        figsize=(12, 12),
        subplot_kw={
            "xlabel": f"{blabel} - {vlabel}",
            "ylabel": blabel,
        },
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    fig.suptitle(f"Decontamination at {member_threshold * 100:.0f}% membership")
    ax_cluster.set_title("Cluster + Field", y=0.9)
    ax_decon.set_title(f"Cluster", y=0.9)
    ax_rej.set_title("Rejection", y=0.9)
    ax_field.set_title("Parallel field", y=0.9)

    plot_cmd(btarget, vtarget, ax_cluster)
    plot_cmd(
        btarget[membership > member_threshold],
        vtarget[membership > member_threshold],
        ax_decon,
    )
    plot_cmd(
        btarget[membership < member_threshold],
        vtarget[membership < member_threshold],
        ax_rej,
    )
    plot_cmd(bfield, vfield, ax_field)

    if xlim is not None:
        ax_cluster.set_xlim(xlim)
    if ylim is not None:
        ax_cluster.set_ylim(ylim)

    ax_decon.set(xlim=ax_cluster.get_xlim(), ylim=ax_cluster.get_ylim())
    ax_rej.set(xlim=ax_cluster.get_xlim(), ylim=ax_cluster.get_ylim())
    ax_field.set(xlim=ax_cluster.get_xlim(), ylim=ax_cluster.get_ylim())

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


def traceplots(
    chains,
    naxes=None,
    useparams=None,
    delchains=None,
    usechains=None,
    showdel=None,
    burnin=None,
    labels=None,
    fnt=None,
    figsize=None,
    title="Trace plots",
    color=None,
    colorburnin=None,
    tickoff=None,
    kwargs=None,
):
    """
    The method plots the traceplots associated to an MCMC run

    Parameters
    -------------->>
    chains		:	Chains from MCMC (shape ndraws x nwalkers x dim)
    naxes		:	Number of rows and columns to plot. Must be a 2d iterable.
                                    Raise an error if the number of axes is larger than the number
                                    of parameters to plot. Default is 6 rows and dim/6
                                    columns
    useparams	:	Indexes corresponing to the parameters to plot. Default all
    delchains	:	Chains to discard. Default None
    Usechains	:	Chains to use. Default all. If usechains is given, delchains
                                    is ignored
    burnin		:	Burn-in. Default None
    labels		:	Labels corresponing to each parameter
    fnt 		: 	Label size. Default {labesize:16}
    figsize		:	Size of figure. Default (6,6)
    title       :	Title of the figure. Default "Trace plots"
    color		:	Color used for the chains (all chains have the same color). Defult black
    colorburnin	:	Color used for the burn-in. Defult salmon
    tickoff		:	if True, delete labels and ticks on overlapping panels
    kwargs		: 	additional dictionary to be used in plot

    """

    ndraws, nwalkers, dim = chains.shape

    # 	#	#	sanity check on parameters

    # 	usechains and delchains work together
    if usechains is None and delchains is None:
        usechains = np.linspace(0, nwalkers - 1, nwalkers, dtype=int)

    if usechains is not None:
        if np.isscalar(usechains):
            usechains = np.asarray([usechains])
        elif np.iterable(usechains):
            usechains = np.asarray(usechains)
        else:
            raise ValueError("usechains: invalid parameter")

    # 	show delated chains
    if showdel is None:
        showdel = False

    # 	parameters to show
    if useparams is None:
        useparams = np.linspace(0, dim - 1, dim, dtype=int)
    if np.iterable(useparams) or np.scalar(useparams):
        useparams = np.asarray(useparams)
    dim = useparams.shape[0]

    # 	number of rows and columns in trace plots
    if naxes is None:
        naxes = [dim, 1] if dim <= 6 else [6, int(dim / 6) + 1]

    if not np.iterable(naxes):
        raise ValueError("naxes: invalid number of rows and cols")
    else:
        nrows, ncols = naxes[0], naxes[1]
        if nrows * ncols < dim:
            raise ValueError("Parameters required do not match number of axes")

    # 	check labels
    if labels is None:
        labels = np.asarray(["param " + str(i) for i in range(dim)])
    if np.iterable(labels) or isinstance(labels, str):
        labels = np.asarray(labels)
        if labels.shape[0] != dim:
            raise ValueError("labels: invalid number of labels")

    # 	remaining parameters
    if burnin is None:
        burnin = 0
    if colorburnin is None:
        colorburnin = "salmon"
    if color is None:
        color = "black"
    if tickoff is None:
        tickoff = False
    if kwargs is None:
        kwargs = {"lw": 0.5, "ls": "-"}
    if fnt is None:
        fnt = {"fontsize": 16}
    if figsize is None:
        figsize = (6, 6)

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    for prm, aa in zip(useparams, ax.T.reshape(-1)):
        aa.plot(
            np.linspace(burnin, ndraws - 1, ndraws - burnin),
            chains[burnin:, usechains, prm],
            color=color,
            **kwargs,
        )
        aa.plot(
            np.linspace(0, burnin - 1, burnin),
            chains[:burnin, usechains, prm],
            color=colorburnin,
            **kwargs,
        )

    # 	show delated chain
    if showdel and delchains is not None:
        for prm, aa in zip(useparams, ax.T.reshape(-1)):
            aa.plot(
                np.linspace(burnin, ndraws - 1, ndraws - burnin),
                chains[burnin:, delchains, prm],
                color=colorburnin,
                **kwargs,
            )
            aa.plot(
                np.linspace(0, burnin - 1, burnin),
                chains[:burnin, delchains, prm],
                color=color,
                **kwargs,
            )

    # 	labels
    for label, aa in zip(labels, ax.T.reshape(nrows * ncols)):
        aa.set_ylabel(label, **fnt)
        aa.set_xlabel("draws", **fnt)
        aa.set_xlim(0, ndraws)

    # 	delete overlapping labels
    if tickoff:
        for aa in ax.reshape(-1)[: (dim - naxes[1])]:
            aa.set_xlabel("")
            aa.set_xticklabels([])

    # 	delete un-used axes
    if dim < nrows * ncols:
        for aa in ax.T.reshape(nrows * ncols)[dim : nrows * ncols]:
            fig.delaxes(aa)

    fig.suptitle(title, fontsize=2.5 * max(fig.get_size_inches()))
    return fig


def plot_corner(
    chain,
    labels=None,
    quantiles=[0.16, 0.5, 0.84],
    bins=20,
    truths=None,
    title="Corner plots",
    **kwargs,
):

    labels = (
        labels if labels is not None else [f"C{i+1}" for i in range(chain.shape[-1])]
    )
    fig = corner(
        chain,
        labels=labels,
        quantiles=quantiles,
        bins=bins,
        verbose=False,
        truths=truths,
        show_titles=True,
        **kwargs,
    )
    fig.suptitle(title, fontsize=2.5 * max(fig.get_size_inches()))
    return fig
