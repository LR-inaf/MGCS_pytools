"""Plotting utilities for the statistical membership and MCMC outcomes.

The functions in this module build ready-to-use :class:`matplotlib.figure.Figure` objects
for CMDs, Voronoi grids, membership and decontamination diagnostics, differential
reddening maps, MCMC traces/corner plots and proper-motion vector point diagrams.
"""

from typing import Any

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.typing import NDArray
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely import Polygon, plotting


########################
# PLOT UTILS FOR STATISTICAL MEMBERSHIP, CMD DISPLAY AND REDDENING
def plot_cmd(
    b: NDArray,
    v: NDArray,
    ax: mpl.axes.Axes | None = None,
    ms: int = 1,
    lw: float = 0.3,
    color: Any = "black",
    cmap: str | None = None,
    alpha: float = 0.3,
    inverty: bool = True,
    **kwargs,
) -> mpl.figure.Figure:
    """Plot CMD (Color-Magnitude Diagram) with optional customization.

    The colour ``b - v`` is shown on the x-axis and the magnitude ``b`` on the y-axis.

    Parameters
    ----------
    b : NDArray
        Magnitudes in band B.
    v : NDArray
        Magnitudes in band V.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    ms : int, optional
        Marker size. Default is 1.
    lw : float, optional
        Line width. Default is 0.3.
    color : str or array-like, optional
        Marker color, or values mapped through ``cmap``. Default is ``"black"``.
    cmap : str or None, optional
        Colormap for coloring points. Default is None.
    alpha : float, optional
        Transparency level. Default is 0.3.
    inverty : bool, optional
        Whether to invert the y-axis. Default is True.
    **kwargs
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.set`.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the CMD plot.
    """
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

    fig = ax.get_figure()
    assert isinstance(fig, mpl.figure.Figure)
    return fig


def plot_membership_overview(
    btarget: NDArray,
    vtarget: NDArray,
    bfield: NDArray,
    vfield: NDArray,
    membership: NDArray,
    blabel: str = r"$m_{\mathrm{B}}$",
    vlabel: str = r"$m_{\mathrm{V}}$",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    cmd_cmap: str = "coolwarm",
) -> mpl.figure.Figure:
    r"""Plot an overview of membership in the CMD space.

    The figure has four panels: target CMD, field CMD, target CMD colored by membership
    and the membership distribution with its 16th, 50th and 84th percentiles.

    Parameters
    ----------
    btarget : NDArray
        Magnitudes in band B for the target.
    vtarget : NDArray
        Magnitudes in band V for the target.
    bfield : NDArray
        Magnitudes in band B for the field.
    vfield : NDArray
        Magnitudes in band V for the field.
    membership : NDArray
        Membership values for the target stars.
    blabel : str, optional
        Label for the B band. Default is ``r"$m_{\mathrm{B}}$"``.
    vlabel : str, optional
        Label for the V band. Default is ``r"$m_{\mathrm{V}}$"``.
    xlim : tuple of float, optional
        X-axis limits. Default is None.
    ylim : tuple of float, optional
        Y-axis limits. Default is None.
    cmd_cmap : str, optional
        Colormap for the CMD plots. Default is ``"coolwarm"``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the membership overview plot.
    """
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

    plot_cmd(btarget, vtarget, ax_cmd_tg_memb, color=membership, cmap=cmd_cmap)
    ax_cmd_tg_memb.set(title="Target with membership", xlabel=f"{blabel} - {vlabel}", ylabel=blabel)
    ax_cmd_tg_memb.set(xlim=ax_cmd_tg.get_xlim(), ylim=ax_cmd_tg.get_ylim())

    _ = ax_cmd_tg_memb.figure.colorbar(
        cm.ScalarMappable(cmap=cmd_cmap),
        ax=ax_cmd_tg_memb,
        orientation="horizontal",
        location="top",
        shrink=0.85,
        aspect=30,
        pad=-0.15,
    )

    ax_memb_dist.hist(membership, bins=30, density=True, align="mid")
    ax_memb_dist.set(title="Membership distribution", xlabel="Membership", ylabel="Density")

    p = np.percentile(
        membership,
        [16, 50, 84],
    )
    ax_memb_dist.axvline(p[0], color="red", linestyle="--", label=f"{p[0]:.2f}: 16th percentile")
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
    cvor: Voronoi,
    cpoints: NDArray,
    b: NDArray,
    v: NDArray,
    padx: float = 0.05,
    pady: float = 0.05,
) -> mpl.figure.Figure:
    """Show CMDs with Voronoi tessellation.

    Parameters
    ----------
    cvor : scipy.spatial.Voronoi
        Voronoi diagram to plot.
    cpoints : NDArray
        Points used to build the diagram; they define the plot limits.
    b : NDArray
        B-band magnitudes.
    v : NDArray
        V-band magnitudes.
    padx : float, optional
        Padding for the x-axis. Default is 0.05.
    pady : float, optional
        Padding for the y-axis. Default is 0.05.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the CMD plot with Voronoi tessellation.
    """
    fig, ax = plt.subplots(layout="constrained")

    voronoi_plot_2d(
        cvor,
        ax=ax,
        show_vertices=False,
        show_points=False,
        line_colors="orange",
        line_alpha=0.5,
        line_width=0.5,
    )

    ax.set(
        xlim=[cpoints[:, 0].min() - padx, cpoints[:, 0].max() + padx],
        ylim=[cpoints[:, 1].min() - pady, cpoints[:, 1].max() + pady],
    )
    _ = plot_cmd(b, v, ax=ax)
    return fig


def _plot_grid_with_counts(
    ax: mpl.axes.Axes,
    regions: list[Polygon] | NDArray,
    pts_in_regions: list[tuple],
    cmap: str = "viridis",
):
    """Plot a Voronoi grid with counts of points in each region.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    regions : list of Polygon or NDArray
        The Voronoi regions.
    pts_in_regions : list of tuple
        The ``(region_id, points)`` pairs with the points in each region.
    cmap : str, optional
        The colormap used to encode the number of points. Default is ``"viridis"``.
    """
    max_number = max([len(pts) for _, pts in pts_in_regions]) if pts_in_regions else 1
    cmap_obj = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    counts_by_region = {int(rid): len(pts) for rid, pts in pts_in_regions}

    for rid, poli in enumerate(regions):
        count = counts_by_region.get(rid, 0)
        color = cmap_obj(count / max_number) if max_number > 0 else cmap_obj(0.0)
        plotting.plot_polygon(
            poli,
            ax=ax,
            facecolor=color,
            edgecolor="black",
            add_points=False,
            linewidth=0.5,
        )

    cbar = plt.colorbar(
        cm.ScalarMappable(cmap=cmap_obj),
        ax=ax,
        orientation="vertical",
        label="Number of stars",
    )
    cbar.set_ticks(list(cbar.get_ticks()))
    cbar.set_ticklabels([str(int(el * max_number)) for el in cbar.get_ticks()])


def plot_cmd_and_vorgrid(b, v, regions, pts_in_regions, cmap="viridis"):
    """Plot a CMD on top of a Voronoi grid colored by the number of stars per cell.

    Parameters
    ----------
    b : NDArray
        B-band magnitudes.
    v : NDArray
        V-band magnitudes.
    regions : list of Polygon
        The Voronoi regions.
    pts_in_regions : list of tuple
        The ``(region_id, points)`` pairs with the stars in each region.
    cmap : str, optional
        Colormap used to encode the number of stars per cell. Default is ``"viridis"``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the CMD and the Voronoi grid.
    """
    fig, ax = plt.subplots(layout="constrained")
    ax.set(aspect="auto")
    _plot_grid_with_counts(ax, regions, pts_in_regions, cmap=cmap)
    plot_cmd(b, v, ax=ax, color="red")
    return fig


def plot_spatial_membership(ra, dec, membership, member_threshold):
    """Plot the sky positions of rejected and member stars side by side.

    Parameters
    ----------
    ra : NDArray
        Right ascension of the stars.
    dec : NDArray
        Declination of the stars.
    membership : NDArray
        Membership probability of each star.
    member_threshold : float
        Stars with membership strictly above (below) this value are shown as members
        (rejected).

    Returns
    -------
    matplotlib.figure.Figure
        Figure object with the "Rejected stars" and "Member stars" panels.
    """
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
    blabel=r"$m_{\mathrm{B}}$",
    vlabel=r"$m_{\mathrm{V}}$",
    xlim=None,
    ylim=None,
):
    r"""Plot the CMD decontamination result at a given membership threshold.

    The figure has four panels: the target (cluster + field) CMD, the stars retained as
    members, the rejected stars and the parallel field CMD.

    Parameters
    ----------
    btarget : NDArray
        Magnitudes in band B for the target.
    vtarget : NDArray
        Magnitudes in band V for the target.
    bfield : NDArray
        Magnitudes in band B for the field.
    vfield : NDArray
        Magnitudes in band V for the field.
    membership : NDArray
        Membership probability of the target stars.
    member_threshold : float
        Membership threshold separating members from rejected stars.
    blabel : str, optional
        Label for the B band. Default is ``r"$m_{\mathrm{B}}$"``.
    vlabel : str, optional
        Label for the V band. Default is ``r"$m_{\mathrm{V}}$"``.
    xlim : tuple of float, optional
        X-axis limits. Default is None.
    ylim : tuple of float, optional
        Y-axis limits. Default is None.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the decontamination snapshot.
    """
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
    ax_decon.set_title("Cluster", y=0.9)
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
    """Plot the differential reddening map as a hexbin of the median delta E(B-V).

    Parameters
    ----------
    ra : NDArray
        Right ascension of the stars [deg].
    dec : NDArray
        Declination of the stars [deg].
    ra0 : float
        Reference right ascension subtracted from ``ra`` [deg].
    dec0 : float
        Reference declination subtracted from ``dec`` [deg].
    delta_E_B_V : NDArray
        Differential reddening of each star [mag].

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the reddening map.
    """
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
    cbar.set_label(r"$\Delta$ E(B-V) [mag]")
    cbar.ax.tick_params(length=8, pad=10)
    plt.ticklabel_format(style="plain", useOffset=False, axis="x")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    return fig


########################
# PLOT UTILS FOR MCMC
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
    """Plot the trace plots associated to an MCMC run.

    Parameters
    ----------
    chains : NDArray
        Chains from MCMC, with shape (ndraws, nwalkers, dim).
    naxes : iterable of int, optional
        Number of rows and columns to plot, as ``(nrows, ncols)``. Must contain enough
        axes for all the parameters to plot. With ``dim`` the number of plotted
        parameters, the default is ``(dim, 1)`` if ``dim <= 6``, otherwise
        ``(6, dim // 6 + 1)``.
    useparams : array-like of int, optional
        Indexes corresponding to the parameters to plot. Default is all.
    delchains : array-like of int, optional
        Indexes of the walkers to discard. They are drawn only if ``showdel`` is True.
        Default is None.
    usechains : array-like of int, optional
        Indexes of the walkers to use. Default is all walkers (when ``delchains`` is
        also None).
    showdel : bool, optional
        If True, also draw the discarded walkers with swapped colors. Default is False.
    burnin : int, optional
        Number of burn-in draws, drawn with ``colorburnin``. Default is 0.
    labels : array-like of str, optional
        Labels corresponding to each parameter. Default is ``"param <i>"``.
    fnt : dict, optional
        Font keyword arguments for the axis labels. Default is ``{"fontsize": 16}``.
    figsize : tuple of float, optional
        Size of the figure. Default is (6, 6).
    title : str, optional
        Title of the figure. Default is ``"Trace plots"``.
    color : str, optional
        Color used for the chains (all chains have the same color). Default is ``"black"``.
    colorburnin : str, optional
        Color used for the burn-in. Default is ``"salmon"``.
    tickoff : bool, optional
        If True, delete labels and ticks on overlapping panels. Default is False.
    kwargs : dict, optional
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.plot`.
        Default is ``{"lw": 0.5, "ls": "-"}``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the trace plots.

    Raises
    ------
    ValueError
        If ``usechains``, ``naxes`` or ``labels`` are invalid, or if the requested axes
        are not enough for the parameters to plot.
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
    if np.iterable(useparams) or np.isscalar(useparams):
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
    """Plot the corner plot of an MCMC chain.

    Parameters
    ----------
    chain : NDArray
        Samples with shape (nsamples, ndim).
    labels : list of str, optional
        Labels of the parameters. Default is ``"C<i>"``.
    quantiles : list of float, optional
        Quantiles shown on the 1D histograms. Default is [0.16, 0.5, 0.84].
    bins : int, optional
        Number of bins of the histograms. Default is 20.
    truths : array-like, optional
        Reference values to overplot. Default is None.
    title : str, optional
        Title of the figure. Default is ``"Corner plots"``.
    **kwargs
        Additional keyword arguments passed to :func:`corner.corner`.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the corner plot.
    """
    labels = labels if labels is not None else [f"C{i + 1}" for i in range(chain.shape[-1])]
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


def univar_norm_pdf(
    x: NDArray,
    mu: NDArray | float,
    sigma: NDArray | float,
    w: NDArray | float = 1.0,
) -> NDArray:
    """Weighted PDF of the univariate normal distribution.

    Parameters
    ----------
    x : NDArray
        Values at which to compute the PDF, shape (Nx,).
    mu : NDArray or float
        Mean of the distribution (single or multiple), shape (Nparams,).
    sigma : NDArray or float
        Standard deviation (single or multiple), shape (Nparams,).
    w : NDArray or float, optional
        Weight of each distribution, shape (Nparams,). Default is 1.0.

    Returns
    -------
    NDArray
        Weighted PDF, shape (Nx,) if ``mu``, ``sigma`` and ``w`` are all scalars (size 1),
        otherwise (Nparams, Nx).

    Raises
    ------
    ValueError
        If ``mu``, ``sigma`` and ``w`` do not have the same length (or are not scalars).
    """
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    w = np.atleast_1d(w)

    # Allinea le dimensioni
    max_len = max(mu.size, sigma.size, w.size)

    if mu.size == 1:
        mu = np.full(max_len, mu.item(), dtype=float)
    if sigma.size == 1:
        sigma = np.full(max_len, sigma.item(), dtype=float)
    if w.size == 1:
        w = np.full(max_len, w.item(), dtype=float)

    if not (mu.size == sigma.size == w.size):
        raise ValueError("mu and sigma must have the same length or be scalars.")

    # Broadcasting on x: (Nparams, Nx)
    diff = x[None, :] - mu[:, None]
    pdf = (
        w[:, None]
        * (1.0 / (sigma[:, None] * np.sqrt(2.0 * np.pi)))
        * np.exp(-0.5 * (diff / sigma[:, None]) ** 2)
    )

    # If all inputs were scalars, return a 1D array
    if max_len == 1:
        return pdf.ravel()
    return pdf


def _generate_gaussian_distribution_from_mcmc_chain(
    x, ch, mupar, sigmapar, w=None, sigma_error=None
):
    """Evaluate the normal PDF of a component for every sample of an MCMC chain.

    The standard deviation of each sample is broadened by the median measurement error.

    Parameters
    ----------
    x : NDArray
        Values at which to compute the PDF.
    ch : NDArray
        MCMC samples, with shape (nsamples, ndim).
    mupar : int
        Column of ``ch`` holding the mean.
    sigmapar : int
        Column of ``ch`` holding the intrinsic standard deviation.
    w : NDArray or float, optional
        Weight of the component for each sample. Default is 1.
    sigma_error : NDArray, optional
        Measurement errors, whose median is added in quadrature to the standard deviation.
        Default is None (no broadening).

    Returns
    -------
    NDArray
        PDF values with shape (nsamples, Nx).
    """
    if w is None:
        w = np.ones(ch.shape[0])

    if sigma_error is None:
        sigma_error = np.zeros(ch.shape[0])

    sigma = np.sqrt(ch[:, sigmapar] ** 2 + np.median(sigma_error) ** 2)

    return univar_norm_pdf(
        x,
        ch[:, mupar],
        sigma,
        w=w,
    )


def plot_vpd_and_mdist_component(
    mux,
    muy,
    ch,
    membership=None,
    ncomp=1,
    par_per_comp=5,
    component_labels=None,
    w=None,
    muxpar=0,
    muypar=1,
    sigxpar=2,
    sigypar=3,
    xerror=None,
    yerror=None,
    vpd_xlabel=None,
    vpd_ylabel=None,
    figsize_inches=8,
    figsize_wadjust=0.0,
    figsize_hadjust=-0.6,
    xlim=None,
    ylim=None,
    membership_cmap="coolwarm",
    component_colors=None,
    hist_scale="log",
):
    """Plot a vector point diagram with the marginal distributions of the mixture components.

    The central panel shows the points (optionally colored by membership and with error
    bars); the top and right panels show the histograms of the data and the 5th-95th
    percentile bands of each model component and of their sum, computed from the MCMC chain.

    Parameters
    ----------
    mux : NDArray
        Values along the x-axis (e.g. proper motion in RA).
    muy : NDArray
        Values along the y-axis (e.g. proper motion in Dec).
    ch : NDArray
        MCMC samples, with shape (nsamples, ndim).
    membership : NDArray, optional
        Membership probability of each point, used to color the points. Default is None.
    ncomp : int, optional
        Number of mixture components. Default is 1.
    par_per_comp : int, optional
        Number of parameters per component. Default is 5.
    component_labels : list of str, optional
        Labels of the components (length ``ncomp``); ``"SUM"`` is appended.
        Default is ``"C<i>"``.
    w : NDArray, optional
        Samples of the first ``ncomp - 1`` mixture weights, with shape (ncomp - 1, nsamples).
        The weight of the last component is ``1 - w.sum(axis=0)``. Default is None.
    muxpar : int, optional
        Index of the mean along x of the first component in ``ch``. Default is 0.
    muypar : int, optional
        Index of the mean along y of the first component in ``ch``. Default is 1.
    sigxpar : int, optional
        Index of the standard deviation along x of the first component in ``ch``.
        Default is 2.
    sigypar : int, optional
        Index of the standard deviation along y of the first component in ``ch``.
        Default is 3.
    xerror : NDArray, optional
        Measurement errors along x, drawn as error bars and added (median) in quadrature
        to the model widths. Default is None.
    yerror : NDArray, optional
        Measurement errors along y, drawn as error bars and added (median) in quadrature
        to the model widths. Default is None.
    vpd_xlabel : str, optional
        Label of the x-axis of the central panel. Default is None.
    vpd_ylabel : str, optional
        Label of the y-axis of the central panel. Default is None.
    figsize_inches : float, optional
        Base figure size in inches. Default is 8.
    figsize_wadjust : float, optional
        Offset added to the figure width. Default is 0.0.
    figsize_hadjust : float, optional
        Offset added to the figure height. Default is -0.6.
    xlim : tuple of float, optional
        X-axis limits of the central panel. Default is None.
    ylim : tuple of float, optional
        Y-axis limits of the central panel. Default is None.
    membership_cmap : str, optional
        Colormap used for the membership. Default is ``"coolwarm"``.
    component_colors : list, optional
        Colors of the components (length ``ncomp``). Default is the ``"Set1"`` colormap.
    hist_scale : str, optional
        Scale of the marginal PDF axes. Default is ``"log"``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the vector point diagram and marginal distributions.

    Raises
    ------
    AssertionError
        If the length of ``component_colors`` or ``component_labels`` is not ``ncomp``.
    """
    fig = plt.figure(figsize=(figsize_inches + figsize_wadjust, figsize_inches + figsize_hadjust))
    gs = GridSpec(4, 4, wspace=0, hspace=0)

    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_hist_x = fig.add_subplot(gs[0, 0:3])  # sharex=ax_scatter)
    ax_hist_y = fig.add_subplot(gs[1:4, 3])  # sharey=ax_scatter)
    axs_dict = {"vpd": ax_scatter, "histx": ax_hist_x, "histy": ax_hist_y}

    capsize = 1.5
    ms = 15
    elw = 1.0

    # VPD
    if xerror is not None and yerror is None:
        axs_dict["vpd"].errorbar(
            mux,
            muy,
            xerr=xerror,
            fmt="ok",
            capsize=capsize,
            elinewidth=elw,
            ms=0.0,
            alpha=0.1,
        )
    elif yerror is not None and xerror is None:
        axs_dict["vpd"].errorbar(
            mux,
            muy,
            yerr=yerror,
            fmt="ok",
            capsize=capsize,
            elinewidth=elw,
            ms=0.0,
            alpha=0.1,
        )
    elif xerror is not None and yerror is not None:
        axs_dict["vpd"].errorbar(
            mux,
            muy,
            xerr=xerror,
            yerr=yerror,
            fmt="ok",
            capsize=capsize,
            elinewidth=elw,
            ms=0.0,
            alpha=0.1,
        )

    if membership is None:
        sc = axs_dict["vpd"].scatter(mux, muy, s=ms, c="black", zorder=10)
    else:
        sc = axs_dict["vpd"].scatter(
            mux,
            muy,
            s=ms,
            lw=0.5,
            edgecolor="black",
            c=membership,
            cmap=membership_cmap,
            vmin=0,
            vmax=1,
            zorder=10,
        )

        cb_ax = inset_axes(
            axs_dict["vpd"], width="35%", height="3%", loc="upper right", borderpad=1
        )
        cbar = plt.colorbar(sc, cax=cb_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)

    if xlim is not None:
        axs_dict["vpd"].set(xlim=xlim)

    if ylim is not None:
        axs_dict["vpd"].set(ylim=ylim)

    if component_colors is None:
        cmap = mpl.colormaps["Set1"]
        component_colors = [cmap(i) for i in range(ncomp)]
    else:
        assert len(component_colors) == ncomp, "Component colors length mismatch."

    # Mdist
    mdist_x_sum = None
    mdist_y_sum = None
    for i in range(ncomp):
        # set parameters for each component
        muxp = muxpar + i * par_per_comp
        muyp = muypar + i * par_per_comp
        sigxp = sigxpar + i * par_per_comp
        sigyp = sigypar + i * par_per_comp

        if i == ncomp - 1 and w is not None:
            ww = 1 - np.sum(w, axis=0)
        elif w is not None:
            ww = w[i]
        else:
            ww = None

        mdist_x = _generate_gaussian_distribution_from_mcmc_chain(
            x=np.sort(mux),
            ch=ch,
            mupar=muxp,
            sigmapar=sigxp,
            w=ww,
            sigma_error=xerror,
        )
        if mdist_x_sum is None:
            mdist_x_sum = mdist_x
        else:
            mdist_x_sum += mdist_x

        mdist_y = _generate_gaussian_distribution_from_mcmc_chain(
            x=np.sort(muy),
            ch=ch,
            mupar=muyp,
            sigmapar=sigyp,
            w=ww,
            sigma_error=yerror,
        )
        if mdist_y_sum is None:
            mdist_y_sum = mdist_y
        else:
            mdist_y_sum += mdist_y

        # x dist fillbetween
        low = np.percentile(mdist_x, 5, axis=0)
        high = np.percentile(mdist_x, 95, axis=0)
        axs_dict["histx"].fill_between(
            np.sort(mux),
            low,
            high,
            color=component_colors[i],
        )

        # y dist fillbetween
        low = np.percentile(mdist_y, 5, axis=0)
        high = np.percentile(mdist_y, 95, axis=0)
        axs_dict["histy"].fill_betweenx(
            np.sort(muy),
            low,
            high,
            color=component_colors[i],
        )

    # sum fill between
    low = np.percentile(mdist_x_sum, 5, axis=0)
    high = np.percentile(mdist_x_sum, 95, axis=0)
    axs_dict["histx"].fill_between(
        np.sort(mux),
        low,
        high,
        color="gray",
    )

    low = np.percentile(mdist_y_sum, 5, axis=0)
    high = np.percentile(mdist_y_sum, 95, axis=0)
    axs_dict["histy"].fill_betweenx(
        np.sort(muy),
        low,
        high,
        color="gray",
    )

    if component_labels is None:
        component_labels = [f"C{i + 1}" for i in range(ncomp)] + ["SUM"]
    else:
        assert len(component_labels) == ncomp, "Component labels length mismatch."
        component_labels = component_labels + ["SUM"]

    # Create legend
    handles = [
        mpl.lines.Line2D([0], [0], color=component_colors[i], lw=2.0) for i in range(ncomp)
    ] + [mpl.lines.Line2D([0], [0], color="gray", lw=2.0)]
    axs_dict["histx"].legend(handles, component_labels, loc="upper left", ncols=2, frameon=False)

    # Histograms
    bins = 100
    _ = axs_dict["histx"].hist(
        mux,
        bins=bins,
        density=True,
        alpha=0.3,
        color="gray",
        histtype="stepfilled",
        linewidth=1.5,
        edgecolor="black",
    )
    _ = axs_dict["histy"].hist(
        muy,
        bins=bins,
        density=True,
        alpha=0.3,
        color="gray",
        histtype="stepfilled",
        orientation="horizontal",
        linewidth=1.5,
        edgecolor="black",
    )

    # axes adjustments
    axs_dict["vpd"].set(aspect="equal", xlabel=vpd_xlabel, ylabel=vpd_ylabel)
    vpd_xlim = axs_dict["vpd"].get_xlim()
    vpd_ylim = axs_dict["vpd"].get_ylim()

    axs_dict["histx"].set(
        ylabel="PDF",
        xticks=[],
        xlim=vpd_xlim,
        yscale=hist_scale,
        ylim=(1e-4, axs_dict["histx"].get_ylim()[1]),
    )
    axs_dict["histy"].set(
        xlabel="PDF",
        yticks=[],
        ylim=vpd_ylim,
        xscale=hist_scale,
        xlim=(1e-4, axs_dict["histy"].get_xlim()[1]),
    )
    axs_dict["histy"].tick_params(which="both", direction="in")
    axs_dict["histx"].tick_params(which="both", direction="in")
    axs_dict["vpd"].tick_params(which="both", direction="in")
    return fig


# Alias for compatibility
plot_differential_reddening_map = plot_reddening_map
