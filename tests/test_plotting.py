import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from mgcs_pytools.utils.plotting import (
    plot_cmd,
    plot_corner,
    plot_membership_overview,
    plot_vpd_and_mdist_component,
    traceplots,
    univar_norm_pdf,
)


def test_plot_cmd():
    np.random.seed(42)
    b = np.random.normal(20, 1, 100)
    v = np.random.normal(19, 1, 100)

    fig = plot_cmd(b, v)
    assert isinstance(fig, mpl.figure.Figure)
    plt.close(fig)

    fig, ax = plt.subplots()
    ret_fig = plot_cmd(b, v, ax=ax)
    assert ret_fig == fig
    plt.close(fig)


def test_plot_membership_overview():
    np.random.seed(42)
    btarget = np.random.uniform(15, 22, 100)
    vtarget = np.random.uniform(14, 21, 100)
    bfield = np.random.uniform(15, 22, 80)
    vfield = np.random.uniform(14, 21, 80)
    membership = np.random.uniform(0, 1, 100)

    fig = plot_membership_overview(btarget, vtarget, bfield, vfield, membership)
    assert isinstance(fig, mpl.figure.Figure)
    plt.close(fig)


def test_traceplots():
    np.random.seed(42)
    # chain of shape (nsteps, nwalkers, ndim)
    chain = np.random.randn(50, 10, 4)

    fig = traceplots(chain)
    assert isinstance(fig, mpl.figure.Figure)
    plt.close(fig)


def test_plot_corner():
    np.random.seed(42)
    # flat chain (nsamples, ndim)
    chain = np.random.randn(200, 3)

    fig = plot_corner(chain, labels=["a", "b", "c"])
    assert isinstance(fig, mpl.figure.Figure)
    plt.close(fig)


def test_univar_norm_pdf():
    x = np.linspace(-3, 3, 100)
    pdf = univar_norm_pdf(x, 0.0, 1.0)
    assert len(pdf) == 100
    assert np.isclose(pdf[50], 1.0 / np.sqrt(2 * np.pi), atol=0.05)


def test_plot_vpd_and_mdist_component():
    np.random.seed(42)
    # fake chain: (nsamples, 11) for 2 components (5 params each + 1 weight)
    # [mux1, muy1, s1, s2, rho1, mux2, muy2, s3, s4, rho2, w1]
    chain = np.zeros((50, 11))
    chain[:, 0] = np.random.normal(0, 0.1, 50)
    chain[:, 1] = np.random.normal(0, 0.1, 50)
    chain[:, 2] = np.random.uniform(0.5, 1.0, 50)
    chain[:, 3] = np.random.uniform(0.5, 1.0, 50)
    chain[:, 4] = np.random.uniform(-0.1, 0.1, 50)
    chain[:, 5] = np.random.normal(2, 0.1, 50)
    chain[:, 6] = np.random.normal(2, 0.1, 50)
    chain[:, 7] = np.random.uniform(1.0, 1.5, 50)
    chain[:, 8] = np.random.uniform(1.0, 1.5, 50)
    chain[:, 9] = np.random.uniform(-0.1, 0.1, 50)
    chain[:, 10] = np.random.uniform(0.4, 0.6, 50)

    # fake data: (N, 4) -> [pm_ra, pm_dec, epm_ra, epm_dec]
    data = np.random.randn(100, 4)
    data[:, 2:] = 0.05

    fig = plot_vpd_and_mdist_component(data[:, 0], data[:, 1], chain, ncomp=2)
    assert isinstance(fig, mpl.figure.Figure)
    plt.close(fig)
