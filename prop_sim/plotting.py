"""可視化ヘルパ (matplotlib は遅延インポート)."""

from __future__ import annotations

import numpy as np

from .frames import COMPONENT_NAMES

__all__ = [
    "plot_blade_geometry",
    "plot_static_sweep",
    "plot_six_components",
    "plot_spectrum",
    "plot_disk_loading",
    "plot_polar",
]

_FORCE_UNITS = ["N", "N", "N", "N m", "N m", "N m"]


def _plt():
    import matplotlib

    if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
        try:  # pragma: no cover - 環境依存
            matplotlib.use("Agg")
        except Exception:
            pass
    import matplotlib.pyplot as plt

    return plt


def plot_blade_geometry(geometry, ax=None):
    """コード長・ねじり角の半径方向分布."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    x = np.linspace(geometry.hub_radius_ratio, 1.0, 200)
    ax[0].plot(x, geometry.chord(x) * 1e3)
    ax[0].set_xlabel("r/R")
    ax[0].set_ylabel("chord [mm]")
    ax[0].grid(alpha=0.3)
    ax[1].plot(x, np.rad2deg(geometry.twist(x)))
    ax[1].set_xlabel("r/R")
    ax[1].set_ylabel("twist [deg]")
    ax[1].grid(alpha=0.3)
    ax[0].figure.suptitle(geometry.name)
    return ax


def plot_static_sweep(result, x: str = "rpm", ys=("Fz_true", "Mz_true"), ax=None):
    """静的掃引結果の折れ線."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for y in ys:
        ax.plot(result[x], result[y], "o-", label=y)
    ax.set_xlabel(x)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_six_components(time, wrench, measured=None, time_measured=None, axes=None):
    """6 分力の時系列を 2x3 で表示する."""
    plt = _plt()
    if axes is None:
        _, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True,
                               sharex=True)
    axes = np.asarray(axes).ravel()
    for i, (name, unit) in enumerate(zip(COMPONENT_NAMES, _FORCE_UNITS)):
        if measured is not None:
            axes[i].plot(
                time_measured if time_measured is not None else time,
                measured[:, i], lw=0.7, color="tab:orange", label="measured",
            )
        axes[i].plot(time, wrench[:, i], lw=1.0, color="tab:blue", label="true")
        axes[i].set_ylabel(f"{name} [{unit}]")
        axes[i].grid(alpha=0.3)
        if i >= 3:
            axes[i].set_xlabel("time [s]")
    axes[0].legend(fontsize=8)
    return axes


def plot_spectrum(freq, amp, ax=None, label=None, marks=()):
    """スペクトル (対数軸) と特徴周波数の縦線."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.semilogy(freq, np.maximum(amp, 1e-12), lw=0.8, label=label)
    for f, name in marks:
        ax.axvline(f, color="gray", ls="--", lw=0.8)
        ax.text(f, ax.get_ylim()[1], f" {name}", va="top", fontsize=8, color="gray")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("amplitude")
    ax.grid(alpha=0.3, which="both")
    if label:
        ax.legend()
    return ax


def plot_disk_loading(sections, ax=None, quantity: str = "dfz_dr"):
    """ロータディスク上の荷重分布 (極座標)."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"},
                             figsize=(5, 4.5), constrained_layout=True)
    z = getattr(sections, quantity)
    psi = sections.psi
    r = sections.r_R
    psi_c = np.concatenate([psi, psi[:1] + 2 * np.pi])
    z_c = np.vstack([z, z[:1]])
    m = ax.pcolormesh(psi_c, r, z_c.T, shading="gouraud", cmap="viridis")
    ax.figure.colorbar(m, ax=ax, label=quantity)
    ax.set_title(f"{quantity} (psi=0 は +x 方向)")
    return ax


def plot_polar(airfoil, alpha_deg=None, ax=None):
    """翼型の 360 deg ポーラ."""
    plt = _plt()
    if alpha_deg is None:
        alpha_deg = np.linspace(-180, 180, 721)
    cl, cd, _ = airfoil.coefficients(np.deg2rad(alpha_deg))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(alpha_deg, cl, label="Cl")
    ax.plot(alpha_deg, cd, label="Cd")
    ax.set_xlabel("alpha [deg]")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(getattr(airfoil, "name", "airfoil"))
    return ax
