#!/usr/bin/env python3
"""Hubble's law, from Hubble's 1929 nebulae to modern Type Ia supernovae.

Three things are done here, each with proper uncertainties:

1. Hubble's own 24 galaxies (Table 1 of his 1929 paper) are fitted with a straight
   line, and the result is checked with a bootstrap.
2. Modern Type Ia supernova distances (the Pantheon+ / SH0ES compilation, 1,701
   light curves) give a measurement of H0 in the local "Hubble flow".
3. A small simulation of a uniformly expanding universe shows why *every* observer
   sees the same Hubble law, so nobody sits at the "centre".

Run `python hubbles_law.py` to redo all of it. See README.md for the physics.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

# --------------------------------------------------------------------------- constants

C_KM_S = 299_792.458  # speed of light, km/s (exact)
MPC_IN_KM = 3.0856775814913673e19  # kilometres in one megaparsec
GYR_IN_S = 3.15576e16  # seconds in a gigayear (Julian years)

# Deceleration parameter used for the low-redshift expansion of d_L (Riess et al. 2016).
Q0_LOW_Z = -0.55

# --------------------------------------------------------------------------- Hubble 1929

# Table 1 of Hubble (1929), PNAS 15, 168: object, distance r in Mpc (the paper prints it
# in units of 10^6 parsecs), radial velocity v in km/s. Negative v means approaching.
HUBBLE_1929 = [
    ("S. Mag.", 0.032, +170),
    ("L. Mag.", 0.034, +290),
    ("NGC 6822", 0.214, -130),
    ("NGC 598", 0.263, -70),
    ("NGC 221", 0.275, -185),
    ("NGC 224", 0.275, -220),
    ("NGC 5457", 0.45, +200),
    ("NGC 4736", 0.5, +290),
    ("NGC 5194", 0.5, +270),
    ("NGC 4449", 0.63, +200),
    ("NGC 4214", 0.8, +300),
    ("NGC 3031", 0.9, -30),
    ("NGC 3627", 0.9, +650),
    ("NGC 4826", 0.9, +150),
    ("NGC 5236", 0.9, +500),
    ("NGC 1068", 1.0, +920),
    ("NGC 5055", 1.1, +450),
    ("NGC 7331", 1.1, +500),
    ("NGC 4258", 1.4, +500),
    ("NGC 4151", 1.7, +960),
    ("NGC 4382", 2.0, +500),
    ("NGC 4472", 2.0, +850),
    ("NGC 4486", 2.0, +800),
    ("NGC 4649", 2.0, +1090),
]

# Hubble's own result for K (km/s per Mpc), solving the same 24 galaxies for the Sun's
# motion as well: K = 465 +/- 50. From the 9 groups: 513 +/- 60.
HUBBLE_1929_PUBLISHED = (465.0, 50.0)

# Published H0 values used for comparison (km/s/Mpc, 1-sigma, statistical + systematic).
LITERATURE_H0 = [
    ("Hubble 1929", 465.0, 50.0),
    ("HST Key Project (Freedman+ 2001)", 72.0, 8.0),
    ("TRGB (Freedman 2019)", 69.8, 1.9),
    ("SH0ES (Riess+ 2022)", 73.04, 1.04),
    ("Planck 2018 (Planck Collab. 2020)", 67.4, 0.5),
]

# --------------------------------------------------------------------------- Pantheon+

PANTHEON_URL = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
    "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
)
PANTHEON_CACHE_NAME = "pantheon_plus_shoes.dat"
PANTHEON_COLUMNS = [
    "CID",  # supernova name
    "zHD",  # Hubble-diagram redshift (CMB frame, peculiar-velocity corrected)
    "MU_SH0ES",  # distance modulus, absolute scale set by the SH0ES Cepheid calibration
    "MU_SH0ES_ERR_DIAG",  # its 1-sigma uncertainty (diagonal of the covariance matrix)
    "IS_CALIBRATOR",  # 1 = host galaxy also has a Cepheid distance
    "USED_IN_SH0ES_HF",  # 1 = in the SH0ES Hubble-flow sample (0.023 < z < 0.15)
]

# --------------------------------------------------------------------------- plot style

# Reference palette (light surface). Slots are assigned by role, never by rank.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "font.size": 10,
        }
    )


# --------------------------------------------------------------------------- basic maths


def hubble_time_gyr(h0: float) -> float:
    """1/H0 in gigayears, for H0 in km/s/Mpc."""
    return MPC_IN_KM / h0 / GYR_IN_S


def distance_modulus(d_mpc):
    """mu = 5 log10(d / 10 pc), with d in Mpc."""
    return 5.0 * np.log10(d_mpc) + 25.0


def distance_from_modulus(mu):
    """Inverse of distance_modulus: distance in Mpc."""
    return 10.0 ** ((np.asarray(mu, dtype=float) - 25.0) / 5.0)


# --------------------------------------------------------------------------- linear fits


@dataclass
class LineFit:
    """Result of fitting v = H0 d (+ intercept) to velocity-distance data."""

    h0: float  # slope, km/s/Mpc
    h0_err: float  # 1-sigma standard error on the slope
    intercept: float | None  # km/s, None for a fit through the origin
    intercept_err: float | None
    scatter: float  # standard deviation of the residuals, km/s (n - parameters dof)
    n: int


def fit_through_origin(d, v) -> LineFit:
    """Least-squares fit of v = H0 d, i.e. Hubble's law with no offset.

    H0 = sum(d v) / sum(d^2). The residual variance uses n - 1 degrees of freedom
    because one parameter was fitted.
    """
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(d)
    h0 = float((d * v).sum() / (d * d).sum())
    resid = v - h0 * d
    s2 = float((resid**2).sum() / (n - 1))
    err = float(np.sqrt(s2 / (d * d).sum()))
    return LineFit(h0, err, None, None, float(np.sqrt(s2)), n)


def fit_with_intercept(d, v) -> LineFit:
    """Least-squares fit of v = H0 d + v0. The offset soaks up a net drift, such as the
    Sun's own motion, that would otherwise bias the slope."""
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(d)
    X = np.column_stack([d, np.ones(n)])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    resid = v - X @ beta
    s2 = float((resid**2).sum() / (n - 2))
    cov = s2 * np.linalg.inv(X.T @ X)
    return LineFit(
        float(beta[0]),
        float(np.sqrt(cov[0, 0])),
        float(beta[1]),
        float(np.sqrt(cov[1, 1])),
        float(np.sqrt(s2)),
        n,
    )


def bootstrap_slope(d, v, through_origin: bool, n_boot: int = 20000, seed: int = 0):
    """Resample the galaxies with replacement and refit, n_boot times.

    Returns the array of fitted slopes. Its spread is an error bar that does not
    assume the residuals are Gaussian.
    """
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    D, V = d[idx], v[idx]
    if through_origin:
        return (D * V).sum(axis=1) / (D * D).sum(axis=1)
    Dc = D - D.mean(axis=1, keepdims=True)
    Vc = V - V.mean(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = (Dc * Vc).sum(axis=1) / (Dc * Dc).sum(axis=1)
    return slopes[np.isfinite(slopes)]


def hubble_1929_arrays():
    d = np.array([row[1] for row in HUBBLE_1929], dtype=float)
    v = np.array([row[2] for row in HUBBLE_1929], dtype=float)
    return d, v


# --------------------------------------------------------------------------- supernova fit


def fit_hubble_flow(z, mu, err, q0: float = Q0_LOW_Z) -> dict:
    """H0 from supernovae in the smooth Hubble flow (0.023 < z < 0.15).

    To second order in z the luminosity distance is
        d_L = (c z / H0) [1 + (1 - q0) z / 2],
    so mu = 25 + 5 log10(D / H0) with D = c z [1 + (1 - q0) z / 2] in km/s. That is linear
    in log10(H0), and the weighted least-squares answer is a weighted mean:
        log10 H0 = sum(w x) / sum(w),   x = log10(D) - (mu - 25) / 5,   w = 1 / sigma_x^2.
    """
    z, mu, err = (np.asarray(a, dtype=float) for a in (z, mu, err))
    x = np.log10(C_KM_S * z * (1 + 0.5 * (1 - q0) * z)) - (mu - 25.0) / 5.0
    w = 25.0 / err**2  # sigma_x = sigma_mu / 5
    log_h0 = float((w * x).sum() / w.sum())
    h0 = 10.0**log_h0
    h0_err = float(h0 * np.log(10.0) / np.sqrt(w.sum()))
    chi2 = float((w * (x - log_h0) ** 2).sum())
    return {"h0": float(h0), "h0_err": h0_err, "chi2": chi2, "n": int(len(z)), "q0": q0}


def bootstrap_hubble_flow(z, mu, err, n_boot: int = 20000, seed: int = 0, q0: float = Q0_LOW_Z):
    """Bootstrap distribution of H0 for the Hubble-flow sample (resampling supernovae)."""
    z, mu, err = (np.asarray(a, dtype=float) for a in (z, mu, err))
    x = np.log10(C_KM_S * z * (1 + 0.5 * (1 - q0) * z)) - (mu - 25.0) / 5.0
    w = 25.0 / err**2
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(z), size=(n_boot, len(z)))
    return 10.0 ** ((w[idx] * x[idx]).sum(axis=1) / w[idx].sum(axis=1))


# --------------------------------------------------------------------------- data loading


def download(url: str, dest: Path, timeout: float = 60.0, opener=urllib.request.urlopen) -> None:
    """Fetch url into dest, writing to a temporary file first so a failed download never
    leaves a half-written cache behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with opener(url, timeout=timeout) as response:
        payload = response.read()
    tmp.write_bytes(payload)
    tmp.replace(dest)


def read_pantheon(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=r"\s+", usecols=PANTHEON_COLUMNS)
    except ValueError as exc:
        raise ValueError(f"{path} does not look like the Pantheon+SH0ES table: {exc}") from exc
    return df


def load_pantheon(
    data_dir: str | Path = "data",
    path: str | Path | None = None,
    refresh: bool = False,
    opener=urllib.request.urlopen,
) -> pd.DataFrame:
    """Load the Pantheon+SH0ES distance table, downloading and caching it on first use."""
    if path is not None:
        return read_pantheon(Path(path))
    cache = Path(data_dir) / PANTHEON_CACHE_NAME
    if refresh or not cache.exists():
        print(f"Downloading Pantheon+SH0ES table to {cache} ...")
        download(PANTHEON_URL, cache, opener=opener)
    return read_pantheon(cache)


def hubble_flow_sample(df: pd.DataFrame) -> pd.DataFrame:
    """The SH0ES Hubble-flow sample: 0.023 < z < 0.15, no Cepheid calibrators."""
    return df[df["USED_IN_SH0ES_HF"] == 1].sort_values("zHD")


# --------------------------------------------------------------------------- expansion simulation

H0_SIM = 70.0  # km/s/Mpc, the input Hubble constant of the simulated universe
KMS_TO_MPC_PER_GYR = GYR_IN_S / MPC_IN_KM  # 1 km/s expressed in Mpc per Gyr


def make_universe(n: int = 300, radius_mpc: float = 100.0, sigma_pec: float = 160.0, seed: int = 1):
    """Galaxies scattered uniformly over a disc (positions in Mpc), each with a random
    peculiar velocity (km/s per component). sigma_pec = 160 gives a scatter about the
    Hubble line of roughly 230 km/s, similar to Hubble's 1929 galaxies."""
    rng = np.random.default_rng(seed)
    r = radius_mpc * np.sqrt(rng.random(n))
    theta = 2 * np.pi * rng.random(n)
    pos = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    pec = rng.normal(0.0, sigma_pec, size=(n, 2))
    return pos, pec


def scale_factor(t_gyr: float, h0: float = H0_SIM) -> float:
    """a(t) = 1 + H0 t: constant-speed (coasting) expansion, a purely kinematic toy."""
    return 1.0 + h0 * KMS_TO_MPC_PER_GYR * t_gyr


def universe_at(pos, pec, t_gyr: float, h0: float = H0_SIM):
    """Physical positions (Mpc) and velocities (km/s) of every galaxy at time t.

    Each galaxy is carried outward with the Hubble flow, x = a(t) x0, and drifts with its
    own peculiar velocity. H(t) = H0 / a(t), so v_flow = H(t) x.
    """
    a = scale_factor(t_gyr, h0)
    x = a * pos + pec * KMS_TO_MPC_PER_GYR * t_gyr
    v = (h0 / a) * (a * pos) + pec
    return x, v


def observer_view(x, v, k: int):
    """What the observer in galaxy k measures: distance and radial velocity of all others."""
    rel = np.delete(x - x[k], k, axis=0)
    vrel = np.delete(v - v[k], k, axis=0)
    dist = np.linalg.norm(rel, axis=1)
    v_r = (vrel * rel).sum(axis=1) / dist
    return dist, v_r


# --------------------------------------------------------------------------- figures


def plot_hubble_1929(fit0: LineFit, fit1: LineFit, out: Path) -> None:
    d, v = hubble_1929_arrays()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    xx = np.linspace(0, 2.2, 50)
    ax.axhline(0, color=AXIS, lw=1)
    ax.plot(xx, 70 * xx, color=MUTED, lw=1.4, label="Modern value, H₀ ≈ 70")
    ax.plot(xx, fit0.h0 * xx, color=BLUE, label=f"Through origin: H₀ = {fit0.h0:.0f} ± {fit0.h0_err:.0f}")
    ax.plot(
        xx,
        fit1.h0 * xx + fit1.intercept,
        color=ORANGE,
        label=f"With intercept: H₀ = {fit1.h0:.0f} ± {fit1.h0_err:.0f}",
    )
    ax.scatter(d, v, s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3, label="Hubble's 24 galaxies")
    for name, dist, vel in HUBBLE_1929:
        if name in ("NGC 224", "NGC 4649", "NGC 3031"):
            tag = {"NGC 224": "M31 Andromeda", "NGC 4649": "NGC 4649", "NGC 3031": "M81"}[name]
            ax.annotate(tag, (dist, vel), xytext=(6, -12 if vel < 0 else 6), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 2.2)
    ax.set_ylim(-400, 1300)
    ax.set_xlabel("Distance (Mpc), as estimated by Hubble in 1929")
    ax.set_ylabel("Radial velocity (km/s)")
    ax.set_title("Hubble's 1929 galaxies: velocity rises with distance, but the slope is ~6× too steep")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_1929_analysis(fit0: LineFit, boot0, fit1: LineFit, boot1, out: Path) -> None:
    d, v = hubble_1929_arrays()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    resid = v - fit0.h0 * d
    a.axhline(0, color=AXIS, lw=1)
    a.scatter(d, resid, s=30, color=BLUE, edgecolor=SURFACE, linewidth=1.1, zorder=3)
    a.axhspan(-fit0.scatter, fit0.scatter, color=BLUE, alpha=0.08, lw=0)
    a.text(2.18, -fit0.scatter - 45, f"±1σ = {fit0.scatter:.0f} km/s", ha="right", va="center", color=INK2, fontsize=9)
    a.set_xlim(0, 2.2)
    a.set_xlabel("Distance (Mpc)")
    a.set_ylabel("Residual from best fit (km/s)")
    a.set_title("Scatter about the line: peculiar velocities")
    lo, hi = np.percentile(np.concatenate([boot0, boot1]), [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    for data, colour, label, fit in (
        (boot0, BLUE, "Through origin", fit0),
        (boot1, ORANGE, "With intercept", fit1),
    ):
        p16, p50, p84 = np.percentile(data, [16, 50, 84])
        b.hist(data, bins=bins, color=colour, alpha=0.55, lw=0, label=f"{label}: {p50:.0f} (+{p84 - p50:.0f} / −{p50 - p16:.0f})")
    b.axvline(70, color=MUTED, lw=1.4)
    b.text(72, b.get_ylim()[1] * 0.92, "H₀ ≈ 70", color=INK2, fontsize=9)
    b.set_xlabel("H₀ from resampled data (km/s/Mpc)")
    b.set_ylabel("Number of resamples")
    b.set_title("Bootstrap: H₀ from resampled galaxies")
    b.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_hubble_flow(hf: pd.DataFrame, fit: dict, stability: pd.DataFrame, out: Path) -> None:
    z = hf["zHD"].to_numpy()
    d_l = distance_from_modulus(hf["MU_SH0ES"].to_numpy())
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.7), gridspec_kw={"width_ratios": [1.15, 1]})
    a.scatter(d_l, C_KM_S * z, s=12, color=BLUE, alpha=0.55, lw=0, label=f"{len(z)} supernovae, 0.023 < z < 0.15")
    zz = np.linspace(0, 0.16, 100)
    a.plot(
        C_KM_S * zz / fit["h0"] * (1 + 0.5 * (1 - fit["q0"]) * zz),
        C_KM_S * zz,
        color=ORANGE,
        label=f"H₀ = {fit['h0']:.1f} ± {fit['h0_err']:.1f} km/s/Mpc",
    )
    a.set_xlabel("Luminosity distance (Mpc)")
    a.set_ylabel("cz (km/s)")
    a.set_title("The Hubble flow: a clean straight line")
    a.legend(loc="upper left")
    b.axhline(67.4, color=MUTED, lw=1.3)
    b.text(0.152, 67.1, "Planck 2018 (67.4)", ha="right", va="top", color=INK2, fontsize=9)
    b.fill_between(stability["z_max"], stability["h0"] - stability["h0_err"], stability["h0"] + stability["h0_err"], color=BLUE, alpha=0.18, lw=0)
    b.plot(stability["z_max"], stability["h0"], color=BLUE, marker="o", ms=4, label="H₀ using supernovae with z below z_max")
    b.set_ylim(64, 80)
    b.set_xlabel("Redshift cut z_max")
    b.set_ylabel("H₀ (km/s/Mpc)")
    b.set_title("Stable against the redshift cut")
    b.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_h0_summary(this_work: list[tuple[str, float, float]], out: Path) -> None:
    rows = [(n, h, e, "lit") for n, h, e in LITERATURE_H0] + [(n, h, e, "this") for n, h, e in this_work]
    rows.sort(key=lambda r: r[1])  # smallest H0 at the bottom, so 1929 is at the top
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    for i, (name, h0, err, kind) in enumerate(rows):
        colour = ORANGE if kind == "this" else BLUE
        ax.errorbar(h0, i, xerr=err, fmt="o", color=colour, ms=6, lw=1.6, capsize=0, zorder=3)
        ax.annotate(f"{h0:.1f} ± {err:.1f}", (h0, i), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xticks([50, 70, 100, 200, 400, 600])
    ax.set_xticklabels(["50", "70", "100", "200", "400", "600"])
    ax.set_xlim(45, 700)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("H₀ (km/s/Mpc), log scale")
    ax.set_title("A century of H₀: from ~500 down to ~70")
    ax.plot([], [], "o", color=BLUE, label="Published")
    ax.plot([], [], "o", color=ORANGE, label="This project")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _expansion_axes(fig):
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.32, wspace=0.22)
    return [(fig.add_subplot(gs[0, i]), fig.add_subplot(gs[1, i])) for i in range(2)]


def _draw_expansion(axes_pairs, pos, pec, observers, t_gyr: float, h0: float, lim: float = 300.0, vmax: float = 14000.0):
    """Draw both observers' sky maps (top) and Hubble diagrams (bottom) at time t."""
    x, v = universe_at(pos, pec, t_gyr, h0)
    a_t = scale_factor(t_gyr, h0)
    for (ax_map, ax_hd), k, colour, name in zip(axes_pairs, observers, (BLUE, ORANGE), ("Observer A", "Observer B")):
        ax_map.clear()
        ax_hd.clear()
        others = np.delete(x - x[k], k, axis=0)
        ax_map.set_facecolor("#f4f3ef")
        ax_map.scatter(others[:, 0], others[:, 1], s=7, color=MUTED, alpha=0.8, lw=0)
        ax_map.scatter([0], [0], s=70, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax_map.plot([-lim * 0.92, -lim * 0.92 + 100], [-lim * 0.88] * 2, color=INK2, lw=1.4)
        ax_map.text(-lim * 0.92 + 50, -lim * 0.82, "100 Mpc", ha="center", color=INK2, fontsize=8.5)
        ax_map.set_xlim(-lim, lim)
        ax_map.set_ylim(-lim, lim)
        ax_map.set_aspect("equal")
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.grid(False)
        for spine in ax_map.spines.values():
            spine.set_visible(False)
        ax_map.set_title(f"{name}'s view of the sky")
        dist, v_r = observer_view(x, v, k)
        fit = fit_through_origin(dist, v_r)
        ax_hd.scatter(dist, v_r, s=8, color=colour, alpha=0.6, lw=0)
        ax_hd.plot([0, lim], [0, fit.h0 * lim], color=INK, lw=1.3)
        ax_hd.set_xlim(0, lim)
        ax_hd.set_ylim(-1500, vmax)
        ax_hd.set_xlabel("Distance (Mpc)")
        ax_hd.set_ylabel("Radial velocity (km/s)")
        ax_hd.text(0.04, 0.9, f"fitted H = {fit.h0:.1f} km/s/Mpc", transform=ax_hd.transAxes, fontsize=9.5, color=INK)
    return a_t


def plot_expansion_static(out: Path, seed: int = 1) -> dict:
    pos, pec = make_universe(seed=seed)
    observers = choose_observers(pos)
    fig = plt.figure(figsize=(9.6, 7.4))
    axes_pairs = _expansion_axes(fig)
    _draw_expansion(axes_pairs, pos, pec, observers, 0.0, H0_SIM, lim=190.0, vmax=14000.0)
    fig.suptitle("Two observers, one Hubble law: every galaxy sees the others recede in proportion to distance", x=0.06, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    x, v = universe_at(pos, pec, 0.0)
    return {f"observer_{n}": fit_through_origin(*observer_view(x, v, k)).h0 for n, k in zip("AB", observers)}


def choose_observers(pos):
    """Observer A: the galaxy nearest the middle. Observer B: one about 70% of the way out."""
    r = np.linalg.norm(pos, axis=1)
    a = int(np.argmin(r))
    b = int(np.argmin(np.abs(r - 0.7 * r.max())))
    return a, b


def animate_expansion(out: Path, seed: int = 1, frames: int = 54, t_max_gyr: float = 9.0) -> None:
    pos, pec = make_universe(seed=seed)
    observers = choose_observers(pos)
    fig = plt.figure(figsize=(8.4, 6.6))
    axes_pairs = _expansion_axes(fig)
    title = fig.suptitle("", x=0.06, ha="left", fontsize=11, fontweight="bold")

    def update(i):
        t = t_max_gyr * i / (frames - 1)
        a_t = _draw_expansion(axes_pairs, pos, pec, observers, t, H0_SIM)
        title.set_text(f"t = {t:.1f} Gyr    size of universe × {a_t:.2f}    H(t) = {H0_SIM / a_t:.1f} km/s/Mpc")

    anim = FuncAnimation(fig, update, frames=frames, interval=90)
    anim.save(out, writer=PillowWriter(fps=11), dpi=72)
    plt.close(fig)


# --------------------------------------------------------------------------- driver


def stability_table(hf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for z_max in np.arange(0.04, 0.1501, 0.01):
        s = hf[hf["zHD"] <= z_max]
        f = fit_hubble_flow(s["zHD"], s["MU_SH0ES"], s["MU_SH0ES_ERR_DIAG"])
        rows.append({"z_max": round(float(z_max), 2), "h0": f["h0"], "h0_err": f["h0_err"], "n": f["n"]})
    return pd.DataFrame(rows)


def report_1929(fit0, fit1, boot0, boot1) -> dict:
    p0 = np.percentile(boot0, [16, 50, 84])
    p1 = np.percentile(boot1, [16, 50, 84])
    print("\nHubble 1929 (24 galaxies)")
    print(f"  through origin : H0 = {fit0.h0:6.1f} ± {fit0.h0_err:4.1f} km/s/Mpc   scatter {fit0.scatter:.0f} km/s")
    print(f"  bootstrap      : {p0[1]:6.1f} (+{p0[2] - p0[1]:.1f} / -{p0[1] - p0[0]:.1f})")
    print(f"  with intercept : H0 = {fit1.h0:6.1f} ± {fit1.h0_err:4.1f} km/s/Mpc   intercept {fit1.intercept:.0f} ± {fit1.intercept_err:.0f} km/s")
    print(f"  bootstrap      : {p1[1]:6.1f} (+{p1[2] - p1[1]:.1f} / -{p1[1] - p1[0]:.1f})")
    print(f"  Hubble's own K = {HUBBLE_1929_PUBLISHED[0]:.0f} ± {HUBBLE_1929_PUBLISHED[1]:.0f}  (solves for the Sun's motion too)")
    print(f"  1/H0 = {hubble_time_gyr(fit0.h0):.2f} Gyr (through origin), {hubble_time_gyr(fit1.h0):.2f} Gyr (with intercept); Earth is ~4.5 Gyr old")
    return {
        "through_origin": {**asdict(fit0), "bootstrap_16_50_84": p0.tolist(), "hubble_time_gyr": hubble_time_gyr(fit0.h0)},
        "with_intercept": {**asdict(fit1), "bootstrap_16_50_84": p1.tolist(), "hubble_time_gyr": hubble_time_gyr(fit1.h0)},
    }


def report_supernovae(df: pd.DataFrame, out_dir: Path) -> tuple[dict, list]:
    hf = hubble_flow_sample(df)
    flow = fit_hubble_flow(hf["zHD"], hf["MU_SH0ES"], hf["MU_SH0ES_ERR_DIAG"])
    boot = bootstrap_hubble_flow(hf["zHD"], hf["MU_SH0ES"], hf["MU_SH0ES_ERR_DIAG"])
    flow["bootstrap_16_50_84"] = np.percentile(boot, [16, 50, 84]).tolist()
    stab = stability_table(hf)

    print(f"\nPantheon+SH0ES: {len(df)} light curves, {df['CID'].nunique()} unique supernovae")
    print(f"Hubble-flow sample ({flow['n']} SNe, 0.023 < z < 0.15)")
    print(f"  H0 = {flow['h0']:.2f} ± {flow['h0_err']:.2f} km/s/Mpc (formal); bootstrap {flow['bootstrap_16_50_84'][1]:.2f} "
          f"(+{flow['bootstrap_16_50_84'][2] - flow['bootstrap_16_50_84'][1]:.2f} / -{flow['bootstrap_16_50_84'][1] - flow['bootstrap_16_50_84'][0]:.2f})")
    print(f"  chi2 = {flow['chi2']:.0f} for {flow['n'] - 1} dof  (< 1 per dof: the quoted errors are conservative)")
    print(f"  1/H0 = {hubble_time_gyr(flow['h0']):.2f} Gyr")
    plot_hubble_flow(hf, flow, stab, out_dir / "fig3_hubble_flow.png")
    return {"hubble_flow": flow, "n_light_curves": int(len(df)), "n_unique": int(df["CID"].nunique()), "stability": stab.to_dict("records")}, [
        ("This project: supernovae", flow["h0"], flow["h0_err"])
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", help="use a local Pantheon+SH0ES table instead of downloading it")
    parser.add_argument("--data-dir", default="data", help="where the downloaded table is cached (default: data)")
    parser.add_argument("--refresh", action="store_true", help="download the supernova table again")
    parser.add_argument("--out-dir", default="figures", help="where figures are written (default: figures)")
    parser.add_argument("--no-supernovae", action="store_true", help="skip the supernova part (needs no internet)")
    parser.add_argument("--no-animation", action="store_true", help="skip the expansion GIF")
    parser.add_argument("--results", default="results.json", help="where the numbers are saved (default: results.json)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for the bootstrap (default: 0)")
    args = parser.parse_args(argv)

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d, v = hubble_1929_arrays()
    fit0, fit1 = fit_through_origin(d, v), fit_with_intercept(d, v)
    boot0 = bootstrap_slope(d, v, True, seed=args.seed)
    boot1 = bootstrap_slope(d, v, False, seed=args.seed)
    results = {"hubble_1929": report_1929(fit0, fit1, boot0, boot1)}
    plot_hubble_1929(fit0, fit1, out_dir / "fig1_hubble_1929.png")
    plot_1929_analysis(fit0, boot0, fit1, boot1, out_dir / "fig2_hubble_1929_analysis.png")
    this_work = [("This project: Hubble's 1929 data", fit0.h0, fit0.h0_err)]

    if not args.no_supernovae:
        try:
            df = load_pantheon(args.data_dir, args.data, args.refresh)
        except Exception as exc:  # network down, bad file, ...
            print(f"\nCould not load the supernova data: {exc}", file=sys.stderr)
            print("Re-run with --no-supernovae to skip that part, or --data FILE to use your own copy.", file=sys.stderr)
            return 1
        sn_results, sn_rows = report_supernovae(df, out_dir)
        results["supernovae"] = sn_results
        this_work += sn_rows
        plot_h0_summary(this_work, out_dir / "fig4_h0_over_time.png")

    sim = plot_expansion_static(out_dir / "fig5_two_observers.png")
    print("\nSimulated expanding universe (input H0 = 70): fitted H0 seen by observer A = "
          f"{sim['observer_A']:.1f}, by observer B = {sim['observer_B']:.1f}")
    results["expansion_simulation"] = sim
    if not args.no_animation:
        animate_expansion(out_dir / "expansion.gif")

    Path(args.results).write_text(json.dumps(results, indent=2, default=float) + "\n")
    print(f"\nFigures written to {out_dir}/, numbers to {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
