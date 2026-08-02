"""後処理: 次数分析・スペクトル・回転平均・係数換算.

実際の 6 分力計測でも同じ処理を行うので, 計測データ処理コードの
検証用としても使える.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "harmonic_amplitudes",
    "revolution_average",
    "spectrum",
    "order_spectrum",
    "propeller_coefficients",
    "HarmonicFit",
]


@dataclass
class HarmonicFit:
    """次数分析の結果."""

    orders: np.ndarray        # (n_order,) 解析した次数 (1 = 1/rev)
    mean: float               # 直流成分
    amplitude: np.ndarray     # (n_order,) 振幅
    phase_deg: np.ndarray     # (n_order,) 位相 [deg] : y = A cos(n*psi - phase)
    residual_rms: float       # 当てはめ残差 RMS

    def as_dict(self) -> dict[str, float]:
        out = {"mean": self.mean, "residual_rms": self.residual_rms}
        for o, a, p in zip(self.orders, self.amplitude, self.phase_deg):
            out[f"amp_{o:g}P"] = float(a)
            out[f"phase_{o:g}P_deg"] = float(p)
        return out


def harmonic_amplitudes(
    signal_values: np.ndarray,
    azimuth_rad: np.ndarray,
    orders: np.ndarray | list[float] = (1, 2, 3, 4),
) -> HarmonicFit:
    """方位角基準の次数分析 (回転同期成分の最小二乗抽出).

    回転数が変化していても方位角を使うため次数成分を正しく取り出せる
    (オーダートラッキング).

    Parameters
    ----------
    signal_values:
        時系列データ (1 次元).
    azimuth_rad:
        各サンプルにおけるロータ方位角 [rad] (連続値, ラップしていてもよい).
    orders:
        抽出する次数.
    """
    y = np.asarray(signal_values, dtype=float).ravel()
    psi = np.unwrap(np.asarray(azimuth_rad, dtype=float).ravel())
    orders = np.asarray(orders, dtype=float)

    cols = [np.ones_like(y)]
    for o in orders:
        cols.append(np.cos(o * psi))
        cols.append(np.sin(o * psi))
    a = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ coef

    c = coef[1::2]
    s = coef[2::2]
    return HarmonicFit(
        orders=orders,
        mean=float(coef[0]),
        amplitude=np.hypot(c, s),
        phase_deg=np.rad2deg(np.arctan2(s, c)),
        residual_rms=float(np.sqrt(np.mean(resid**2))),
    )


def revolution_average(
    signal_values: np.ndarray, azimuth_rad: np.ndarray, n_bins: int = 72
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """方位角ビンごとの平均 (回転同期平均).

    Returns
    -------
    (方位角 [deg], 平均値, 標準偏差)
    """
    y = np.asarray(signal_values, dtype=float).ravel()
    psi = np.mod(np.asarray(azimuth_rad, dtype=float).ravel(), 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
    idx = np.clip(np.digitize(psi, edges) - 1, 0, n_bins - 1)
    mean = np.full(n_bins, np.nan)
    std = np.full(n_bins, np.nan)
    for k in range(n_bins):
        m = idx == k
        if np.any(m):
            mean[k] = y[m].mean()
            std[k] = y[m].std()
    centers = np.rad2deg(0.5 * (edges[:-1] + edges[1:]))
    return centers, mean, std


def spectrum(
    time: np.ndarray, signal_values: np.ndarray, window: str = "hann"
) -> tuple[np.ndarray, np.ndarray]:
    """片側振幅スペクトル (振幅は正弦波の振幅と一致するよう正規化).

    Returns
    -------
    (周波数 [Hz], 振幅)
    """
    t = np.asarray(time, dtype=float).ravel()
    y = np.asarray(signal_values, dtype=float).ravel()
    n = y.size
    if n < 4:
        return np.zeros(0), np.zeros(0)
    dt = float(np.mean(np.diff(t)))
    if window == "hann":
        w = np.hanning(n)
    elif window in (None, "none", "boxcar"):
        w = np.ones(n)
    else:  # pragma: no cover
        from scipy import signal as sig

        w = sig.get_window(window, n)
    yw = (y - y.mean()) * w
    spec = np.fft.rfft(yw)
    freq = np.fft.rfftfreq(n, dt)
    amp = 2.0 * np.abs(spec) / np.sum(w)
    return freq, amp


def order_spectrum(
    azimuth_rad: np.ndarray,
    signal_values: np.ndarray,
    n_order: int = 32,
    samples_per_rev: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """方位角で等間隔リサンプリングした次数スペクトル.

    回転数が変動する試験でも次数成分がにじまない.
    """
    psi = np.unwrap(np.asarray(azimuth_rad, dtype=float).ravel())
    y = np.asarray(signal_values, dtype=float).ravel()
    if psi[-1] < psi[0]:
        psi, y = psi[::-1], y[::-1]
    n_rev = int((psi[-1] - psi[0]) / (2.0 * np.pi))
    if n_rev < 1:
        return np.zeros(0), np.zeros(0)
    grid = psi[0] + np.arange(n_rev * samples_per_rev) * (
        2.0 * np.pi / samples_per_rev
    )
    yr = np.interp(grid, psi, y)
    w = np.hanning(yr.size)
    spec = np.fft.rfft((yr - yr.mean()) * w)
    amp = 2.0 * np.abs(spec) / np.sum(w)
    orders = np.fft.rfftfreq(yr.size, 1.0 / samples_per_rev)
    keep = orders <= n_order
    return orders[keep], amp[keep]


def propeller_coefficients(
    thrust: np.ndarray,
    torque: np.ndarray,
    rpm: np.ndarray,
    diameter: float,
    v_axial: np.ndarray | float = 0.0,
    density: float = 1.225,
) -> dict[str, np.ndarray]:
    """計測値からプロペラ係数を計算する (実験の後処理と同じ手順)."""
    n = np.asarray(rpm, dtype=float) / 60.0
    t = np.asarray(thrust, dtype=float)
    q = np.asarray(torque, dtype=float)
    v = np.asarray(v_axial, dtype=float)
    d = float(diameter)
    with np.errstate(divide="ignore", invalid="ignore"):
        ct = t / (density * n**2 * d**4)
        cq = q / (density * n**2 * d**5)
        cp = 2.0 * np.pi * cq
        j = v / (n * d)
        eta = np.where(cp != 0.0, j * ct / cp, 0.0)
    return {"J": j, "Ct": ct, "Cq": cq, "Cp": cp, "eta": eta}
