"""誘導速度場 (インフロー) の表現と斜め流入モデル."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["InflowField", "skew_coefficients"]


def skew_coefficients(
    mu: float, lam: float, model: str = "drees"
) -> tuple[float, float]:
    """後流スキュー角に基づく線形インフロー分布の係数 (kx, ky).

    誘導速度は

        v(r, psi) = v0(r) * [1 - kx * (r/R) * cos(psi - psi_e)
                                + ky * (r/R) * sin(psi - psi_e)]

    と表される. ``psi_e`` は面内速度の方位で, ディスクの前縁側
    (機体が進む側) で流入が小さく, 後縁側で大きくなる.

    Parameters
    ----------
    mu:
        前進比 V_edge / (Omega R).
    lam:
        全流入比 (V_axial + v_i) / (Omega R).
    model:
        ``"drees"`` (既定) / ``"pitt"`` / ``"coleman"`` / ``"none"``.
    """
    if model == "none":
        return 0.0, 0.0
    mu = abs(float(mu))
    lam = abs(float(lam))
    if mu < 1e-8:
        return 0.0, 0.0
    chi = np.arctan2(mu, max(lam, 1e-8))  # 後流スキュー角 [rad]

    if model == "coleman":
        kx = float(np.tan(chi / 2.0))
        ky = 0.0
    elif model == "pitt":
        kx = float(15.0 * np.pi / 23.0 * np.tan(chi / 2.0))
        ky = 0.0
    elif model == "drees":
        s = np.sin(chi)
        kx = float(4.0 / 3.0 * (1.0 - np.cos(chi) - 1.8 * mu**2) / max(s, 1e-6))
        ky = float(-2.0 * mu)
    else:
        raise ValueError(f"未知の skew model: {model}")
    return float(np.clip(kx, -2.0, 2.0)), float(np.clip(ky, -2.0, 2.0))


@dataclass
class InflowField:
    """半径方向分布 + 1 次の方位角分布をもつ誘導速度場.

    Parameters
    ----------
    r_R:
        半径格子 (無次元).
    v0:
        各半径での方位平均誘導速度 [m/s] (推力方向と逆向きを正).
    kx, ky:
        線形スキュー係数.
    psi_edge:
        面内速度の方位角 [rad].
    swirl:
        旋回誘導速度 [m/s] (ブレードの進行方向と同じ向きを正 = 相対速度を
        減らす向き). 運動量理論系のモデルでは 0, 渦法では後流の回転から
        直接求まる. None なら 0 とみなす.
    """

    r_R: np.ndarray
    v0: np.ndarray
    kx: float = 0.0
    ky: float = 0.0
    psi_edge: float = 0.0
    swirl: np.ndarray | None = None
    converged: bool = True
    iterations: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.r_R = np.asarray(self.r_R, dtype=float)
        self.v0 = np.asarray(self.v0, dtype=float)
        if self.swirl is not None:
            self.swirl = np.asarray(self.swirl, dtype=float)
        # 面積重み (mean_v0 用, 総和 1 に正規化した台形則)
        x = self.r_R
        w = np.zeros_like(x)
        if x.size > 1:
            d = np.diff(x)
            w[:-1] += 0.5 * d
            w[1:] += 0.5 * d
        w = w * 2.0 * x
        total = w.sum()
        self._area_weights = w / total if total > 1e-12 else np.full_like(x, 1.0 / x.size)

    def mean_v0(self) -> float:
        """ディスク面積で重み付けした平均誘導速度 [m/s]."""
        w = self._area_weights
        return float(w @ self.v0)

    def induced(self, r_R: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """任意の (r/R, psi) における誘導速度 [m/s].

        ``r_R`` と ``psi`` はブロードキャスト可能な形状であること.
        """
        r_R = np.asarray(r_R, dtype=float)
        psi = np.asarray(psi, dtype=float)
        v = np.interp(r_R, self.r_R, self.v0)
        if self.kx == 0.0 and self.ky == 0.0:
            return np.broadcast_to(v, np.broadcast_shapes(v.shape, psi.shape)).copy()
        dpsi = psi - self.psi_edge
        factor = 1.0 - self.kx * r_R * np.cos(dpsi) + self.ky * r_R * np.sin(dpsi)
        return v * np.clip(factor, -1.0, 3.0)

    def induced_swirl(self, r_R: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """旋回誘導速度 [m/s]. 与えられていなければ 0."""
        if self.swirl is None:
            return np.zeros(np.broadcast_shapes(
                np.shape(r_R), np.shape(psi)))
        return np.interp(np.asarray(r_R, dtype=float), self.r_R, self.swirl)

    @property
    def has_swirl(self) -> bool:
        return self.swirl is not None

    def scaled(self, factor: float) -> "InflowField":
        return InflowField(
            self.r_R, self.v0 * factor, self.kx, self.ky, self.psi_edge,
            self.swirl if self.swirl is None else self.swirl * factor,
            self.converged, self.iterations, self.warnings,
        )

    @classmethod
    def uniform(cls, r_R: np.ndarray, v0: float, **kwargs) -> "InflowField":
        r_R = np.asarray(r_R, dtype=float)
        return cls(r_R, np.full(r_R.shape, float(v0)), **kwargs)
