"""翼型の空力係数モデル.

翼素理論では迎角が -180..180 deg の全域で必要になる (静止推力試験の
根元域, 逆流域, 逆回転など). ここでは

  * 付着流域 : 線形 (または実測ポーラ) の内挿
  * 失速後   : Viterna-Corrigan 外挿
  * 逆流域   : 対称性を用いた折り返し

を組み合わせた 360 deg モデルを提供する.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Airfoil",
    "LinearAirfoil",
    "TabulatedAirfoil",
    "wrap_to_pi",
    "NACA0012",
    "CLARK_Y",
    "FLAT_PLATE",
    "LOW_RE_THIN",
]


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """角度を [-pi, pi) に折り返す."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _viterna_coeffs(
    alpha_stall: float, cl_stall: float, cd_stall: float, aspect_ratio: float
) -> tuple[float, float, float, float]:
    """Viterna-Corrigan 外挿の係数 (A1, B1, A2, B2)."""
    ar = float(np.clip(aspect_ratio, 1.0, 50.0))
    cd_max = float(np.clip(1.11 + 0.018 * ar, 1.11, 2.01))
    sa = np.sin(alpha_stall)
    ca = np.cos(alpha_stall)
    ca = float(np.clip(ca, 1e-3, None))
    sa = float(np.clip(sa, 1e-3, None))
    b1 = cd_max
    a1 = b1 / 2.0
    b2 = (cd_stall - cd_max * sa**2) / ca
    a2 = (cl_stall - cd_max * sa * ca) * sa / ca**2
    return a1, b1, a2, b2


def _viterna_eval(
    alpha: np.ndarray, a1: float, b1: float, a2: float, b2: float
) -> tuple[np.ndarray, np.ndarray]:
    """[alpha_stall, pi/2] における Viterna 外挿値."""
    sa = np.clip(np.sin(alpha), 1e-6, None)
    ca = np.cos(alpha)
    cl = a1 * np.sin(2.0 * alpha) + a2 * ca**2 / sa
    cd = b1 * sa**2 + b2 * ca
    return cl, cd


class Airfoil(ABC):
    """翼型モデルの基底クラス."""

    name: str = "airfoil"

    @abstractmethod
    def _base_coefficients(
        self, alpha: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """基準 Reynolds 数・非圧縮での (cl, cd, cm). alpha は [rad]."""

    # 補正パラメータ (サブクラスで上書き可)
    reynolds_ref: float = 5.0e5
    reynolds_exponent: float = 0.2
    mach_divergence: float = 0.72
    compressibility: bool = True

    def coefficients(
        self,
        alpha: np.ndarray,
        reynolds: np.ndarray | None = None,
        mach: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """迎角 [rad], Re, Mach から (cl, cd, cm) を返す."""
        alpha = np.asarray(alpha, dtype=float)
        cl, cd, cm = self._base_coefficients(alpha)

        if reynolds is not None and self.reynolds_exponent:
            re = np.clip(np.asarray(reynolds, dtype=float), 1.0e3, None)
            # 低 Re では抗力が増え, 揚力傾斜がわずかに落ちる
            scale = (self.reynolds_ref / re) ** self.reynolds_exponent
            cd = cd * np.clip(scale, 0.4, 6.0)
            cl = cl * np.clip(1.0 - 0.06 * np.log10(self.reynolds_ref / re), 0.7, 1.05)

        if mach is not None and self.compressibility:
            ma = np.clip(np.asarray(mach, dtype=float), 0.0, 0.95)
            # Prandtl-Glauert (亜音速域のみ)
            beta = np.sqrt(np.clip(1.0 - ma**2, 0.04, 1.0))
            cl = cl / beta
            # 抵抗発散
            dm = np.clip(ma - self.mach_divergence, 0.0, None)
            cd = cd + 12.0 * dm**3

        return cl, cd, cm


@dataclass
class LinearAirfoil(Airfoil):
    """線形揚力 + 放物線抗力 + Viterna 外挿の解析的翼型.

    Parameters
    ----------
    cl_alpha:
        揚力傾斜 [1/rad].
    alpha0_deg:
        ゼロ揚力角 [deg] (キャンバ付き翼型では負).
    cl_max, cl_min:
        正/負の最大揚力係数.
    cd0, cd_k:
        cd = cd0 + cd_k * (cl - cl_cd0)^2.
    aspect_ratio:
        Viterna 外挿で使うブレードのアスペクト比 (R/c 目安).
    """

    name: str = "linear"
    cl_alpha: float = 2.0 * np.pi * 0.92
    alpha0_deg: float = -2.0
    cl_max: float = 1.20
    cl_min: float = -0.90
    cd0: float = 0.012
    cd_k: float = 0.020
    cl_cd0: float = 0.25
    cm: float = -0.03
    aspect_ratio: float = 8.0
    reynolds_ref: float = 5.0e5
    reynolds_exponent: float = 0.2

    def __post_init__(self) -> None:
        self.alpha0 = np.deg2rad(self.alpha0_deg)
        # 正側失速
        self._a_s_p = self.cl_max / self.cl_alpha
        cd_s_p = self.cd0 + self.cd_k * (self.cl_max - self.cl_cd0) ** 2
        self._vit_p = _viterna_coeffs(
            max(self._a_s_p, np.deg2rad(5.0)), self.cl_max, cd_s_p, self.aspect_ratio
        )
        # 負側失速
        self._a_s_n = abs(self.cl_min) / self.cl_alpha
        cd_s_n = self.cd0 + self.cd_k * (self.cl_min - self.cl_cd0) ** 2
        self._vit_n = _viterna_coeffs(
            max(self._a_s_n, np.deg2rad(5.0)), abs(self.cl_min), cd_s_n,
            self.aspect_ratio,
        )

    def _branch(
        self, a_rel: np.ndarray, positive: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """0 <= a_rel <= pi に対する (|cl| 相当, cd). 符号は呼び出し側で処理."""
        if positive:
            a_s, vit, cl_s = self._a_s_p, self._vit_p, self.cl_max
            cl_lin_max = self.cl_max
        else:
            a_s, vit, cl_s = self._a_s_n, self._vit_n, abs(self.cl_min)
            cl_lin_max = abs(self.cl_min)
        a1, b1, a2, b2 = vit

        sgn = 1.0 if positive else -1.0
        cl = np.zeros_like(a_rel)
        cd = np.zeros_like(a_rel)

        # 1) 付着流域
        m_att = a_rel <= a_s
        cl[m_att] = np.minimum(self.cl_alpha * a_rel[m_att], cl_lin_max)
        cd[m_att] = self.cd0 + self.cd_k * (sgn * cl[m_att] - self.cl_cd0) ** 2

        # 2) 失速後 (Viterna, a_s..pi/2)
        m_vit = (a_rel > a_s) & (a_rel <= np.pi / 2)
        if np.any(m_vit):
            cl[m_vit], cd[m_vit] = _viterna_eval(a_rel[m_vit], a1, b1, a2, b2)

        # 3) 後縁から流入する領域 (pi/2 .. pi - a_s) : pi - a で折り返し
        m_rev = (a_rel > np.pi / 2) & (a_rel <= np.pi - a_s)
        if np.any(m_rev):
            cl_v, cd_v = _viterna_eval(np.pi - a_rel[m_rev], a1, b1, a2, b2)
            cl[m_rev] = -cl_v
            cd[m_rev] = cd_v

        # 4) 完全逆流域 (pi - a_s .. pi) : -cl_s -> 0 へ線形に戻す
        m_back = a_rel > np.pi - a_s
        if np.any(m_back):
            cd_edge = float(_viterna_eval(np.asarray([a_s]), a1, b1, a2, b2)[1][0])
            t = (a_rel[m_back] - (np.pi - a_s)) / max(a_s, 1e-6)
            cl[m_back] = -cl_s * (1.0 - t)
            cd[m_back] = cd_edge + t * (2.5 * self.cd0 - cd_edge)
        return cl, cd

    def _base_coefficients(
        self, alpha: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = wrap_to_pi(alpha - self.alpha0)
        pos = a >= 0.0
        a_abs = np.abs(a)

        cl = np.zeros_like(a)
        cd = np.zeros_like(a)

        if np.any(pos):
            cl_p, cd_p = self._branch(a_abs[pos], positive=True)
            cl[pos] = cl_p
            cd[pos] = cd_p
        neg = ~pos
        if np.any(neg):
            cl_n, cd_n = self._branch(a_abs[neg], positive=False)
            cl[neg] = -cl_n
            cd[neg] = cd_n

        cm = np.full_like(a, self.cm)
        return cl, np.clip(cd, 1e-4, None), cm


@dataclass
class TabulatedAirfoil(Airfoil):
    """実測 / XFOIL 由来のポーラを内挿する翼型.

    与えたテーブルの範囲外は Viterna 外挿で 360 deg に拡張する.
    """

    alpha_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray | None = None
    name: str = "tabulated"
    aspect_ratio: float = 8.0
    reynolds_ref: float = 5.0e5
    reynolds_exponent: float = 0.2
    _extended: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        a = np.deg2rad(np.asarray(self.alpha_deg, dtype=float))
        cl = np.asarray(self.cl, dtype=float)
        cd = np.asarray(self.cd, dtype=float)
        cm = (
            np.zeros_like(cl)
            if self.cm is None
            else np.asarray(self.cm, dtype=float)
        )
        order = np.argsort(a)
        a, cl, cd, cm = a[order], cl[order], cd[order], cm[order]

        # 高迎角側を Viterna で補完する解析翼型を作る
        proxy = LinearAirfoil(
            cl_alpha=float(np.gradient(cl, a)[np.argmin(np.abs(a))]) or 5.7,
            alpha0_deg=float(np.rad2deg(np.interp(0.0, cl, a))),
            cl_max=float(np.max(cl)),
            cl_min=float(np.min(cl)),
            cd0=float(np.min(cd)),
            aspect_ratio=self.aspect_ratio,
        )
        grid = np.deg2rad(np.arange(-180.0, 180.5, 1.0))
        cl_g, cd_g, cm_g = proxy._base_coefficients(grid)
        # テーブルがある区間は実測値で置き換える
        inside = (grid >= a[0]) & (grid <= a[-1])
        cl_g[inside] = np.interp(grid[inside], a, cl)
        cd_g[inside] = np.interp(grid[inside], a, cd)
        cm_g[inside] = np.interp(grid[inside], a, cm)
        # 継ぎ目を平滑化
        cl_g = _smooth(cl_g, 3)
        cd_g = _smooth(cd_g, 3)
        self._extended = (grid, cl_g, cd_g, cm_g)

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "TabulatedAirfoil":
        """``alpha_deg, cl, cd[, cm]`` の列を持つ CSV を読む."""
        data = np.genfromtxt(path, delimiter=",", names=True)
        cm = data["cm"] if "cm" in (data.dtype.names or ()) else None
        return cls(
            alpha_deg=data["alpha_deg"], cl=data["cl"], cd=data["cd"], cm=cm, **kwargs
        )

    def _base_coefficients(
        self, alpha: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid, cl_g, cd_g, cm_g = self._extended
        a = wrap_to_pi(alpha)
        return (
            np.interp(a, grid, cl_g),
            np.clip(np.interp(a, grid, cd_g), 1e-4, None),
            np.interp(a, grid, cm_g),
        )


def _smooth(y: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return y
    k = np.ones(n) / n
    return np.convolve(y, k, mode="same")


# ---------------------------------------------------------------- 既定の翼型
#: 対称翼 (低 Re, 小型プロペラ根元向け)
NACA0012 = LinearAirfoil(
    name="NACA0012",
    cl_alpha=2 * np.pi * 0.90,
    alpha0_deg=0.0,
    cl_max=1.05,
    cl_min=-1.05,
    cd0=0.011,
    cd_k=0.020,
    cl_cd0=0.0,
    cm=0.0,
)

#: キャンバ付き汎用翼 (小型プロペラの標準的な断面)
CLARK_Y = LinearAirfoil(
    name="ClarkY",
    cl_alpha=2 * np.pi * 0.92,
    alpha0_deg=-3.5,
    cl_max=1.25,
    cl_min=-0.75,
    cd0=0.013,
    cd_k=0.022,
    cl_cd0=0.35,
    cm=-0.075,
)

#: 超低 Reynolds 数 (Re ~ 1e4) の薄翼.
#: 31 mm 級のマイクロプロペラは翼端でも Re = 5e3 - 2e4 にしかならず,
#: 層流剥離のせいで揚力傾斜が 2 pi の 6 割程度まで落ち, 抗力は
#: 通常の翼型の 3 - 5 倍になる. 平板・薄いキャンバ板の低 Re 実測
#: (Re = 1e4 級) の傾向に合わせた係数.
LOW_RE_THIN = LinearAirfoil(
    name="low_Re_thin",
    cl_alpha=2 * np.pi * 0.62,
    alpha0_deg=-2.5,
    cl_max=0.85,
    cl_min=-0.55,
    cd0=0.045,
    cd_k=0.060,
    cl_cd0=0.30,
    cm=-0.04,
    aspect_ratio=3.0,
    reynolds_ref=1.0e4,
    reynolds_exponent=0.40,
)

#: 平板 (翼端付近・薄い部分の近似)
FLAT_PLATE = LinearAirfoil(
    name="flat_plate",
    cl_alpha=2 * np.pi * 0.85,
    alpha0_deg=0.0,
    cl_max=0.85,
    cl_min=-0.85,
    cd0=0.020,
    cd_k=0.030,
    cl_cd0=0.0,
    cm=0.0,
)
