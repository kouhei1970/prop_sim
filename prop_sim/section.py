"""翼断面形状 (キャンバ線 + 厚み分布) から翼型モデルを作る.

ブレードを切断した断面写真から読み取った形状を空力係数に落とし込むための
モジュール. 手順は

    1. キャンバ線 z(x)/c と厚み分布 t(x)/c を与える (:class:`SectionShape`)
    2. 薄翼理論でゼロ揚力角・空力中心モーメント・設計揚力係数を求める
    3. Reynolds 数に応じた粘性補正をかけて :class:`~prop_sim.airfoil.LinearAirfoil`
       のパラメータに変換する (:func:`airfoil_from_section`)

マイクロプロペラが動く Re ~ 10^4 では層流剥離のせいで非粘性理論から
大きくずれるため、2 段目の補正が結果を支配する。補正係数はすべて
引数で上書きできるようにしてある。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .airfoil import LinearAirfoil

__all__ = ["SectionShape", "ThinAirfoilProperties", "airfoil_from_section"]


@dataclass
class ThinAirfoilProperties:
    """薄翼理論による非粘性の断面特性."""

    alpha0_rad: float        # ゼロ揚力角
    alpha_ideal_rad: float   # 理想迎角 (前縁で流れが滑らかに入る迎角)
    cl_ideal: float          # 理想迎角における揚力係数 (= 設計揚力係数)
    cm_ac: float             # 空力中心まわりのモーメント係数
    camber_max: float        # 最大キャンバ f/c
    camber_max_x: float      # その位置 x/c
    thickness_max: float     # 最大厚み t/c
    thickness_max_x: float   # その位置 x/c

    @property
    def alpha0_deg(self) -> float:
        return float(np.rad2deg(self.alpha0_rad))

    def summary(self) -> str:  # pragma: no cover - 表示のみ
        return (
            f"f/c={self.camber_max*100:.2f}% @ {self.camber_max_x:.2f}c, "
            f"t/c={self.thickness_max*100:.2f}% @ {self.thickness_max_x:.2f}c, "
            f"alpha0={self.alpha0_deg:+.2f} deg, cl_i={self.cl_ideal:.3f}, "
            f"cm_ac={self.cm_ac:+.3f}"
        )


@dataclass
class SectionShape:
    """翼断面の幾何形状 (コード長で正規化).

    Parameters
    ----------
    x:
        コード方向座標 x/c (0 = 前縁, 1 = 後縁, 昇順).
    camber:
        キャンバ線 z(x)/c. 上面側 (正圧面と反対側) が凸なら正.
    thickness:
        厚み分布 t(x)/c.
    name:
        識別名.

    Notes
    -----
    断面写真から読む場合、カメラ軸が切断面に垂直でないと投影で形状が
    圧縮される。``scaled()`` で厚み・キャンバをまとめて補正できる。
    """

    x: np.ndarray
    camber: np.ndarray
    thickness: np.ndarray
    name: str = "section"

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.camber = np.asarray(self.camber, dtype=float)
        self.thickness = np.asarray(self.thickness, dtype=float)
        if not (self.x.shape == self.camber.shape == self.thickness.shape):
            raise ValueError("x / camber / thickness は同じ長さが必要です")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("x は狭義単調増加である必要があります")

    # ------------------------------------------------------------- 派生量
    @property
    def upper(self) -> np.ndarray:
        return self.camber + 0.5 * self.thickness

    @property
    def lower(self) -> np.ndarray:
        return self.camber - 0.5 * self.thickness

    def scaled(self, thickness: float = 1.0, camber: float = 1.0) -> "SectionShape":
        """厚み・キャンバをまとめて拡大縮小する (投影誤差の補正用)."""
        return SectionShape(
            self.x, self.camber * camber, self.thickness * thickness,
            name=f"{self.name}(t x{thickness:g}, f x{camber:g})",
        )

    def resample(self, n: int = 200) -> "SectionShape":
        """余弦分布で細かくリサンプルする."""
        xn = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
        return SectionShape(
            xn, np.interp(xn, self.x, self.camber),
            np.interp(xn, self.x, self.thickness), self.name,
        )

    # ------------------------------------------------------------ 薄翼理論
    def thin_airfoil_properties(self, n: int = 400) -> ThinAirfoilProperties:
        """薄翼理論による非粘性の断面特性を求める.

        キャンバ線の傾き dz/dx を Glauert 級数に展開し

            alpha_0     = -(1/pi) Int_0^pi (dz/dx)(cos th - 1) dth
            alpha_ideal =  (1/pi) Int_0^pi (dz/dx) dth
            cm_ac       =  (pi/4) (A2 - A1)

        から求める (x = (1 - cos th)/2).
        """
        s = self.resample(n)
        x = np.clip(s.x, 1e-6, 1.0 - 1e-6)
        z = s.camber
        dz = np.gradient(z, x)
        th = np.arccos(np.clip(1.0 - 2.0 * x, -1.0, 1.0))

        alpha_ideal = float(np.trapezoid(dz, th) / np.pi)
        alpha0 = float(-(1.0 / np.pi) * np.trapezoid(dz * (np.cos(th) - 1.0), th))
        a1 = float((2.0 / np.pi) * np.trapezoid(dz * np.cos(th), th))
        a2 = float((2.0 / np.pi) * np.trapezoid(dz * np.cos(2.0 * th), th))
        cm_ac = float(0.25 * np.pi * (a2 - a1))

        i_c = int(np.argmax(np.abs(z)))
        i_t = int(np.argmax(s.thickness))
        return ThinAirfoilProperties(
            alpha0_rad=alpha0,
            alpha_ideal_rad=alpha_ideal,
            cl_ideal=2.0 * np.pi * (alpha_ideal - alpha0),
            cm_ac=cm_ac,
            camber_max=float(z[i_c]),
            camber_max_x=float(s.x[i_c]),
            thickness_max=float(s.thickness[i_t]),
            thickness_max_x=float(s.x[i_t]),
        )


def _lift_slope_efficiency(reynolds: float) -> float:
    """揚力傾斜が 2 pi の何割になるかの経験式.

    低 Re の平板・薄いキャンバ板の実測 (Re = 10^3 - 10^6) で
    cl_alpha が 3.5 (Re=10^4) から 6.0 (Re=10^6) 程度まで変わる傾向に合わせた.
    """
    r = float(np.clip(reynolds, 1.0e3, 1.0e7))
    return float(np.clip(0.20 + 0.105 * np.log10(r), 0.35, 0.96))


def _min_drag(reynolds: float, t_c: float, bubble_penalty: float) -> float:
    """最小抗力係数の推定.

    層流平板の摩擦抵抗 (両面) x 厚み形状係数 x 層流剥離バブルのペナルティ.
    """
    re = float(np.clip(reynolds, 1.0e3, 1.0e8))
    cf = 2.0 * 1.328 / np.sqrt(re)
    form = 1.0 + 2.0 * t_c + 60.0 * t_c**4
    return float(cf * form * bubble_penalty)


def airfoil_from_section(
    section: SectionShape,
    *,
    reynolds_ref: float = 1.0e4,
    reynolds_exponent: float = 0.40,
    lift_slope_efficiency: float | None = None,
    camber_efficiency: float = 0.85,
    bubble_penalty: float = 1.6,
    cl_max: float | None = None,
    cl_min: float | None = None,
    cd_k: float = 0.060,
    aspect_ratio: float = 3.0,
    name: str | None = None,
) -> LinearAirfoil:
    """断面形状 + Reynolds 数から :class:`LinearAirfoil` を作る.

    Parameters
    ----------
    section:
        キャンバ線と厚み分布.
    reynolds_ref:
        代表 Reynolds 数. ここでの係数を基準に, 実際の Re へは
        ``reynolds_exponent`` で外挿する.
    lift_slope_efficiency:
        揚力傾斜 / (2 pi). None なら ``reynolds_ref`` からの経験式.
    camber_efficiency:
        粘性によるデキャンバ率. 低 Re では境界層と剥離泡でキャンバの効果が
        目減りするため, 薄翼理論のゼロ揚力角に掛ける (実測ではおよそ 0.8-0.9).
    bubble_penalty:
        層流剥離バブルによる抗力増加率 (最小抗力の推定に使う).
    cl_max, cl_min:
        指定しなければキャンバから推定する.
    cd_k:
        cd = cd0 + cd_k (cl - cl_i)^2 の係数.

    Returns
    -------
    LinearAirfoil
        ``prop_sim`` の 360 deg 翼型モデル.

    Notes
    -----
    低 Re の翼型データは分散が大きく, ここでの推定はあくまで「断面形状が
    分かっているときの妥当な出発点」である。静止推力を 1 点でも実測できるなら
    ``Rotor(collective_deg=...)`` か ``camber_efficiency`` で較正すること。
    """
    props = section.thin_airfoil_properties()
    eta = (
        _lift_slope_efficiency(reynolds_ref)
        if lift_slope_efficiency is None
        else float(lift_slope_efficiency)
    )
    cl_alpha = 2.0 * np.pi * eta
    alpha0 = props.alpha0_rad * camber_efficiency
    f_c = abs(props.camber_max)
    t_c = props.thickness_max

    if cl_max is None:
        # 低 Re の平板 (cl_max ~ 0.8) を基準に, キャンバで上積み
        cl_max = float(np.clip(0.80 + 3.0 * f_c, 0.70, 1.45))
    if cl_min is None:
        # キャンバが強いほど負側は早く失速する
        cl_min = float(-np.clip(0.55 - 2.0 * f_c, 0.20, 0.90))

    cd0 = _min_drag(reynolds_ref, t_c, bubble_penalty)
    cl_i = float(np.clip(cl_alpha * (props.alpha_ideal_rad - alpha0), 0.0, cl_max))

    return LinearAirfoil(
        name=name or f"{section.name} @Re={reynolds_ref:.0f}",
        cl_alpha=cl_alpha,
        alpha0_deg=float(np.rad2deg(alpha0)),
        cl_max=cl_max,
        cl_min=cl_min,
        cd0=cd0,
        cd_k=cd_k,
        cl_cd0=cl_i,
        cm=props.cm_ac * camber_efficiency,
        aspect_ratio=aspect_ratio,
        reynolds_ref=reynolds_ref,
        reynolds_exponent=reynolds_exponent,
    )
