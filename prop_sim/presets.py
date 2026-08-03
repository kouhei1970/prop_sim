"""実機プロペラのプリセット.

形状データの出所は各関数の docstring に明記する. 平面形が写真からの
実測の場合でも, ねじり角 (ピッチ分布) は真上からの写真では読めないため
公称ピッチからの推定になる — その旨も明記する.
"""

from __future__ import annotations

import numpy as np

from .airfoil import Airfoil
from .geometry import INCH, PropellerGeometry
from .section import SectionShape, airfoil_from_section

__all__ = [
    "stampfly_1209",
    "stampfly_1209_section",
    "stampfly_1209_airfoil",
    "STAMPFLY_1209_SECTION",
    "PRESETS",
    "get_preset",
]


# --------------------------------------------------------------------------
# StampFly 1209 (31 mm 4 枚)
# --------------------------------------------------------------------------
#: 写真計測によるコード長分布 (r/R, c/R).
#: 真上から撮影した写真 (定規基準 77.0 px/mm) を色分割し, 半径ごとの
#: 方位角方向の広がり (円弧長) を求めたうえで, ブレード中心線の後退角
#: Lambda(r) による cos(Lambda) 補正をかけて「ピッチ軸に直交する弦長」に
#: 換算したもの.
_STAMPFLY_1209_CHORD = np.array([
    #  r/R    c [mm]
    [0.226, 5.00],
    [0.300, 5.15],
    [0.350, 5.30],
    [0.400, 5.42],
    [0.450, 5.62],
    [0.500, 5.80],
    [0.550, 5.86],
    [0.600, 5.82],
    [0.650, 5.68],
    [0.700, 5.50],
    [0.750, 5.17],
    [0.800, 4.82],
    [0.850, 4.55],
    [0.900, 4.10],
    [0.950, 3.35],
    [0.980, 2.60],
    [1.000, 1.50],
])

#: 写真から読み取ったブレード中心線の後退角 [deg] (空力計算には未使用,
#: 形状の記録として保持する).
STAMPFLY_1209_SWEEP_DEG = np.array([
    [0.32, 1.2], [0.45, 6.3], [0.57, 16.0], [0.69, 21.9],
    [0.75, 24.9], [0.87, 26.0], [0.93, 32.8], [1.00, 34.5],
])

#: 切断面の写真から読み取った翼断面 (x/c, キャンバ z/c, 厚み t/c).
#:
#: 黒く塗った断面を「明度が低く かつ 彩度も低い」画素として抽出する
#: (gray < 85 かつ sat < 50). 明度だけで切ると前縁下面に落ちた影 (暗い赤)
#: まで拾ってしまい, 下面が実際より 2-3 %c 下側に膨らむ.
#: 薄くて背景と紛れやすい後縁側 (x/c > 0.7) だけ緩い閾値の結果を併用する.
#: そのうえで最遠点対を前縁-後縁としてコード系に回転し, コード方向
#: 100 分割の上下面からキャンバ線と厚み分布を求めて多項式で平滑化した.
_STAMPFLY_1209_SECTION = np.array([
    #  x/c     z/c      t/c
    [0.0000, 0.00000, 0.00400],
    [0.0125, 0.00573, 0.03261],
    [0.0250, 0.01096, 0.04511],
    [0.0500, 0.02004, 0.06086],
    [0.0750, 0.02758, 0.07092],
    [0.1000, 0.03382, 0.07770],
    [0.1500, 0.04337, 0.08507],
    [0.2000, 0.05015, 0.08711],
    [0.2500, 0.05520, 0.08590],
    [0.3000, 0.05913, 0.08279],
    [0.3500, 0.06228, 0.07880],
    [0.4000, 0.06473, 0.07468],
    [0.4500, 0.06641, 0.07099],
    [0.5000, 0.06714, 0.06807],
    [0.5500, 0.06668, 0.06606],
    [0.6000, 0.06479, 0.06492],
    [0.6500, 0.06127, 0.06434],
    [0.7000, 0.05602, 0.06381],
    [0.7500, 0.04905, 0.06256],
    [0.8000, 0.04050, 0.05958],
    [0.8500, 0.03071, 0.05356],
    [0.9000, 0.02018, 0.04292],
    [0.9500, 0.00963, 0.02580],
    [1.0000, 0.00000, 0.01500],
])

#: 上の表を :class:`~prop_sim.section.SectionShape` にしたもの.
STAMPFLY_1209_SECTION = SectionShape(
    x=_STAMPFLY_1209_SECTION[:, 0],
    camber=_STAMPFLY_1209_SECTION[:, 1],
    thickness=_STAMPFLY_1209_SECTION[:, 2],
    name="StampFly 1209 section",
)


def stampfly_1209_section(
    thickness_scale: float = 1.0, camber_scale: float = 1.0
) -> SectionShape:
    """実測翼断面 (投影誤差を補正したい場合はスケールを与える)."""
    if thickness_scale == 1.0 and camber_scale == 1.0:
        return STAMPFLY_1209_SECTION
    return STAMPFLY_1209_SECTION.scaled(thickness_scale, camber_scale)


def stampfly_1209_airfoil(reynolds_ref: float = 1.2e4, **kwargs):
    """実測断面から作った翼型モデル.

    ``reynolds_ref`` の既定 1.2e4 はホバー付近 (3 万 rpm) の 0.75R における
    Reynolds 数. ``prop_sim.section.airfoil_from_section`` の引数を
    そのまま渡せる.
    """
    kwargs.setdefault("name", "StampFly 1209 (measured section)")
    return airfoil_from_section(
        STAMPFLY_1209_SECTION, reynolds_ref=reynolds_ref, **kwargs
    )


#: 写真計測のまとめ
STAMPFLY_1209_MEASURED = {
    "source": "IMG_3578.jpeg (真上からの写真, 金属定規 77.0 px/mm で校正)",
    "diameter_mm": 31.21,          # 実測 (外周 99.5 パーセンタイル)
    "n_blades": 4,
    "hub_outer_diameter_mm": 7.04,  # 方位角被覆率が 93% を切る半径 x2
    "max_chord_mm": 5.86,           # r/R = 0.55 付近
    "projected_area_mm2": 292.9,    # 4 枚 + ハブの投影面積
    "blade_area_ratio": 0.383,      # 投影面積 / ディスク面積
    "tip_sweep_deg": 34.5,
    "section_source": "IMG_3720.jpeg (切断面を黒塗りした写真)",
    "section_thickness_max": 0.087,     # t/c
    "section_thickness_max_x": 0.20,
    "section_camber_max": 0.067,        # f/c
    "section_camber_max_x": 0.50,
    "section_alpha0_deg_inviscid": -6.90,   # 薄翼理論
    "section_cl_ideal": 0.923,
    "section_cm_ac": -0.175,
    "estimated_mass_g": 0.26,       # 面積 x 板厚 0.55mm/ハブ 2.0mm x PC 1.20 g/cm^3
    "polar_inertia_kgm2": 1.67e-8,  # 上記質量分布から (= 0.264 m R^2)
}


def stampfly_1209(
    *,
    pitch_in: float = 0.9,
    pitch_distribution: str = "geometric",
    airfoil: Airfoil | None = None,
    mass: float = 0.26e-3,
    polar_inertia: float | None = 1.67e-8,
    diameter_mm: float = 31.21,
    n_stations: int = 24,
) -> PropellerGeometry:
    """M5Stack StampFly 用 1209 プロペラ (31 mm, 4 枚).

    平面形 (直径・コード長分布・ハブ径・後退角・面積) と翼断面 (キャンバ線・
    厚み分布) は実機の写真から実測した値。ピッチ分布だけは写真から読み取れ
    ないため、型番 "1209" の公称ピッチ 0.9 inch から幾何ピッチ一定として
    与えている。

    Parameters
    ----------
    pitch_in:
        公称ピッチ [inch]. 既定は型番の 0.9 inch.
    pitch_distribution:
        ``"geometric"``: 幾何ピッチ一定 beta = atan(P / (2 pi r))
        ``"constant"``  : 0.75R のねじり角で一定
        ``"washout"``   : 幾何ピッチ一定から翼端で 2 deg 抜く
    airfoil:
        翼型. None なら実測翼断面から作ったモデル
        (:func:`stampfly_1209_airfoil`) を使う.
        この大きさのプロペラは翼端でも Re = 5e3 - 2e4 にしかならないため,
        通常の翼型ポーラを使うと推力を大きく過大評価する.
    mass, polar_inertia:
        質量 [kg] と極慣性モーメント [kg m^2]. 既定は写真の面積分布と
        板厚・樹脂密度の仮定からの推定値.

    Notes
    -----
    絶対値の精度を求める場合は, 静止推力を 1 点だけ実測して
    ``Rotor(collective_deg=...)`` を較正するのが最も効果的.
    ピッチが 0.1 inch ずれると静止推力は 10% 程度変わる.
    """
    if airfoil is None:
        airfoil = stampfly_1209_airfoil()
    radius = 0.5 * diameter_mm * 1e-3
    x_root = float(_STAMPFLY_1209_CHORD[0, 0])
    x = np.linspace(x_root, 1.0, n_stations)
    chord_R = np.interp(
        x, _STAMPFLY_1209_CHORD[:, 0], _STAMPFLY_1209_CHORD[:, 1]
    ) * 1e-3 / radius

    pitch = pitch_in * INCH
    beta_geo = np.arctan2(pitch, 2.0 * np.pi * x * radius)
    if pitch_distribution == "geometric":
        beta = beta_geo
    elif pitch_distribution == "constant":
        beta = np.full_like(x, float(np.interp(0.75, x, beta_geo)))
    elif pitch_distribution == "washout":
        beta = beta_geo - np.deg2rad(2.0) * np.clip((x - 0.5) / 0.5, 0.0, 1.0)
    else:
        raise ValueError(f"未知の pitch_distribution: {pitch_distribution}")
    beta = np.minimum(beta, np.deg2rad(48.0))

    return PropellerGeometry(
        radius=radius,
        n_blades=4,
        r_R=x,
        chord_R=chord_R,
        twist_deg=np.rad2deg(beta),
        airfoils=[airfoil],
        hub_radius_ratio=x_root,
        mass=mass,
        polar_inertia=polar_inertia,
        name=f"StampFly 1209 (31mm x4, P={pitch_in:g}in)",
    )


PRESETS = {
    "stampfly_1209": stampfly_1209,
}


def get_preset(name: str, **kwargs) -> PropellerGeometry:
    """名前からプリセット形状を生成する."""
    key = name.lower().replace("-", "_")
    if key not in PRESETS:
        raise KeyError(f"未知のプリセット '{name}'. 利用可能: {sorted(PRESETS)}")
    return PRESETS[key](**kwargs)
