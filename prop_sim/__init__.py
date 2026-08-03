"""prop_sim — 無人機プロペラ 6 分力計測実験のシミュレータ.

構成
----
``geometry`` / ``airfoil``   : プロペラ形状と翼型
``operating``                : 作動点 (回転数・流入速度・流入角・機体角速度)
``rotor``                    : 翼素荷重 -> ハブ 6 分力の組み立て
``models``                   : 空力モデル (BEMT / BET / 代理モデル ...)
``drivetrain``               : モータ + ESC + 回転慣性
``sensor``                   : 6 分力計 (干渉・ノイズ・動特性・A/D)
``experiments``              : 静的試験 (掃引) と動的試験 (時間領域)
``postproc``                 : 係数換算・次数分析・FFT

使用例
------
>>> import prop_sim as ps
>>> rotor = ps.Rotor(ps.from_diameter_pitch(10, 4.7))
>>> model = ps.BEMT()
>>> sol = model.solve(rotor, ps.OperatingPoint(rpm=6000))
>>> round(sol.thrust, 2) > 0
True
"""

from __future__ import annotations

from .airfoil import (
    CLARK_Y,
    FLAT_PLATE,
    LOW_RE_THIN,
    NACA0012,
    Airfoil,
    LinearAirfoil,
    TabulatedAirfoil,
)
from .atmosphere import SEA_LEVEL, Atmosphere
from .frames import COMPONENT_NAMES, Wrench
from .geometry import PropellerGeometry, from_diameter_pitch
from .inflow import InflowField
from .models import BEMT, BET, AeroModel, QuadraticModel, RotorSolution, get_model
from .operating import OperatingPoint
from .presets import (
    PRESETS,
    STAMPFLY_1209_SECTION,
    get_preset,
    stampfly_1209,
    stampfly_1209_airfoil,
    stampfly_1209_section,
)
from .rotor import Rotor, Unbalance
from .section import SectionShape, ThinAirfoilProperties, airfoil_from_section

__version__ = "0.1.0"

__all__ = [
    "Airfoil",
    "LinearAirfoil",
    "TabulatedAirfoil",
    "CLARK_Y",
    "NACA0012",
    "FLAT_PLATE",
    "LOW_RE_THIN",
    "Atmosphere",
    "SEA_LEVEL",
    "Wrench",
    "COMPONENT_NAMES",
    "PropellerGeometry",
    "from_diameter_pitch",
    "stampfly_1209",
    "stampfly_1209_airfoil",
    "stampfly_1209_section",
    "STAMPFLY_1209_SECTION",
    "get_preset",
    "PRESETS",
    "SectionShape",
    "ThinAirfoilProperties",
    "airfoil_from_section",
    "InflowField",
    "OperatingPoint",
    "Rotor",
    "Unbalance",
    "AeroModel",
    "RotorSolution",
    "BEMT",
    "BET",
    "QuadraticModel",
    "get_model",
    "__version__",
]


def __getattr__(name: str):  # 遅延インポート (matplotlib 等の重い依存を避ける)
    if name in {"Drivetrain", "Motor", "ESC"}:
        from . import drivetrain

        return getattr(drivetrain, name)
    if name in {"LoadCell", "TestRig", "SensorConfig"}:
        from . import sensor

        return getattr(sensor, name)
    if name in {"static_sweep", "dynamic_run", "StaticResult", "DynamicResult"}:
        from . import experiments

        return getattr(experiments, name)
    raise AttributeError(name)
