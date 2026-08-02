"""静的試験 (定常掃引) のシミュレーション.

回転数・流入速度・流入角の格子上で定常解を求め, 6 分力計を通した
「計測値」と真値を並べて返す. 実際の試験計画 (どの条件を何点測るか)
の検討や, 後処理アルゴリズムの検証に使う.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from ..atmosphere import Atmosphere, SEA_LEVEL
from ..drivetrain import Drivetrain
from ..frames import COMPONENT_NAMES
from ..models.base import AeroModel
from ..operating import OperatingPoint
from ..rotor import Rotor
from ..sensor import TestRig

__all__ = ["StaticResult", "static_sweep"]

_CONDITION_KEYS = ("rpm", "v_inf", "inflow_angle_deg", "inflow_azimuth_deg")


@dataclass
class StaticResult:
    """静的掃引の結果.

    ``columns`` は列名 -> 1 次元配列の辞書. CSV にそのまま書ける.
    """

    columns: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0

    def __getitem__(self, key: str) -> np.ndarray:
        return self.columns[key]

    def keys(self):
        return self.columns.keys()

    def truth(self) -> np.ndarray:
        return np.stack([self.columns[c + "_true"] for c in COMPONENT_NAMES], axis=1)

    def measured(self) -> np.ndarray:
        key = COMPONENT_NAMES[0]
        if key not in self.columns:
            return self.truth()
        return np.stack([self.columns[c] for c in COMPONENT_NAMES], axis=1)

    def to_csv(self, path: str) -> None:
        from ..io import write_csv

        write_csv(path, self.columns)

    def summary(self) -> str:  # pragma: no cover - 表示のみ
        return f"StaticResult: {self.n_points} 点, {len(self.columns)} 列"


def static_sweep(
    rotor: Rotor,
    model: AeroModel,
    *,
    rpm: np.ndarray | float = 5000.0,
    v_inf: np.ndarray | float = 0.0,
    inflow_angle_deg: np.ndarray | float = 0.0,
    inflow_azimuth_deg: np.ndarray | float = 0.0,
    atmosphere: Atmosphere = SEA_LEVEL,
    rig: TestRig | None = None,
    drivetrain: Drivetrain | None = None,
    n_average: int = 1000,
    include_sections: bool = False,
    progress: bool = False,
) -> StaticResult:
    """条件格子上で定常 6 分力を計算する.

    Parameters
    ----------
    rotor, model:
        供試プロペラと空力モデル.
    rpm, v_inf, inflow_angle_deg, inflow_azimuth_deg:
        掃引条件. スカラーまたは配列 (直積で格子を作る).
    rig:
        試験スタンド + 6 分力計. 与えると計測値の列も追加される.
    drivetrain:
        与えるとスロットル・電流・電気入力の推定値も出力する.
    n_average:
        1 点あたりの平均化サンプル数 (計測ノイズの低減量に影響).
    include_sections:
        r/R = 0.75 における迎角など翼素の代表値を列に加える.

    Returns
    -------
    StaticResult
        条件列 + 真値列 (``Fx_true`` 等) + 計測列 (``Fx`` 等) + 係数列.
    """
    grids = [np.atleast_1d(np.asarray(v, dtype=float))
             for v in (rpm, v_inf, inflow_angle_deg, inflow_azimuth_deg)]
    combos = list(product(*grids))
    n = len(combos)

    cols: dict[str, list[float]] = {k: [] for k in _CONDITION_KEYS}
    for c in COMPONENT_NAMES:
        cols[c + "_true"] = []
    coeff_keys: list[str] = []
    extra: dict[str, list[float]] = {}
    truths = np.zeros((n, 6))
    warnings: list[str] = []

    for i, (r, v, ang, azi) in enumerate(combos):
        op = OperatingPoint(
            rpm=float(r), v_inf=float(v), inflow_angle_deg=float(ang),
            inflow_azimuth_deg=float(azi), atmosphere=atmosphere,
        )
        sol = model.solve(rotor, op)
        hub = sol.wrench + rotor.inertial_wrench(
            op, psi1=0.0, omega_dot=0.0, include_unbalance=False
        )
        w = rig.to_sensor(hub) if rig is not None else hub
        truths[i] = w.as_array()

        cols["rpm"].append(op.rpm)
        cols["v_inf"].append(op.v_inf)
        cols["inflow_angle_deg"].append(op.inflow_angle_deg)
        cols["inflow_azimuth_deg"].append(op.inflow_azimuth_deg)
        for j, c in enumerate(COMPONENT_NAMES):
            cols[c + "_true"].append(truths[i, j])

        cf = sol.coefficients(rotor, op)
        if not coeff_keys:
            coeff_keys = list(cf)
            for k in coeff_keys:
                extra[k] = []
        for k in coeff_keys:
            extra[k].append(cf.get(k, np.nan))
        extra.setdefault("thrust", []).append(sol.thrust)
        extra.setdefault("torque", []).append(sol.torque(rotor.spin))
        extra.setdefault("power_shaft", []).append(sol.power(op, rotor.spin))
        extra.setdefault("v_induced_mean", []).append(sol.inflow.mean_v0())
        extra.setdefault("tip_mach", []).append(op.tip_mach(rotor.radius))

        if drivetrain is not None:
            q = sol.torque(rotor.spin)
            thr = drivetrain.steady_throttle(op.rpm, q)
            volt = drivetrain.esc.voltage_command(thr)
            _, cur, p_el = drivetrain.motor.electrical(volt, op.omega)
            extra.setdefault("throttle", []).append(thr)
            extra.setdefault("voltage", []).append(volt)
            extra.setdefault("current", []).append(cur)
            extra.setdefault("power_electrical", []).append(p_el)
            extra.setdefault("efficiency_total", []).append(
                sol.power(op, rotor.spin) / p_el if abs(p_el) > 1e-9 else np.nan
            )

        if include_sections and sol.sections is not None:
            st = sol.sections
            k75 = int(np.argmin(np.abs(st.r_R - 0.75)))
            extra.setdefault("alpha75_deg", []).append(
                float(np.rad2deg(st.alpha[:, k75].mean()))
            )
            extra.setdefault("re75", []).append(float(st.reynolds[:, k75].mean()))
            extra.setdefault("alpha_max_deg", []).append(
                float(np.rad2deg(st.alpha.max()))
            )

        for msg in sol.warnings:
            entry = f"[rpm={op.rpm:.0f}, V={op.v_inf:.1f}, a={op.inflow_angle_deg:.0f}] {msg}"
            if entry not in warnings:
                warnings.append(entry)

        if progress and (i % max(n // 20, 1) == 0):  # pragma: no cover
            print(f"  static sweep {i+1}/{n}", flush=True)

    columns = {k: np.asarray(v, dtype=float) for k, v in cols.items()}

    if rig is not None:
        meas = rig.load_cell.measure_static(truths, n_average=n_average)
        for j, c in enumerate(COMPONENT_NAMES):
            columns[c] = meas[:, j]
            columns[c + "_error"] = meas[:, j] - truths[:, j]

    for k, v in extra.items():
        columns[k] = np.asarray(v, dtype=float)

    return StaticResult(columns=columns, warnings=warnings)
