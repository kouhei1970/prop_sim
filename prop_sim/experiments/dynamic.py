"""動的試験 (時間領域) のシミュレーション.

時間積分する状態
----------------
    psi1   : ブレード 1 の方位角 [rad]
    Omega  : 回転角速度 [rad/s]        (ドライブトレインを与えた場合)
    v0(r)  : 動的インフロー状態 [m/s]  (BEMT の場合)

出力される 6 分力には次が重畳する.

    * 定常空力荷重
    * 斜め流入による 1/rev 変動 (Mx, My, Fx, Fy)
    * ブレード通過に伴う B/rev 変動
    * ブレード間ピッチ差 (空力アンバランス) による 1/rev 変動
    * 質量アンバランスによる 1/rev 遠心力
    * 回転数変化に伴う慣性反トルク -J dOmega/dt
    * 試験台の角運動に伴うジャイロモーメント
    * 6 分力計の構造共振・干渉・ノイズ・量子化
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..atmosphere import Atmosphere, SEA_LEVEL
from ..drivetrain import Drivetrain
from ..frames import COMPONENT_NAMES, Wrench
from ..models.base import AeroModel
from ..operating import OperatingPoint
from ..rotor import Rotor
from ..sensor import MeasurementResult, TestRig

__all__ = ["DynamicResult", "dynamic_run", "Schedule"]

Schedule = float | Callable[[float], float]


def _call(s, t: float) -> float:
    return float(s(t)) if callable(s) else float(s)


def _call_vec(s, t: float) -> np.ndarray:
    v = s(t) if callable(s) else s
    return np.asarray(v, dtype=float).reshape(3)


@dataclass
class DynamicResult:
    """時間領域シミュレーションの結果."""

    time: np.ndarray                      # (n,) シミュレーション時刻
    hub_wrench: np.ndarray                # (n, 6) ハブ中心の真値
    sensor_wrench: np.ndarray             # (n, 6) センサ原点の真値
    states: dict[str, np.ndarray] = field(default_factory=dict)
    measurement: MeasurementResult | None = None

    def columns(self, sampled: bool = True) -> dict[str, np.ndarray]:
        """CSV 出力用の列辞書."""
        if sampled and self.measurement is not None:
            out = {"time": self.measurement.time}
            for i, c in enumerate(COMPONENT_NAMES):
                out[c] = self.measurement.measured[:, i]
                out[c + "_true"] = self.measurement.truth[:, i]
            idx = np.searchsorted(self.time, self.measurement.time)
            idx = np.clip(idx, 0, self.time.size - 1)
            for k, v in self.states.items():
                out[k] = np.asarray(v)[idx]
            return out
        out = {"time": self.time}
        for i, c in enumerate(COMPONENT_NAMES):
            out[c + "_true"] = self.sensor_wrench[:, i]
        out.update({k: np.asarray(v) for k, v in self.states.items()})
        return out

    def to_csv(self, path: str, sampled: bool = True) -> None:
        from ..io import write_csv

        write_csv(path, self.columns(sampled=sampled))

    def summary(self) -> str:  # pragma: no cover - 表示のみ
        return (
            f"DynamicResult: {self.time.size} steps, "
            f"{self.time[-1] - self.time[0]:.3f} s, "
            f"Fz mean={self.sensor_wrench[:, 2].mean():.3f} N"
        )


def dynamic_run(
    rotor: Rotor,
    model: AeroModel,
    *,
    duration: float = 0.5,
    rpm: Schedule | None = None,
    throttle: Schedule | None = None,
    drivetrain: Drivetrain | None = None,
    initial_rpm: float | None = None,
    v_inf: Schedule = 0.0,
    inflow_angle_deg: Schedule = 0.0,
    inflow_azimuth_deg: Schedule = 0.0,
    body_rate: np.ndarray | Callable[[float], np.ndarray] = (0.0, 0.0, 0.0),
    atmosphere: Atmosphere = SEA_LEVEL,
    rig: TestRig | None = None,
    sim_rate: float | None = None,
    azimuth_steps_per_rev: int = 180,
    initial_azimuth_deg: float = 0.0,
    measure: bool = True,
    progress: bool = False,
) -> DynamicResult:
    """時間領域シミュレーションを実行する.

    回転数の与え方は 2 通り.

    1. ``rpm`` にスケジュール (定数または ``f(t)``) を与える
       — 回転数制御されたダイナモを模擬. 必要トルクは逆算される.
    2. ``throttle`` と ``drivetrain`` を与える
       — スロットル入力に対するモータ + プロペラの応答を解く.

    Parameters
    ----------
    duration:
        シミュレーション時間 [s].
    sim_rate:
        積分の刻み周波数 [Hz]. None なら翼素の方位角分解能
        (``azimuth_steps_per_rev``) と 6 分力計のサンプリング周波数から決める.
    azimuth_steps_per_rev:
        1 回転あたりの積分ステップ数の下限.
    rig:
        試験スタンド. 与えると 6 分力計を通した計測値も得られる.
    """
    if rpm is None and throttle is None:
        raise ValueError("rpm か throttle のいずれかを指定してください")
    if throttle is not None and drivetrain is None:
        raise ValueError("throttle を使う場合は drivetrain も指定してください")

    # --- 積分刻みの決定
    rpm_probe = [
        _call(rpm, t) for t in np.linspace(0.0, duration, 21)
    ] if rpm is not None else [initial_rpm or 5000.0]
    rpm_max = max(max(rpm_probe), initial_rpm or 0.0, 1.0)
    if sim_rate is None:
        f_az = azimuth_steps_per_rev * rpm_max / 60.0
        f_meas = 4.0 * rig.load_cell.config.sample_rate if rig is not None else 0.0
        sim_rate = max(f_az, f_meas, 2000.0)
    dt = 1.0 / float(sim_rate)
    n = int(round(duration / dt)) + 1
    t_arr = np.arange(n) * dt

    # --- 初期状態
    omega = (2.0 * np.pi / 60.0) * (
        initial_rpm if initial_rpm is not None
        else (_call(rpm, 0.0) if rpm is not None else 0.0)
    )
    psi1 = np.deg2rad(initial_azimuth_deg)

    def make_op(t: float, om: float) -> OperatingPoint:
        return OperatingPoint(
            rpm=om * 60.0 / (2.0 * np.pi),
            v_inf=_call(v_inf, t),
            inflow_angle_deg=_call(inflow_angle_deg, t),
            inflow_azimuth_deg=_call(inflow_azimuth_deg, t),
            body_rate=_call_vec(body_rate, t),
            atmosphere=atmosphere,
        )

    op0 = make_op(0.0, max(omega, 1e-6))
    inflow = model.initial_inflow(rotor, op0)

    hub = np.zeros((n, 6))
    states = {
        k: np.zeros(n)
        for k in (
            "rpm", "psi_deg", "omega_dot", "thrust", "torque_aero",
            "v_induced_mean", "v_inf", "inflow_angle_deg",
        )
    }
    if drivetrain is not None:
        for k in ("throttle", "voltage", "current", "motor_torque",
                  "electrical_power"):
            states[k] = np.zeros(n)

    refresh = getattr(model, "refresh_skew", None)

    for i, t in enumerate(t_arr):
        op = make_op(t, omega)
        if refresh is not None:
            refresh(rotor, op, inflow)

        w_aero, dv0 = model.step(rotor, op, inflow, psi1)
        q_aero = -rotor.spin * float(w_aero.moment[2])

        # --- 回転数の更新則
        aux: dict[str, float] = {}
        if drivetrain is not None:
            thr = _call(throttle, t)
            volt = drivetrain.esc.voltage_command(thr)
            omega_dot, aux = drivetrain.omega_dot(omega, volt, q_aero)
            aux["throttle"] = thr
            aux["voltage"] = volt
        else:
            om_next = (2.0 * np.pi / 60.0) * _call(rpm, min(t + dt, duration))
            omega_dot = (om_next - omega) / dt

        w = w_aero + rotor.inertial_wrench(op, psi1=psi1, omega_dot=omega_dot)
        hub[i] = w.as_array()

        states["rpm"][i] = op.rpm
        states["psi_deg"][i] = np.rad2deg(psi1) % 360.0
        states["omega_dot"][i] = omega_dot
        states["thrust"][i] = float(w_aero.force[2])
        states["torque_aero"][i] = q_aero
        states["v_induced_mean"][i] = inflow.mean_v0()
        states["v_inf"][i] = op.v_inf
        states["inflow_angle_deg"][i] = op.inflow_angle_deg
        for k, v in aux.items():
            if k in states:
                states[k][i] = v

        # --- 前進 Euler (インフローの時定数 >> dt を前提)
        psi1 += rotor.spin * omega * dt
        omega = max(omega + omega_dot * dt, 0.0)
        inflow.v0 = inflow.v0 + dv0 * dt

        if progress and n > 20 and i % (n // 10) == 0:  # pragma: no cover
            print(f"  dynamic {100 * i // n:3d}%", flush=True)

    # --- センサ座標系へ
    if rig is not None:
        sensor = rig.to_sensor(Wrench.from_array(hub)).as_array()
    else:
        sensor = hub

    meas = None
    if rig is not None and measure:
        meas = rig.load_cell.measure_series(t_arr, sensor)

    return DynamicResult(
        time=t_arr, hub_wrench=hub, sensor_wrench=sensor,
        states=states, measurement=meas,
    )
