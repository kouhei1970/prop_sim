"""実測データによるモデル較正.

低 Reynolds 数のマイクロプロペラでは, 形状が正確に分かっていても
翼型の粘性特性の不確かさで推力が数十 % ずれる. 実測が 1 点でもあれば
そこに合わせ込むのが最短で, 2 点以上あれば「どのパラメータがずれて
いるのか」まで切り分けられる.

較正できるパラメータ
--------------------
================  ================================  ======================
パラメータ         物理的な意味                       回転数依存性
================  ================================  ======================
``collective``     取付角 / ピッチのずれ [deg]        ほぼ無し (Ct が一定倍)
``camber``         キャンバの有効率 (粘性デキャンバ)  ほぼ無し
``re_lift``        揚力の Reynolds 数依存の強さ       **あり**
``cd_scale``       抗力の倍率 (トルクに効く)          わずか
================  ================================  ======================

**1 点だけの実測では ``collective`` と ``re_lift`` を区別できない** —
どちらでもその 1 点には合わせられるが, 他の回転数での予測は大きく食い違う.
:func:`compare_hypotheses` がその食い違いを定量的に出す.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .models.base import AeroModel
from .operating import OperatingPoint
from .rotor import Rotor

__all__ = ["ThrustMeasurement", "CalibrationResult", "calibrate", "compare_hypotheses"]

G = 9.80665
_PARAMS = ("collective", "camber", "re_lift", "cd_scale")


@dataclass
class ThrustMeasurement:
    """推力試験の 1 点.

    Parameters
    ----------
    rpm:
        回転数 [rev/min]. ``omega`` を使う場合は省略可.
    thrust_gf:
        推力 [gf]. ``thrust_n`` を使う場合は省略可.
    torque_nm:
        軸トルク [N m] (あれば抗力側の較正にも使う).
    v_inf, inflow_angle_deg:
        風洞試験なら流入条件.
    weight:
        最小二乗の重み.
    """

    rpm: float | None = None
    thrust_gf: float | None = None
    omega: float | None = None
    thrust_n: float | None = None
    torque_nm: float | None = None
    v_inf: float = 0.0
    inflow_angle_deg: float = 0.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.rpm is None:
            if self.omega is None:
                raise ValueError("rpm か omega のどちらかが必要です")
            self.rpm = float(self.omega) * 60.0 / (2.0 * np.pi)
        if self.thrust_n is None:
            if self.thrust_gf is None:
                raise ValueError("thrust_gf か thrust_n のどちらかが必要です")
            self.thrust_n = float(self.thrust_gf) * G * 1e-3
        if self.thrust_gf is None:
            self.thrust_gf = self.thrust_n / G * 1e3

    def operating_point(self, **kwargs) -> OperatingPoint:
        return OperatingPoint(
            rpm=float(self.rpm), v_inf=self.v_inf,
            inflow_angle_deg=self.inflow_angle_deg, **kwargs,
        )


@dataclass
class CalibrationResult:
    """較正結果."""

    parameters: dict[str, float]
    rotor: Rotor
    residual_gf: np.ndarray
    measurements: list[ThrustMeasurement]
    predicted_gf: np.ndarray
    success: bool = True
    message: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def rms_error_gf(self) -> float:
        return float(np.sqrt(np.mean(self.residual_gf**2)))

    @property
    def max_error_pct(self) -> float:
        meas = np.array([m.thrust_gf for m in self.measurements])
        return float(np.max(np.abs(self.residual_gf / np.maximum(meas, 1e-12))) * 100)

    def report(self) -> str:
        lines = ["較正結果", "  " + ", ".join(
            f"{k}={v:+.4g}" for k, v in self.parameters.items())]
        lines.append(f"  {'rpm':>7}{'実測[gf]':>10}{'モデル[gf]':>11}{'誤差':>9}")
        for m, p in zip(self.measurements, self.predicted_gf):
            e = (p - m.thrust_gf) / max(m.thrust_gf, 1e-12) * 100
            lines.append(f"  {m.rpm:7.0f}{m.thrust_gf:10.3f}{p:11.3f}{e:8.1f}%")
        lines.append(f"  RMS 誤差 = {self.rms_error_gf:.4f} gf "
                     f"(最大 {self.max_error_pct:.1f} %)")
        return "\n".join(lines)


def _apply(rotor: Rotor, params: dict[str, float]) -> Rotor:
    """較正パラメータを反映した ``Rotor`` を作る."""
    from .airfoil import LinearAirfoil
    from .geometry import PropellerGeometry

    geo = rotor.geometry
    foils = []
    for f in geo.airfoils:
        if not isinstance(f, LinearAirfoil):
            foils.append(f)
            continue
        g = LinearAirfoil(
            name=f.name, cl_alpha=f.cl_alpha,
            alpha0_deg=f.alpha0_deg * params.get("camber", 1.0),
            cl_max=f.cl_max, cl_min=f.cl_min,
            cd0=f.cd0 * params.get("cd_scale", 1.0),
            cd_k=f.cd_k * params.get("cd_scale", 1.0),
            cl_cd0=f.cl_cd0, cm=f.cm, aspect_ratio=f.aspect_ratio,
            reynolds_ref=f.reynolds_ref, reynolds_exponent=f.reynolds_exponent,
            reynolds_lift_slope=params.get("re_lift", f.reynolds_lift_slope),
            reynolds_lift_floor=f.reynolds_lift_floor,
        )
        foils.append(g)
    new_geo = PropellerGeometry(
        radius=geo.radius, n_blades=geo.n_blades, r_R=geo.r_R,
        chord_R=geo.chord_R, twist_deg=geo.twist_deg, airfoils=foils,
        airfoil_index=geo.airfoil_index, hub_radius_ratio=geo.hub_radius_ratio,
        mass=geo.mass, polar_inertia=geo.polar_inertia, name=geo.name,
    )
    return Rotor(
        new_geo, spin=rotor.spin,
        collective_deg=rotor.collective_deg + params.get("collective", 0.0),
        blade_pitch_offsets_deg=rotor.blade_pitch_offsets_deg,
        blade_azimuth_offsets_deg=rotor.blade_azimuth_offsets_deg,
        unbalance=rotor.unbalance,
        include_pitching_moment=rotor.include_pitching_moment,
    )


def _predict(
    rotor: Rotor, model: AeroModel, meas: list[ThrustMeasurement], atmosphere
) -> tuple[np.ndarray, np.ndarray]:
    thrust, torque = [], []
    for m in meas:
        sol = model.solve(rotor, m.operating_point(atmosphere=atmosphere))
        thrust.append(sol.thrust / G * 1e3)
        torque.append(sol.torque(rotor.spin))
    return np.asarray(thrust), np.asarray(torque)


def calibrate(
    rotor: Rotor,
    model: AeroModel,
    measurements,
    *,
    parameters=("collective",),
    bounds: dict[str, tuple[float, float]] | None = None,
    atmosphere=None,
    torque_weight: float = 0.0,
    verbose: bool = False,
) -> CalibrationResult:
    """実測推力にモデルを合わせ込む.

    Parameters
    ----------
    rotor, model:
        較正対象.
    measurements:
        :class:`ThrustMeasurement` のリスト (1 点でも可).
    parameters:
        調整するパラメータ名のタプル. ``"collective"`` / ``"camber"`` /
        ``"re_lift"`` / ``"cd_scale"``.
        **点数より多いパラメータを指定しないこと** (不定になる).
    torque_weight:
        トルク実測がある場合の残差重み (0 なら推力のみ).

    Returns
    -------
    CalibrationResult
        較正後の ``Rotor`` と残差.
    """
    from .atmosphere import SEA_LEVEL

    atmosphere = atmosphere or SEA_LEVEL
    meas = list(measurements)
    names = tuple(parameters)
    for n in names:
        if n not in _PARAMS:
            raise ValueError(f"未知のパラメータ '{n}'. 使えるのは {_PARAMS}")
    if len(names) > len(meas) and torque_weight <= 0.0:
        raise ValueError(
            f"実測 {len(meas)} 点に対してパラメータ {len(names)} 個は"
            "不定です (点数を増やすかパラメータを減らしてください)"
        )

    default = {"collective": 0.0, "camber": 1.0, "re_lift": 0.06, "cd_scale": 1.0}
    lo_hi = {"collective": (-15.0, 10.0), "camber": (0.0, 2.0),
             "re_lift": (0.0, 3.0), "cd_scale": (0.2, 8.0)}
    if bounds:
        lo_hi.update(bounds)
    lo = np.array([lo_hi[n][0] for n in names])
    hi = np.array([lo_hi[n][1] for n in names])
    span = np.maximum(hi - lo, 1e-12)
    # 探索変数は [0, 1] に正規化する. 初期値が厳密に 0 だと scipy の
    # 相対差分ステップが 0 になって勾配が取れないため, この正規化が要る.
    u0 = np.clip((np.array([default[n] for n in names]) - lo) / span, 1e-3, 1 - 1e-3)

    w = np.array([m.weight for m in meas])
    t_meas = np.array([m.thrust_gf for m in meas])
    q_meas = np.array([np.nan if m.torque_nm is None else m.torque_nm
                       for m in meas])

    def residual(u):
        x = lo + np.clip(u, 0.0, 1.0) * span
        p = dict(zip(names, x))
        r = _apply(rotor, p)
        t, q = _predict(r, model, meas, atmosphere)
        res = list(w * (t - t_meas))
        if torque_weight > 0.0:
            ok = np.isfinite(q_meas)
            if np.any(ok):
                res += list(torque_weight * w[ok] * (q[ok] - q_meas[ok]) * 1e3)
        if verbose:  # pragma: no cover
            print("  ", np.round(x, 4), "->", np.round(res, 5), flush=True)
        return np.asarray(res)

    out = least_squares(
        residual, u0, bounds=(0.0, 1.0), xtol=1e-12, ftol=1e-12,
        diff_step=0.02, max_nfev=200,
    )
    p = dict(zip(names, lo + np.clip(out.x, 0.0, 1.0) * span))
    cal = _apply(rotor, p)
    t, _ = _predict(cal, model, meas, atmosphere)
    return CalibrationResult(
        parameters=p, rotor=cal, residual_gf=t - t_meas, measurements=meas,
        predicted_gf=t, success=bool(out.success), message=str(out.message),
    )


def compare_hypotheses(
    rotor: Rotor,
    model: AeroModel,
    measurements,
    *,
    rpm_range=None,
    parameters=("collective", "camber", "re_lift"),
    atmosphere=None,
) -> dict:
    """同じ実測に合う複数の仮説を作り, 外挿の食い違いを見る.

    1 点しか実測がないとき「取付角がずれている」のか「低 Re で性能が
    落ちている」のかは区別できない。どちらでもその点には合うが, 他の
    回転数では予測が食い違う — その幅を返す。

    Returns
    -------
    dict
        ``{"rpm": ..., "curves": {パラメータ名: 推力[gf]}, "spread_pct": ...}``
    """
    meas = list(measurements)
    if rpm_range is None:
        r = np.array([m.rpm for m in meas], dtype=float)
        rpm_range = np.linspace(0.4 * r.min(), 3.0 * r.max(), 12)
    rpm_range = np.asarray(rpm_range, dtype=float)

    curves, results = {}, {}
    for name in parameters:
        res = calibrate(rotor, model, meas, parameters=(name,),
                        atmosphere=atmosphere)
        results[name] = res
        curves[name] = np.array([
            model.solve(res.rotor, ThrustMeasurement(
                rpm=float(n), thrust_gf=1.0).operating_point()).thrust / G * 1e3
            for n in rpm_range
        ])
    stack = np.stack(list(curves.values()))
    spread = (stack.max(axis=0) - stack.min(axis=0)) / np.maximum(
        stack.mean(axis=0), 1e-12) * 100
    return {"rpm": rpm_range, "curves": curves, "spread_pct": spread,
            "results": results}
