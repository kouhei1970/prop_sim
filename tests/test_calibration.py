"""実測データによる較正のテスト."""

import numpy as np
import pytest

from prop_sim import (
    BEMT,
    OperatingPoint,
    Rotor,
    ThrustMeasurement,
    calibrate,
    compare_hypotheses,
    stampfly_1209,
)

G = 9.80665


@pytest.fixture(scope="module")
def rotor():
    return Rotor(stampfly_1209())


@pytest.fixture(scope="module")
def model():
    return BEMT()


def thrust_gf(rotor, model, rpm):
    return model.solve(rotor, OperatingPoint(rpm=rpm)).thrust / G * 1e3


# ------------------------------------------------------- ThrustMeasurement
def test_measurement_converts_omega_and_grams():
    m = ThrustMeasurement(omega=1209.0, thrust_gf=1.0)
    assert m.rpm == pytest.approx(1209.0 * 60 / (2 * np.pi))
    assert m.thrust_n == pytest.approx(1.0 * G * 1e-3)


def test_measurement_accepts_si_units():
    m = ThrustMeasurement(rpm=6000.0, thrust_n=0.05)
    assert m.thrust_gf == pytest.approx(0.05 / G * 1e3)


def test_measurement_requires_speed_and_thrust():
    with pytest.raises(ValueError):
        ThrustMeasurement(thrust_gf=1.0)
    with pytest.raises(ValueError):
        ThrustMeasurement(rpm=6000.0)


# --------------------------------------------------------------- calibrate
def test_calibration_hits_a_single_measurement(rotor, model):
    meas = [ThrustMeasurement(omega=1209.0, thrust_gf=1.0)]
    res = calibrate(rotor, model, meas, parameters=("collective",))
    assert res.rms_error_gf < 1e-3
    assert res.parameters["collective"] < 0.0     # モデルは過大なので下げる向き


def test_calibrated_rotor_reproduces_the_point(rotor, model):
    meas = [ThrustMeasurement(rpm=20000.0, thrust_gf=3.0)]
    res = calibrate(rotor, model, meas, parameters=("collective",))
    assert thrust_gf(res.rotor, model, 20000.0) == pytest.approx(3.0, abs=2e-3)


def test_calibration_recovers_a_known_offset(rotor, model):
    """既知のオフセットを入れた合成データから元の値を復元できる."""
    truth = Rotor(stampfly_1209(), collective_deg=-2.5)
    meas = [ThrustMeasurement(rpm=r, thrust_gf=thrust_gf(truth, model, r))
            for r in (12000.0, 25000.0)]
    res = calibrate(rotor, model, meas, parameters=("collective",))
    assert res.parameters["collective"] == pytest.approx(-2.5, abs=0.05)


def test_two_parameter_calibration_needs_two_points(rotor, model):
    meas = [ThrustMeasurement(rpm=12000.0, thrust_gf=1.2)]
    with pytest.raises(ValueError):
        calibrate(rotor, model, meas, parameters=("collective", "re_lift"))


def test_joint_calibration_recovers_both_parameters(rotor, model):
    """3 点あれば取付角と Re 依存性を同時に同定できる."""
    from prop_sim.calibration import _apply

    truth = _apply(rotor, {"collective": -1.5, "re_lift": 0.6})
    meas = [ThrustMeasurement(rpm=r, thrust_gf=thrust_gf(truth, model, r))
            for r in (11545.0, 22000.0, 32000.0)]
    res = calibrate(rotor, model, meas, parameters=("collective", "re_lift"))
    assert res.parameters["collective"] == pytest.approx(-1.5, abs=0.2)
    assert res.parameters["re_lift"] == pytest.approx(0.6, abs=0.08)
    assert res.rms_error_gf < 5e-3


def test_unknown_parameter_is_rejected(rotor, model):
    meas = [ThrustMeasurement(rpm=12000.0, thrust_gf=1.2)]
    with pytest.raises(ValueError):
        calibrate(rotor, model, meas, parameters=("no_such_knob",))


def test_calibration_leaves_geometry_untouched(rotor, model):
    """較正は翼型と取付角だけを変え, 形状は変えない."""
    meas = [ThrustMeasurement(omega=1209.0, thrust_gf=1.0)]
    res = calibrate(rotor, model, meas, parameters=("collective",))
    assert res.rotor.geometry.radius == rotor.geometry.radius
    assert np.allclose(res.rotor.geometry.chord_R, rotor.geometry.chord_R)
    assert np.allclose(res.rotor.geometry.twist_deg, rotor.geometry.twist_deg)
    assert rotor.collective_deg == 0.0            # 元の Rotor は不変


def test_report_is_readable(rotor, model):
    meas = [ThrustMeasurement(omega=1209.0, thrust_gf=1.0)]
    txt = calibrate(rotor, model, meas, parameters=("collective",)).report()
    assert "較正結果" in txt and "RMS" in txt


# -------------------------------------------------------- 仮説の切り分け
def test_single_point_cannot_separate_hypotheses(rotor, model):
    """1 点では取付角仮説と Re 仮説が区別できず, 外挿が大きく食い違う."""
    meas = [ThrustMeasurement(omega=1209.0, thrust_gf=1.0)]
    out = compare_hypotheses(
        rotor, model, meas, rpm_range=np.array([11545.0, 30000.0]),
        parameters=("collective", "re_lift"),
    )
    # 較正点ではどちらもぴたりと合う
    assert out["spread_pct"][0] < 1.0
    # 3 万 rpm では 3 割以上食い違う
    assert out["spread_pct"][1] > 25.0


def test_reynolds_lift_slope_affects_low_rpm_more(rotor, model):
    """Re 依存を強めると低回転ほど推力が落ちる (回転数依存性がある)."""
    from prop_sim.calibration import _apply

    strong = _apply(rotor, {"re_lift": 1.0})
    lo = thrust_gf(strong, model, 8000.0) / thrust_gf(rotor, model, 8000.0)
    hi = thrust_gf(strong, model, 32000.0) / thrust_gf(rotor, model, 32000.0)
    assert lo < hi < 1.0


def test_collective_offset_is_nearly_rpm_independent(rotor, model):
    """取付角のずれは Ct をほぼ一定倍にする (回転数依存性がない)."""
    from prop_sim.calibration import _apply

    shifted = _apply(rotor, {"collective": -3.0})
    lo = thrust_gf(shifted, model, 8000.0) / thrust_gf(rotor, model, 8000.0)
    hi = thrust_gf(shifted, model, 32000.0) / thrust_gf(rotor, model, 32000.0)
    assert abs(lo - hi) < 0.05
