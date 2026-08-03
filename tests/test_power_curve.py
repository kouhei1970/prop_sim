"""回転数を含まない試験データ (推力 + 電気入力) の扱いのテスト."""

import numpy as np
import pytest

from prop_sim import (
    BEMT,
    PowerCurvePoint,
    Rotor,
    consistency_check,
    ideal_power,
    stampfly_1209,
)
from prop_sim.data import GEMFAN_1209_4, gemfan_1209_4_points

G = 9.80665


@pytest.fixture(scope="module")
def rotor():
    return Rotor(stampfly_1209())


@pytest.fixture(scope="module")
def model():
    return BEMT()


# ------------------------------------------------------- PowerCurvePoint
def test_point_computes_power_from_voltage_and_current():
    p = PowerCurvePoint(thrust_gf=10.0, voltage=3.7, current=2.0)
    assert p.power_w == pytest.approx(7.4)
    assert p.thrust_n == pytest.approx(10.0 * G * 1e-3)


def test_point_requires_power_information():
    with pytest.raises(ValueError):
        PowerCurvePoint(thrust_gf=10.0)


# ------------------------------------------------------------ ideal_power
def test_ideal_power_matches_momentum_theory(rotor):
    """P_ideal = T v_h = T sqrt(T/(2 rho A))."""
    t = 0.1
    a = rotor.geometry.disk_area
    assert ideal_power(t, rotor) == pytest.approx(t * np.sqrt(t / (2 * 1.225 * a)))


def test_ideal_power_scales_as_thrust_to_the_three_halves(rotor):
    assert ideal_power(0.4, rotor) / ideal_power(0.1, rotor) == pytest.approx(
        4.0**1.5, rel=1e-9
    )


# ------------------------------------------------------ consistency_check
def test_consistency_check_recovers_a_synthetic_case(rotor, model):
    """モータ効率 0.5 の合成データからその効率を復元できる."""
    from prop_sim import OperatingPoint

    pts, rpms = [], (12000.0, 25000.0, 38000.0)
    for n in rpms:
        op = OperatingPoint(rpm=n)
        sol = model.solve(rotor, op)
        pts.append(PowerCurvePoint(
            thrust_gf=sol.thrust / G * 1e3, power_w=sol.power(op) / 0.5))
    out = consistency_check(rotor, model, pts)
    assert np.allclose(out["motor_efficiency"], 0.5, atol=1e-3)
    assert np.allclose(out["rpm"], rpms, rtol=2e-3)
    assert not out["falsified"]


def test_impossible_efficiency_is_flagged(rotor, model):
    """必要な軸動力が電気入力を超えたら反証扱いになる."""
    from prop_sim import OperatingPoint

    op = OperatingPoint(rpm=25000.0)
    sol = model.solve(rotor, op)
    pts = [PowerCurvePoint(thrust_gf=sol.thrust / G * 1e3,
                           power_w=0.5 * sol.power(op))]
    out = consistency_check(rotor, model, pts)
    assert out["falsified"]
    assert out["motor_efficiency"][0] > 1.0


def test_fm_times_eta_is_independent_of_the_model(rotor, model):
    """FM x eta は実測だけで決まり, どのモデルを使っても同じ値になる."""
    pts = gemfan_1209_4_points()
    a = consistency_check(rotor, model, pts)["fm_times_eta"]
    b = consistency_check(Rotor(stampfly_1209(pitch_in=1.3)), model,
                          pts)["fm_times_eta"]
    assert np.allclose(a, b)


# ------------------------------------------------------- Gemfan 参照データ
def test_gemfan_dataset_shape():
    d = GEMFAN_1209_4
    n = d["throttle_pct"].size
    assert n == 10
    assert d["current_a"].size == n and d["thrust_g"].size == n
    assert d["rpm"] is None          # 回転数は公表されていない


def test_gemfan_thrust_column_and_efficiency_column_disagree():
    """データ表の内部矛盾を明示的に記録しておく."""
    d = GEMFAN_1209_4
    p = d["voltage_v"] * d["current_a"]
    reconstructed = d["thrust_efficiency_g_per_w"] * p
    rel = np.abs(reconstructed - d["thrust_g"]) / d["thrust_g"]
    assert rel.max() > 0.3           # 10% スロットルで 38% ずれる
    assert rel[-1] < 0.05            # 全開付近は 2% 程度で一致


@pytest.mark.parametrize("source", ["thrust_column", "efficiency"])
def test_gemfan_points_are_monotone(source):
    pts = gemfan_1209_4_points(source)
    t = np.array([p.thrust_gf for p in pts])
    p_w = np.array([p.power_w for p in pts])
    assert np.all(np.diff(t) > 0) and np.all(np.diff(p_w) > 0)


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError):
        gemfan_1209_4_points("nonsense")


def test_gemfan_data_does_not_falsify_the_model(rotor, model):
    """要求されるモータ効率が 1 を超えない = このデータでは反証できない."""
    out = consistency_check(rotor, model, gemfan_1209_4_points())
    assert not out["falsified"]
    assert 0.25 < out["max_efficiency"] < 0.60     # 1S 微小モータとして妥当


def test_gemfan_data_cannot_separate_the_hypotheses(rotor, model):
    """回転数が無いので取付角仮説と低 Re 仮説のどちらも生き残る."""
    from prop_sim import ThrustMeasurement, compare_hypotheses

    cal = compare_hypotheses(
        rotor, model, [ThrustMeasurement(omega=1209.0, thrust_gf=1.0)],
        rpm_range=np.array([11545.0]), parameters=("collective", "re_lift"),
    )
    pts = gemfan_1209_4_points()
    for r in cal["results"].values():
        out = consistency_check(r.rotor, model, pts)
        assert not out["falsified"]
        assert out["max_efficiency"] < 0.6


def test_low_re_hypothesis_barely_changes_full_throttle(rotor, model):
    """低 Re 仮説は高回転では効かないので, 全開の回転数は未較正とほぼ同じ."""
    from prop_sim import ThrustMeasurement, compare_hypotheses

    cal = compare_hypotheses(
        rotor, model, [ThrustMeasurement(omega=1209.0, thrust_gf=1.0)],
        rpm_range=np.array([11545.0]), parameters=("collective", "re_lift"),
    )
    pts = gemfan_1209_4_points()
    base = consistency_check(rotor, model, pts)["rpm"][-1]
    low_re = consistency_check(cal["results"]["re_lift"].rotor, model, pts)["rpm"][-1]
    coll = consistency_check(cal["results"]["collective"].rotor, model, pts)["rpm"][-1]
    assert abs(low_re / base - 1.0) < 0.05        # ほぼ同じ
    assert coll / base > 1.20                     # 取付角仮説は 2 割以上高い
