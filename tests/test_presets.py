"""プリセット形状 (StampFly 1209) のテスト."""

import numpy as np
import pytest

from prop_sim import BEMT, OperatingPoint, Rotor, get_preset, stampfly_1209
from prop_sim.airfoil import CLARK_Y, FLAT_PLATE, LOW_RE_THIN
from prop_sim.presets import STAMPFLY_1209_MEASURED

G = 9.80665


@pytest.fixture(scope="module")
def rotor():
    return Rotor(stampfly_1209())


@pytest.fixture(scope="module")
def model():
    return BEMT()


# ------------------------------------------------------------------ 形状
def test_geometry_matches_photo_measurement():
    geo = stampfly_1209()
    m = STAMPFLY_1209_MEASURED
    assert geo.diameter * 1e3 == pytest.approx(m["diameter_mm"], rel=1e-6)
    assert geo.n_blades == m["n_blades"]
    assert geo.hub_radius_ratio * geo.diameter * 1e3 == pytest.approx(
        m["hub_outer_diameter_mm"], rel=0.02
    )
    x = np.linspace(geo.hub_radius_ratio, 1.0, 200)
    assert geo.chord(x).max() * 1e3 == pytest.approx(m["max_chord_mm"], rel=0.02)


def test_max_chord_is_inboard_of_midspan():
    """写真の平面形は r/R = 0.55 付近が最大コード."""
    geo = stampfly_1209()
    x = np.linspace(geo.hub_radius_ratio, 1.0, 400)
    assert 0.50 < x[np.argmax(geo.chord(x))] < 0.62


def test_solidity_is_high_for_four_blades():
    geo = stampfly_1209()
    assert 0.35 < geo.solidity < 0.50
    assert 0.25 < geo.blade_area() / geo.disk_area < 0.40


def test_nominal_pitch_is_recovered():
    geo = stampfly_1209(pitch_in=0.9)
    assert geo.pitch_at(0.75) / 0.0254 == pytest.approx(0.9, rel=0.02)
    assert geo.pitch_at(0.5) / 0.0254 == pytest.approx(0.9, rel=0.02)


def test_pitch_distribution_options_differ():
    base = stampfly_1209()
    wash = stampfly_1209(pitch_distribution="washout")
    const = stampfly_1209(pitch_distribution="constant")
    assert wash.twist(np.array(1.0)) < base.twist(np.array(1.0))
    assert const.twist(np.array(0.3)) < base.twist(np.array(0.3))
    with pytest.raises(ValueError):
        stampfly_1209(pitch_distribution="unknown")


def test_registry_lookup():
    assert get_preset("stampfly_1209").n_blades == 4
    assert get_preset("stampfly-1209").n_blades == 4
    with pytest.raises(KeyError):
        get_preset("no_such_prop")


def test_inertia_consistent_with_measured_mass_distribution():
    geo = stampfly_1209()
    coef = geo.polar_inertia / (geo.mass * geo.radius**2)
    assert 0.20 < coef < 0.40


# ------------------------------------------------------------------ 空力
def test_static_thrust_is_in_realistic_range(rotor, model):
    """31 mm 4 枚は 30000 rpm で 1 発 5 - 15 gf 程度."""
    t = model.solve(rotor, OperatingPoint(rpm=30000)).thrust / G * 1e3
    assert 6.0 < t < 16.0


def test_reynolds_number_is_low(rotor, model):
    """このサイズでは 0.75R でも Re は 2 万を切る."""
    sol = model.solve(rotor, OperatingPoint(rpm=30000))
    st = sol.sections
    k = int(np.argmin(np.abs(st.r_R - 0.75)))
    re75 = float(st.reynolds[:, k].mean())
    assert 5.0e3 < re75 < 2.0e4


def test_thrust_follows_rpm_squared(rotor, model):
    t1 = model.solve(rotor, OperatingPoint(rpm=15000)).thrust
    t2 = model.solve(rotor, OperatingPoint(rpm=30000)).thrust
    assert t2 / t1 == pytest.approx(4.0, rel=0.10)


def test_hover_rpm_for_a_36g_quadrotor(rotor, model):
    """全備 36.8 g の 4 発機なら 2.5 万 - 4 万 rpm に収まる."""
    from scipy.optimize import brentq

    def f(n):
        return model.solve(rotor, OperatingPoint(rpm=n)).thrust / G * 1e3 - 36.8 / 4

    n = brentq(f, 5000.0, 80000.0, xtol=10.0)
    assert 24000.0 < n < 38000.0


def test_low_re_airfoil_reduces_performance():
    """低 Re 翼型は高 Re 翼型より推力・効率とも低く出る."""
    model = BEMT()
    op = OperatingPoint(rpm=30000)
    low = model.solve(Rotor(stampfly_1209(airfoil=LOW_RE_THIN)), op)
    high = model.solve(Rotor(stampfly_1209(airfoil=CLARK_Y)), op)
    flat = model.solve(Rotor(stampfly_1209(airfoil=FLAT_PLATE)), op)
    assert low.thrust < high.thrust
    assert flat.thrust < low.thrust
    r = Rotor(stampfly_1209())
    assert low.coefficients(r, op)["FM"] < high.coefficients(r, op)["FM"]


def test_higher_pitch_needs_lower_hover_rpm(model):
    op = OperatingPoint(rpm=30000)
    t08 = model.solve(Rotor(stampfly_1209(pitch_in=0.8)), op).thrust
    t10 = model.solve(Rotor(stampfly_1209(pitch_in=1.0)), op).thrust
    assert t08 < t10


def test_edgewise_flight_increases_thrust(rotor, model):
    """並進揚力: 完全な横流れでは推力が増える."""
    hover = model.solve(rotor, OperatingPoint(rpm=31000)).thrust
    side = model.solve(
        rotor, OperatingPoint(rpm=31000, v_inf=8.0, inflow_angle_deg=90.0)
    ).thrust
    climb = model.solve(
        rotor, OperatingPoint(rpm=31000, v_inf=8.0, inflow_angle_deg=0.0)
    ).thrust
    assert side > hover > climb


def test_gyroscopic_moment_dominates_aero_torque_at_high_body_rate(rotor, model):
    """小さなプロペラでは機体の角速度が上がるとジャイロが空力トルクを超える."""
    rpm, q = 31000.0, 10.0
    op = OperatingPoint(rpm=rpm, body_rate=np.array([0.0, q, 0.0]))
    sol = model.solve(rotor, op)
    w = rotor.inertial_wrench(op)
    jz = rotor.geometry.polar_inertia
    assert abs(w.moment[0]) == pytest.approx(jz * op.omega * q, rel=1e-9)
    assert abs(w.moment[0]) > sol.torque(rotor.spin)


def test_blade_passage_frequency_is_high(rotor):
    """4 枚 x 3 万 rpm は BPF 2 kHz — 6 分力計の共振帯に入る."""
    bpf = rotor.n_blades * 31000.0 / 60.0
    assert bpf == pytest.approx(2067.0, rel=0.01)
