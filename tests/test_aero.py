"""空力モデル (BEMT / BET / 代理モデル) の物理妥当性テスト."""

import numpy as np
import pytest

from prop_sim import (
    BEMT,
    BET,
    OperatingPoint,
    QuadraticModel,
    Rotor,
    from_diameter_pitch,
    get_model,
)


@pytest.fixture(scope="module")
def rotor():
    return Rotor(from_diameter_pitch(10.0, 4.7))


@pytest.fixture(scope="module")
def model():
    return BEMT()


# ---------------------------------------------------------------- ホバリング
def test_hover_produces_positive_thrust_and_torque(rotor, model):
    sol = model.solve(rotor, OperatingPoint(rpm=6000))
    assert sol.thrust > 0.0
    assert sol.torque(rotor.spin) > 0.0
    # 反トルクはハブ Mz として負に現れる (spin = +1)
    assert sol.wrench.moment[2] < 0.0


def test_hover_induced_velocity_matches_momentum_theory(rotor, model):
    """v_i は運動量理論の v_h = sqrt(T/(2 rho A)) と同オーダになる."""
    op = OperatingPoint(rpm=6000)
    sol = model.solve(rotor, op)
    v_h = np.sqrt(sol.thrust / (2.0 * op.atmosphere.density * rotor.geometry.disk_area))
    assert sol.inflow.mean_v0() == pytest.approx(v_h, rel=0.20)


def test_thrust_scales_with_rpm_squared(rotor, model):
    t1 = model.solve(rotor, OperatingPoint(rpm=4000)).thrust
    t2 = model.solve(rotor, OperatingPoint(rpm=8000)).thrust
    assert t2 / t1 == pytest.approx(4.0, rel=0.10)


def test_static_thrust_coefficient_is_in_realistic_range(rotor, model):
    cf = model.solve(rotor, OperatingPoint(rpm=6000)).coefficients(
        rotor, OperatingPoint(rpm=6000)
    )
    assert 0.05 < cf["Ct"] < 0.20
    assert 0.02 < cf["Cp"] < 0.10
    assert 0.3 < cf["FM"] < 0.85


def test_zero_rpm_gives_zero_load(rotor, model):
    sol = model.solve(rotor, OperatingPoint(rpm=0.0))
    assert np.allclose(sol.wrench.as_array(), 0.0, atol=1e-9)


# --------------------------------------------------------------------- 軸流
def test_thrust_decreases_with_advance_ratio(rotor, model):
    n = 6000 / 60.0
    d = rotor.geometry.diameter
    thrust = [
        model.solve(rotor, OperatingPoint(rpm=6000, v_inf=j * n * d)).thrust
        for j in (0.0, 0.2, 0.4, 0.6)
    ]
    assert all(np.diff(thrust) < 0.0)


def test_zero_thrust_advance_ratio_near_geometric_pitch(rotor, model):
    """推力ゼロの前進率は幾何ピッチ比 P/D の 0.6-1.2 倍に入る."""
    n = 6000 / 60.0
    d = rotor.geometry.diameter
    p_d = rotor.geometry.pitch_at(0.75) / d
    js = np.linspace(0.1, 1.4, 27)
    thrust = np.array(
        [model.solve(rotor, OperatingPoint(rpm=6000, v_inf=j * n * d)).thrust for j in js]
    )
    assert thrust[0] > 0.0 and thrust[-1] < 0.0
    j0 = float(np.interp(0.0, thrust[::-1], js[::-1]))
    assert 0.6 * p_d < j0 < 1.2 * p_d


def test_efficiency_peaks_below_zero_thrust_condition(rotor, model):
    n = 6000 / 60.0
    d = rotor.geometry.diameter
    eta = []
    for j in np.linspace(0.1, 0.8, 15):
        op = OperatingPoint(rpm=6000, v_inf=j * n * d)
        sol = model.solve(rotor, op)
        cf = sol.coefficients(rotor, op)
        # 推力が正の領域だけが「プロペラとして」意味のある効率
        eta.append(cf["eta"] if sol.thrust > 0.0 and cf["Cp"] > 0.0 else np.nan)
    eta = np.array(eta)
    assert 0.0 < np.nanmax(eta) < 0.95
    assert 0 < int(np.nanargmax(eta)) < np.count_nonzero(np.isfinite(eta)) - 1


# ----------------------------------------------------------------- 斜め流入
def test_oblique_inflow_generates_all_six_components(rotor, model):
    op = OperatingPoint(rpm=6000, v_inf=10.0, inflow_angle_deg=35.0)
    w = model.solve(rotor, op).wrench
    arr = np.abs(w.as_array())
    assert np.all(arr > 0.0), "斜め流入では 6 成分すべてが非ゼロになるはず"


def test_oblique_inflow_in_plane_force_opposes_motion(rotor, model):
    """面内力 (H 力) は機体の進行方向と逆向き."""
    op = OperatingPoint(rpm=6000, v_inf=12.0, inflow_angle_deg=60.0,
                        inflow_azimuth_deg=0.0)
    w = model.solve(rotor, op).wrench
    assert w.force[0] < 0.0


def test_oblique_inflow_hub_moment_grows_with_speed(rotor, model):
    mags = []
    for v in (0.0, 5.0, 10.0, 15.0):
        op = OperatingPoint(rpm=6000, v_inf=v, inflow_angle_deg=60.0)
        w = model.solve(rotor, op).wrench
        mags.append(np.hypot(w.moment[0], w.moment[1]))
    assert all(np.diff(mags) > 0.0)


def test_axial_flow_has_no_in_plane_load(rotor, model):
    op = OperatingPoint(rpm=6000, v_inf=8.0, inflow_angle_deg=0.0)
    w = model.solve(rotor, op).wrench
    assert np.allclose(w.force[:2], 0.0, atol=1e-9)
    assert np.allclose(w.moment[:2], 0.0, atol=1e-9)


def test_inflow_azimuth_rotates_the_load_vector(rotor, model):
    w0 = model.solve(
        rotor, OperatingPoint(rpm=6000, v_inf=10.0, inflow_angle_deg=50.0,
                              inflow_azimuth_deg=0.0)
    ).wrench
    w90 = model.solve(
        rotor, OperatingPoint(rpm=6000, v_inf=10.0, inflow_angle_deg=50.0,
                              inflow_azimuth_deg=90.0)
    ).wrench
    assert w0.force[2] == pytest.approx(w90.force[2], rel=1e-6)
    # 流入方位を 90 deg 回すと面内力も 90 deg 回る
    assert w90.force[1] == pytest.approx(w0.force[0], rel=1e-3, abs=1e-4)
    assert abs(w90.force[0]) < 0.2 * abs(w0.force[0])


# ------------------------------------------------------------------ 回転方向
def test_reversed_spin_flips_torque_but_not_thrust():
    geo = from_diameter_pitch(10.0, 4.7)
    ccw, cw = Rotor(geo, spin=1), Rotor(geo, spin=-1)
    model = BEMT()
    op = OperatingPoint(rpm=6000)
    a = model.solve(ccw, op).wrench
    b = model.solve(cw, op).wrench
    assert a.force[2] == pytest.approx(b.force[2], rel=1e-9)
    assert a.moment[2] == pytest.approx(-b.moment[2], rel=1e-9)


def test_reversed_spin_mirrors_hub_moment_in_edgewise_flow():
    geo = from_diameter_pitch(10.0, 4.7)
    model = BEMT()
    op = OperatingPoint(rpm=6000, v_inf=12.0, inflow_angle_deg=70.0)
    a = model.solve(Rotor(geo, spin=1), op).wrench
    b = model.solve(Rotor(geo, spin=-1), op).wrench
    # 進行方向は同じなので My (面内速度方向のモーメント) は一致し,
    # 前進/後退側の入れ替わりで Mx が反転する
    assert a.moment[1] == pytest.approx(b.moment[1], rel=0.05)
    assert a.moment[0] == pytest.approx(-b.moment[0], rel=0.05)


# ------------------------------------------------------ ブレード枚数・ピッチ
def test_more_blades_more_thrust():
    model = BEMT()
    op = OperatingPoint(rpm=6000)
    t2 = model.solve(Rotor(from_diameter_pitch(10, 4.7, n_blades=2)), op).thrust
    t3 = model.solve(Rotor(from_diameter_pitch(10, 4.7, n_blades=3)), op).thrust
    assert t3 > t2
    assert t3 < 1.5 * t2      # 誘導損失で比例以下になる


def test_higher_pitch_more_thrust():
    model = BEMT()
    op = OperatingPoint(rpm=6000)
    lo = model.solve(Rotor(from_diameter_pitch(10, 3.8)), op).thrust
    hi = model.solve(Rotor(from_diameter_pitch(10, 6.0)), op).thrust
    assert hi > lo


def test_collective_pitch_increases_thrust(rotor, model):
    base = model.solve(rotor, OperatingPoint(rpm=6000)).thrust
    up = Rotor(rotor.geometry, collective_deg=2.0)
    assert model.solve(up, OperatingPoint(rpm=6000)).thrust > base


# ------------------------------------------------------------ モデル間の比較
def test_bet_and_bemt_agree_within_engineering_tolerance(rotor):
    op = OperatingPoint(rpm=6000)
    t_bemt = BEMT().solve(rotor, op).thrust
    t_bet = BET().solve(rotor, op).thrust
    assert t_bet == pytest.approx(t_bemt, rel=0.35)
    assert t_bet > t_bemt      # 翼端損失を無視する分だけ過大評価


def test_tip_loss_reduces_thrust(rotor):
    op = OperatingPoint(rpm=6000)
    with_loss = BEMT(tip_loss=True, hub_loss=False).solve(rotor, op).thrust
    without = BEMT(tip_loss=False, hub_loss=False).solve(rotor, op).thrust
    assert with_loss < without


def test_quadratic_surrogate_reproduces_bemt(rotor):
    ref = BEMT()
    surrogate = QuadraticModel.fit(
        rotor, ref, rpm=6000.0, advance_ratios=np.linspace(0.0, 0.6, 7)
    )
    for j in (0.0, 0.3, 0.5):
        op = OperatingPoint(rpm=6000, v_inf=j * 100.0 * rotor.geometry.diameter)
        op = OperatingPoint(rpm=6000, v_inf=j * (6000 / 60) * rotor.geometry.diameter)
        t_ref = ref.solve(rotor, op).thrust
        t_sur = surrogate.solve(rotor, op).thrust
        assert t_sur == pytest.approx(t_ref, abs=0.15 * max(abs(t_ref), 1.0))


def test_registry_returns_requested_model():
    assert isinstance(get_model("bemt"), BEMT)
    assert isinstance(get_model("quadratic"), QuadraticModel)
    with pytest.raises(KeyError):
        get_model("no-such-model")


# ------------------------------------------------------------------ 数値健全性
def test_solution_is_deterministic(rotor, model):
    op = OperatingPoint(rpm=7000, v_inf=6.0, inflow_angle_deg=25.0)
    a = model.solve(rotor, op).wrench.as_array()
    b = model.solve(rotor, op).wrench.as_array()
    assert np.allclose(a, b)


def test_grid_refinement_changes_thrust_only_slightly(rotor):
    op = OperatingPoint(rpm=6000)
    coarse = BEMT(n_radial=16).solve(rotor, op).thrust
    fine = BEMT(n_radial=48).solve(rotor, op).thrust
    assert coarse == pytest.approx(fine, rel=0.03)


def test_air_density_scales_thrust(rotor, model):
    from prop_sim.atmosphere import Atmosphere

    sea = model.solve(rotor, OperatingPoint(rpm=6000)).thrust
    alt = model.solve(
        rotor, OperatingPoint(rpm=6000, atmosphere=Atmosphere(altitude_m=3000))
    ).thrust
    ratio = Atmosphere(altitude_m=3000).density / Atmosphere().density
    assert alt / sea == pytest.approx(ratio, rel=0.05)
