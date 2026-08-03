"""Lv3 (揚力線 + 渦後流) と Lv4/LvS (データ駆動) のテスト."""

import numpy as np
import pytest

from prop_sim import (
    BEMT,
    LiftingLine,
    OperatingPoint,
    Rotor,
    SurrogateModel,
    from_diameter_pitch,
    get_model,
)
from prop_sim.models.vortex import biot_savart


@pytest.fixture(scope="module")
def rotor():
    return Rotor(from_diameter_pitch(10.0, 4.7))


@pytest.fixture(scope="module")
def bemt():
    return BEMT()


@pytest.fixture(scope="module")
def hover_ll(rotor):
    return LiftingLine().solve(rotor, OperatingPoint(rpm=6000))


# ======================================================== Biot-Savart 則
def test_biot_savart_matches_infinite_line_vortex():
    """十分長い直線渦は v = Gamma / (2 pi d) を与える."""
    gamma, d, half = 3.0, 0.25, 4000.0
    a = np.array([[0.0, 0.0, -half]])
    b = np.array([[0.0, 0.0, +half]])
    v = biot_savart(np.array([[d, 0.0, 0.0]]), a, b, np.array([gamma]), core=1e-9)
    assert np.linalg.norm(v[0]) == pytest.approx(gamma / (2 * np.pi * d), rel=1e-3)
    # +z 向きの渦は +x の点で +y 方向 (右ねじ, v = Gamma/(2 pi d) * (z_hat x r_hat))
    assert v[0, 1] == pytest.approx(gamma / (2 * np.pi * d), rel=1e-3)
    assert abs(v[0, 0]) < 1e-6 and abs(v[0, 2]) < 1e-6


def test_biot_savart_is_linear_in_circulation():
    a, b = np.array([[0.0, 0.0, -1.0]]), np.array([[0.0, 0.0, 1.0]])
    p = np.array([[0.3, 0.1, 0.0]])
    v1 = biot_savart(p, a, b, np.array([1.0]), core=1e-6)
    v2 = biot_savart(p, a, b, np.array([2.5]), core=1e-6)
    assert np.allclose(v2, 2.5 * v1)


def test_biot_savart_on_the_filament_is_finite():
    a, b = np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
    v = biot_savart(np.array([[0.5, 0.0, 0.0]]), a, b, np.array([1.0]), core=0.01)
    assert np.all(np.isfinite(v))


def test_core_radius_limits_induced_velocity():
    a, b = np.array([[0.0, 0.0, -50.0]]), np.array([[0.0, 0.0, 50.0]])
    p = np.array([[1e-3, 0.0, 0.0]])
    small = np.linalg.norm(biot_savart(p, a, b, np.array([1.0]), core=1e-4))
    large = np.linalg.norm(biot_savart(p, a, b, np.array([1.0]), core=0.1))
    assert large < small


# =========================================================== Lv3 揚力線
def test_lifting_line_rejects_oblique_inflow(rotor):
    with pytest.raises(NotImplementedError):
        LiftingLine().solve(
            rotor, OperatingPoint(rpm=6000, v_inf=5.0, inflow_angle_deg=30.0)
        )


def test_lifting_line_converges_in_hover(hover_ll):
    assert hover_ll.converged
    assert hover_ll.warnings == ()


def test_lifting_line_agrees_with_bemt_when_swirl_is_neglected(rotor, bemt):
    """BEMT の残る近似 (アニュラス独立 + Prandtl 損失) は 2 % 以内."""
    op = OperatingPoint(rpm=6000)
    ref = bemt.solve(rotor, op).thrust
    ll = LiftingLine(include_swirl=False).solve(rotor, op).thrust
    assert ll == pytest.approx(ref, rel=0.02)


def test_swirl_reduces_thrust(rotor):
    """後流の旋回は相対速度を減らすので推力を下げる."""
    op = OperatingPoint(rpm=6000)
    with_swirl = LiftingLine(include_swirl=True).solve(rotor, op).thrust
    without = LiftingLine(include_swirl=False).solve(rotor, op).thrust
    assert with_swirl < without
    assert with_swirl > 0.9 * without          # 差は数 % のオーダ


def test_lifting_line_induced_velocity_is_physical(hover_ll, rotor):
    """誘導速度は正で, 運動量理論の値と同オーダ."""
    vi = hover_ll.inflow.mean_v0()
    v_h = np.sqrt(hover_ll.thrust / (2 * 1.225 * rotor.geometry.disk_area))
    assert vi > 0.0
    assert vi == pytest.approx(v_h, rel=0.25)


def test_lifting_line_circulation_vanishes_at_the_tip(rotor):
    m = LiftingLine()
    m.solve(rotor, OperatingPoint(rpm=6000))
    g = m.circulation["gamma"]
    assert np.all(g[:-1] > 0.0)
    assert g[-1] < 0.5 * g.max()               # 翼端で循環が落ちる


def test_lifting_line_thrust_scales_with_rpm_squared(rotor):
    m = LiftingLine()
    t1 = m.solve(rotor, OperatingPoint(rpm=4000)).thrust
    t2 = m.solve(rotor, OperatingPoint(rpm=8000)).thrust
    assert t2 / t1 == pytest.approx(4.0, rel=0.10)


def test_lifting_line_zero_rpm(rotor):
    sol = LiftingLine().solve(rotor, OperatingPoint(rpm=0.0))
    assert np.allclose(sol.wrench.as_array(), 0.0, atol=1e-9)


def test_lifting_line_axial_flow_has_no_in_plane_load(hover_ll):
    w = hover_ll.wrench
    assert np.allclose(w.force[:2], 0.0, atol=1e-9)
    assert np.allclose(w.moment[:2], 0.0, atol=1e-9)


def test_lifting_line_thrust_falls_with_advance_ratio(rotor):
    m = LiftingLine()
    n, d = 100.0, rotor.geometry.diameter
    t = [m.solve(rotor, OperatingPoint(rpm=6000, v_inf=j * n * d)).thrust
         for j in (0.0, 0.2, 0.4)]
    assert all(np.diff(t) < 0.0)


def test_wake_length_convergence(rotor):
    """後流を伸ばしても解は数 % 以内に収まる."""
    op = OperatingPoint(rpm=6000)
    short = LiftingLine(n_turns=4).solve(rotor, op).thrust
    long_ = LiftingLine(n_turns=8).solve(rotor, op).thrust
    assert short == pytest.approx(long_, rel=0.05)


# ==================================================== Lv4 / LvS 代理モデル
@pytest.fixture(scope="module")
def surrogate(rotor, bemt):
    return SurrogateModel.fit(
        rotor, bemt, rpm=[4000.0, 8000.0],
        advance_ratios=np.linspace(0.0, 0.6, 5),
        inflow_angles_deg=np.array([0.0, 30.0, 60.0]),
    )


def test_surrogate_reproduces_reference_model(rotor, bemt, surrogate):
    for rpm, v, ang in ((6000, 0.0, 0.0), (6000, 6.0, 0.0),
                        (5000, 8.0, 30.0), (7500, 10.0, 50.0)):
        op = OperatingPoint(rpm=rpm, v_inf=v, inflow_angle_deg=ang)
        ref = bemt.solve(rotor, op).wrench
        got = surrogate.solve(rotor, op).wrench
        assert got.force[2] == pytest.approx(ref.force[2], rel=0.05, abs=0.05)
        assert got.moment[2] == pytest.approx(ref.moment[2], rel=0.05, abs=1e-3)


def test_surrogate_enforces_axial_symmetry(rotor, surrogate):
    """軸流では面内成分が厳密に 0 (多項式基底に対称性を課している)."""
    w = surrogate.solve(rotor, OperatingPoint(rpm=6000)).wrench
    assert np.allclose(w.force[:2], 0.0, atol=1e-12)
    assert np.allclose(w.moment[:2], 0.0, atol=1e-12)


def test_surrogate_rpm_scaling_is_exact(rotor, surrogate):
    """係数空間で扱うので rho n^2 D^4 のスケーリングは厳密."""
    a = surrogate.solve(rotor, OperatingPoint(rpm=6000)).thrust
    b = surrogate.solve(rotor, OperatingPoint(rpm=12000)).thrust
    # Reynolds 補正 (log n の項) の分だけ 4 倍からずれる
    assert b / a == pytest.approx(4.0, rel=0.05)


def test_surrogate_rotates_with_inflow_azimuth(rotor, surrogate):
    """流入方位を 90 deg 回すと面内成分も 90 deg 回る."""
    w0 = surrogate.solve(rotor, OperatingPoint(
        rpm=6000, v_inf=8.0, inflow_angle_deg=45.0, inflow_azimuth_deg=0.0)).wrench
    w90 = surrogate.solve(rotor, OperatingPoint(
        rpm=6000, v_inf=8.0, inflow_angle_deg=45.0, inflow_azimuth_deg=90.0)).wrench
    assert w90.force[2] == pytest.approx(w0.force[2], rel=1e-9)
    assert w90.force[1] == pytest.approx(w0.force[0], rel=1e-9)
    assert w90.force[0] == pytest.approx(-w0.force[1], rel=1e-9)


def test_surrogate_training_error_is_small(rotor, surrogate):
    err = surrogate.training_error(rotor)
    assert err["ct"] < 5e-3
    assert err["cq"] < 5e-4


def test_surrogate_is_fast(rotor, surrogate, bemt):
    import time

    op = OperatingPoint(rpm=6000, v_inf=4.0, inflow_angle_deg=20.0)
    t0 = time.perf_counter()
    for _ in range(200):
        surrogate.solve(rotor, op)
    t_sur = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    bemt.solve(rotor, op)
    t_bemt = time.perf_counter() - t0
    assert t_sur < 0.1 * t_bemt


@pytest.mark.parametrize("method", ["poly", "rbf", "linear"])
def test_surrogate_methods_all_work(rotor, bemt, method):
    m = SurrogateModel.fit(
        rotor, bemt, rpm=[6000.0],
        advance_ratios=np.linspace(0.0, 0.6, 5),
        inflow_angles_deg=np.array([0.0, 30.0, 60.0]),
        method=method, degree=2,
    )
    op = OperatingPoint(rpm=6000, v_inf=5.0, inflow_angle_deg=30.0)
    ref = bemt.solve(rotor, op).thrust
    assert m.solve(rotor, op).thrust == pytest.approx(ref, rel=0.10, abs=0.1)


def test_surrogate_axial_only_training_gives_zero_in_plane(rotor, bemt):
    """軸流だけで学習した場合, 面内成分は外挿せず 0 を返す."""
    m = SurrogateModel.fit(
        rotor, bemt, rpm=[6000.0], advance_ratios=np.linspace(0.0, 0.6, 6),
        inflow_angles_deg=np.array([0.0]), degree=2,
    )
    w = m.solve(rotor, OperatingPoint(
        rpm=6000, v_inf=8.0, inflow_angle_deg=45.0)).wrench
    assert np.allclose(w.force[:2], 0.0)


def test_surrogate_rejects_too_few_samples(rotor, bemt):
    with pytest.raises(ValueError):
        SurrogateModel.fit(
            rotor, bemt, rpm=[6000.0], advance_ratios=np.array([0.0, 0.3]),
            inflow_angles_deg=np.array([0.0]), degree=5,
        )


def test_surrogate_from_static_result(rotor, bemt):
    from prop_sim.experiments import static_sweep

    res = static_sweep(
        rotor, bemt, rpm=np.array([5000.0, 8000.0]),
        v_inf=np.array([0.0, 5.0, 10.0]), inflow_angle_deg=np.array([0.0, 40.0]),
    )
    m = SurrogateModel.from_static_result(res, rotor, degree=2)
    op = OperatingPoint(rpm=6500, v_inf=6.0, inflow_angle_deg=20.0)
    assert m.solve(rotor, op).thrust == pytest.approx(
        bemt.solve(rotor, op).thrust, rel=0.08, abs=0.1
    )


def test_surrogate_from_external_six_component_table(rotor):
    """CFD/実測を想定した「作動点 + 6 分力」の直接投入."""
    ops, ws = [], []
    for rpm in (5000.0, 9000.0):
        for v in (0.0, 5.0, 10.0):
            op = OperatingPoint(rpm=rpm, v_inf=v)
            ops.append(op)
            n = rpm / 60.0
            ws.append([0.0, 0.0, 0.10 * 1.225 * n**2 * 0.254**4,
                       0.0, 0.0, -0.008 * 1.225 * n**2 * 0.254**5])
    m = SurrogateModel.from_conditions(rotor, ops, ws, degree=1,
                                       source="pretend CFD")
    sol = m.solve(rotor, OperatingPoint(rpm=7000, v_inf=2.5))
    n = 7000 / 60.0
    assert sol.thrust == pytest.approx(0.10 * 1.225 * n**2 * 0.254**4, rel=1e-6)
    assert m.source == "pretend CFD"


def test_registry_has_all_levels():
    assert isinstance(get_model("lifting_line"), LiftingLine)
    assert isinstance(get_model("surrogate"), SurrogateModel)


def test_surrogate_without_data_raises(rotor):
    with pytest.raises(RuntimeError):
        SurrogateModel().solve(rotor, OperatingPoint(rpm=6000))
