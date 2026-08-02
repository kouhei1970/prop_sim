"""試験シナリオ・センサ・後処理のテスト."""

import numpy as np
import pytest

from prop_sim import BEMT, OperatingPoint, Rotor, Unbalance, from_diameter_pitch
from prop_sim.drivetrain import Drivetrain
from prop_sim.experiments import dynamic_run, static_sweep
from prop_sim.frames import Wrench
from prop_sim.postproc import (
    harmonic_amplitudes,
    order_spectrum,
    propeller_coefficients,
    revolution_average,
    spectrum,
)
from prop_sim.sensor import LoadCell, SensorConfig, TestRig


@pytest.fixture(scope="module")
def rotor():
    return Rotor(from_diameter_pitch(10.0, 4.7))


@pytest.fixture(scope="module")
def model():
    return BEMT()


def make_rig(**kwargs) -> TestRig:
    cfg = SensorConfig(
        full_scale=np.array([20.0, 20.0, 50.0, 1.0, 1.0, 1.0]), **kwargs
    )
    return TestRig(load_cell=LoadCell(cfg))


# ------------------------------------------------------------------ 静的試験
def test_static_sweep_shape_and_monotonicity(rotor, model):
    res = static_sweep(rotor, model, rpm=np.array([3000.0, 5000.0, 7000.0]))
    assert res.n_points == 3
    assert np.all(np.diff(res["Fz_true"]) > 0.0)
    assert np.all(res["Mz_true"] < 0.0)


def test_static_sweep_builds_full_grid(rotor, model):
    res = static_sweep(
        rotor, model,
        rpm=np.array([4000.0, 6000.0]),
        v_inf=np.array([0.0, 5.0]),
        inflow_angle_deg=np.array([0.0, 30.0]),
    )
    assert res.n_points == 8


def test_static_sweep_measurement_is_close_to_truth(rotor, model):
    rig = make_rig(seed=1)
    res = static_sweep(rotor, model, rpm=np.array([6000.0]), rig=rig, n_average=5000)
    # 干渉の校正残差 + ノイズで 1% 以内には収まるはず
    assert abs(res["Fz"][0] - res["Fz_true"][0]) < 0.01 * abs(res["Fz_true"][0]) + 0.01


def test_static_sweep_with_drivetrain_reports_electrical(rotor, model):
    res = static_sweep(
        rotor, model, rpm=np.array([5000.0, 7000.0]), drivetrain=Drivetrain()
    )
    assert np.all(np.diff(res["throttle"]) > 0.0)
    assert np.all(res["current"] > 0.0)
    assert np.all(res["power_electrical"] > res["power_shaft"])


def test_static_result_csv_roundtrip(rotor, model, tmp_path):
    from prop_sim.io import read_csv

    res = static_sweep(rotor, model, rpm=np.array([5000.0, 6000.0]))
    path = tmp_path / "static.csv"
    res.to_csv(path)
    back = read_csv(path)
    assert np.allclose(back["Fz_true"], res["Fz_true"], rtol=1e-6)


# ---------------------------------------------------------------- 試験スタンド
def test_sensor_offset_creates_moment_from_side_force():
    rig = TestRig(sensor_offset=np.array([0.0, 0.0, -0.2]))
    hub = Wrench([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    out = rig.to_sensor(hub)
    assert out.moment[1] == pytest.approx(0.2, rel=1e-9)


def test_sensor_offset_does_not_affect_thrust_or_torque():
    rig = TestRig(sensor_offset=np.array([0.0, 0.0, -0.2]))
    hub = Wrench([0.0, 0.0, 5.0], [0.0, 0.0, -0.1])
    out = rig.to_sensor(hub)
    assert out.force[2] == pytest.approx(5.0)
    assert out.moment[2] == pytest.approx(-0.1)


def test_load_cell_crosstalk_is_partially_corrected():
    cfg = SensorConfig(
        full_scale=np.array([20.0, 20.0, 50.0, 1.0, 1.0, 1.0]),
        crosstalk=0.03, calibration_residual=0.5,
        noise_rms=0.0, gain_error=0.0, adc_bits=0, seed=3,
    )
    cell = LoadCell(cfg)
    truth = np.zeros((1, 6))
    truth[0, 2] = 30.0                      # 推力だけ加える
    meas = cell.measure_static(truth)
    leak = np.abs(meas[0, [0, 1, 3, 4, 5]]).max()
    assert leak > 0.0                       # 残差干渉が残る
    assert leak < 0.03 * 30.0               # ただし生の干渉量より小さい


def test_load_cell_quantization_step():
    cfg = SensorConfig(
        full_scale=np.array([10.0] * 6), crosstalk=0.0, gain_error=0.0,
        noise_rms=0.0, adc_bits=8, seed=0,
    )
    cell = LoadCell(cfg)
    out = cell.measure_static(np.full((1, 6), 1.234567))
    lsb = 2 * 10.0 / 2**8
    assert np.allclose(np.round(out / lsb), out / lsb, atol=1e-9)


def test_load_cell_saturates_beyond_full_scale():
    cfg = SensorConfig(
        full_scale=np.array([10.0] * 6), crosstalk=0.0, gain_error=0.0,
        noise_rms=0.0, adc_bits=0, seed=0,
    )
    cell = LoadCell(cfg)
    out = cell.measure_static(np.full((1, 6), 25.0))
    assert np.all(out <= 10.0 + 1e-9)


def test_load_cell_series_resamples_to_sample_rate():
    cfg = SensorConfig(sample_rate=1000.0, anti_alias_hz=400.0)
    cell = LoadCell(cfg)
    t = np.arange(0.0, 0.1, 1e-4)                # 10 kHz
    w = np.zeros((t.size, 6))
    w[:, 2] = 5.0
    res = cell.measure_series(t, w)
    assert res.time.size == pytest.approx(t.size / 10, rel=0.05)
    assert res.measured[10:, 2].mean() == pytest.approx(5.0, rel=0.02)


# ------------------------------------------------------------------ 動的試験
def test_dynamic_steady_state_matches_static_solution(rotor, model):
    op = OperatingPoint(rpm=6000)
    steady = model.solve(rotor, op)
    res = dynamic_run(rotor, model, duration=0.02, rpm=6000.0, sim_rate=20000)
    assert res.states["thrust"][-1] == pytest.approx(steady.thrust, rel=1e-3)
    assert res.states["torque_aero"][-1] == pytest.approx(
        steady.torque(rotor.spin), rel=1e-3
    )


def test_dynamic_inflow_lag_causes_thrust_overshoot(rotor, model):
    """誘導速度が育つ前は推力が過大になる (インフローの立ち上がり遅れ)."""
    op = OperatingPoint(rpm=6000)
    inflow = model.initial_inflow(rotor, op)
    steady_thrust = model.solve(rotor, op).thrust
    inflow.v0 = inflow.v0 * 0.4
    dt, psi = 2.0e-5, 0.0
    peak = -np.inf
    for _ in range(4000):
        w, dv = model.step(rotor, op, inflow, psi)
        peak = max(peak, float(w.force[2]))
        psi += op.omega * dt
        inflow.v0 = inflow.v0 + dv * dt
    assert peak > 1.1 * steady_thrust
    assert float(w.force[2]) == pytest.approx(steady_thrust, rel=5e-3)


def test_mass_unbalance_produces_1p_force_of_expected_magnitude():
    geo = from_diameter_pitch(10.0, 4.7)
    u = 1.0e-5                                     # [kg m]
    rotor = Rotor(geo, unbalance=Unbalance(static=u))
    rpm = 6000.0
    omega = rpm * 2 * np.pi / 60
    res = dynamic_run(Rotor(geo), BEMT(), duration=0.03, rpm=rpm, sim_rate=20000)
    res_u = dynamic_run(rotor, BEMT(), duration=0.03, rpm=rpm, sim_rate=20000)
    psi = np.deg2rad(res_u.states["psi_deg"])
    fit = harmonic_amplitudes(
        res_u.hub_wrench[:, 0] - res.hub_wrench[:, 0], psi, orders=[1]
    )
    assert fit.amplitude[0] == pytest.approx(u * omega**2, rel=0.02)


def test_blade_pitch_imbalance_creates_1p_hub_moment():
    """空力アンバランスは 1/rev のハブモーメントとして現れる.

    軸流では各ブレードの推力は方位角によらず一定なので, 合計推力 Fz は
    変動しない. 一方 2 枚の推力差はロータと一緒に回る面内モーメントを
    生むため, Mx, My に 1/rev が立つ — 実機の振動診断と同じ現象.
    """
    geo = from_diameter_pitch(10.0, 4.7)
    clean = dynamic_run(Rotor(geo), BEMT(), duration=0.03, rpm=6000.0, sim_rate=20000)
    dirty = dynamic_run(
        Rotor(geo, blade_pitch_offsets_deg=[0.0, 1.0]),
        BEMT(), duration=0.03, rpm=6000.0, sim_rate=20000,
    )
    psi = np.deg2rad(dirty.states["psi_deg"])
    a_clean = harmonic_amplitudes(clean.hub_wrench[:, 3], psi, orders=[1]).amplitude[0]
    a_dirty = harmonic_amplitudes(dirty.hub_wrench[:, 3], psi, orders=[1]).amplitude[0]
    assert a_dirty > 1e-3
    assert a_dirty > 100 * a_clean
    # 合計推力にはほとんど 1/rev が出ない
    fz = harmonic_amplitudes(dirty.hub_wrench[:, 2], psi, orders=[1]).amplitude[0]
    assert fz < 1e-3 * a_dirty


def test_edgewise_flow_hub_moment_has_mean_and_blade_passage_harmonic():
    """B 枚ロータのハブ荷重には B の倍数の次数だけが残る.

    斜め流入によるブレード 1 枚あたりの 1/rev 荷重変動は, ハブでは
    定常成分 + B/rev 成分に変換される (回転系 -> 静止系の次数変換).
    """
    rotor = Rotor(from_diameter_pitch(10.0, 4.7))     # B = 2
    res = dynamic_run(
        rotor, BEMT(), duration=0.03, rpm=6000.0, v_inf=10.0,
        inflow_angle_deg=60.0, sim_rate=20000,
    )
    psi = np.deg2rad(res.states["psi_deg"])
    fit = harmonic_amplitudes(res.hub_wrench[:, 3], psi, orders=[1, 2, 3])
    assert abs(fit.mean) > 1e-3            # 定常ハブモーメント
    assert fit.amplitude[1] > 1e-3         # 2/rev = ブレード通過
    assert fit.amplitude[0] < 0.02 * fit.amplitude[1]   # 1/rev は打ち消される
    assert fit.amplitude[2] < 0.02 * fit.amplitude[1]


def test_spin_up_reaction_torque_exceeds_steady_value(rotor, model):
    """加速中は -J dOmega/dt の分だけ計測トルクが大きくなる."""
    res = dynamic_run(
        rotor, model, duration=0.06, throttle=1.0,
        drivetrain=Drivetrain(prop_inertia=rotor.geometry.polar_inertia),
        initial_rpm=1000.0, sim_rate=20000,
    )
    assert res.states["rpm"][-1] > res.states["rpm"][0]
    accel = res.states["omega_dot"] > 100.0
    assert np.any(accel)
    inertial = -res.hub_wrench[:, 5] - res.states["torque_aero"]
    assert np.all(inertial[accel] > 0.0)


def test_gyroscopic_moment_appears_with_body_rate(rotor, model):
    q = 2.0                                        # [rad/s] ピッチ角速度
    res = dynamic_run(
        rotor, model, duration=0.01, rpm=6000.0,
        body_rate=np.array([0.0, q, 0.0]), sim_rate=20000,
    )
    jz = rotor.geometry.polar_inertia
    omega = 6000 * 2 * np.pi / 60
    expected = jz * omega * q                      # |-w x h| = Jz*Omega*q
    assert abs(res.hub_wrench[-1, 3]) == pytest.approx(expected, rel=0.05)


def test_dynamic_measurement_columns(rotor, model, tmp_path):
    rig = make_rig(sample_rate=4000.0, seed=2)
    res = dynamic_run(
        rotor, model, duration=0.02, rpm=6000.0, rig=rig, sim_rate=20000
    )
    assert res.measurement is not None
    cols = res.columns(sampled=True)
    assert {"time", "Fz", "Fz_true", "rpm"} <= set(cols)
    res.to_csv(tmp_path / "dyn.csv")
    assert (tmp_path / "dyn.csv").exists()


def test_throttle_step_changes_rpm(rotor, model):
    res = dynamic_run(
        rotor, model, duration=0.15,
        throttle=lambda t: 0.4 if t < 0.05 else 0.8,
        drivetrain=Drivetrain(prop_inertia=rotor.geometry.polar_inertia),
        initial_rpm=3000.0, sim_rate=10000,
    )
    rpm = res.states["rpm"]
    assert rpm[-1] > rpm[int(0.05 / (res.time[1] - res.time[0]))]


def test_requires_rpm_or_throttle(rotor, model):
    with pytest.raises(ValueError):
        dynamic_run(rotor, model, duration=0.01)


# -------------------------------------------------------------------- 後処理
def test_harmonic_fit_recovers_synthetic_signal():
    psi = np.linspace(0.0, 40 * np.pi, 4000)
    y = 3.0 + 2.0 * np.cos(psi - 0.3) + 0.5 * np.sin(2 * psi)
    fit = harmonic_amplitudes(y, psi, orders=[1, 2])
    assert fit.mean == pytest.approx(3.0, abs=1e-6)
    assert fit.amplitude[0] == pytest.approx(2.0, rel=1e-6)
    assert fit.amplitude[1] == pytest.approx(0.5, rel=1e-6)
    assert np.deg2rad(fit.phase_deg[0]) == pytest.approx(0.3, abs=1e-6)


def test_revolution_average_reproduces_waveform():
    psi = np.linspace(0.0, 60 * np.pi, 9000)
    y = np.cos(psi)
    centers, mean, _ = revolution_average(y, psi, n_bins=36)
    assert np.allclose(mean, np.cos(np.deg2rad(centers)), atol=0.06)


def test_spectrum_finds_tone():
    t = np.arange(0.0, 1.0, 1e-4)
    y = 1.5 * np.sin(2 * np.pi * 250.0 * t)
    f, a = spectrum(t, y)
    assert f[np.argmax(a)] == pytest.approx(250.0, abs=2.0)
    assert a.max() == pytest.approx(1.5, rel=0.05)


def test_order_spectrum_survives_speed_variation():
    t = np.linspace(0.0, 1.0, 20000)
    omega = 2 * np.pi * (100.0 + 40.0 * t)     # 回転数が変化する
    psi = np.cumsum(omega) * (t[1] - t[0])
    y = np.cos(3.0 * psi)
    orders, amp = order_spectrum(psi, y, n_order=8)
    assert orders[np.argmax(amp)] == pytest.approx(3.0, abs=0.15)


def test_propeller_coefficients_match_definition():
    out = propeller_coefficients(
        thrust=np.array([4.0]), torque=np.array([0.06]),
        rpm=np.array([6000.0]), diameter=0.254, v_axial=0.0, density=1.225,
    )
    n = 100.0
    assert out["Ct"][0] == pytest.approx(4.0 / (1.225 * n**2 * 0.254**4))
    assert out["Cp"][0] == pytest.approx(2 * np.pi * out["Cq"][0])
