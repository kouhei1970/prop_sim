"""座標系・翼型・形状の基本テスト."""

import numpy as np
import pytest

from prop_sim.airfoil import CLARK_Y, NACA0012, LinearAirfoil, TabulatedAirfoil
from prop_sim.atmosphere import Atmosphere
from prop_sim.frames import Wrench
from prop_sim.geometry import INCH, from_diameter_pitch


# ------------------------------------------------------------------ Wrench
def test_wrench_translate_moment_arm():
    """作用点をずらすと F x d のモーメントが現れる."""
    w = Wrench([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    moved = w.translate([0.0, 0.0, -0.2])   # 0.2 m 下の点まわり
    assert np.allclose(moved.force, [1.0, 0.0, 0.0])
    assert np.allclose(moved.moment, [0.0, 0.2, 0.0])


def test_wrench_translate_axial_force_no_moment():
    w = Wrench([0.0, 0.0, 5.0], [0.0, 0.0, 0.0])
    assert np.allclose(w.translate([0.0, 0.0, -0.3]).moment, 0.0)


def test_wrench_array_roundtrip():
    arr = np.arange(12.0).reshape(2, 6)
    assert np.allclose(Wrench.from_array(arr).as_array(), arr)


# ---------------------------------------------------------------- 大気モデル
def test_atmosphere_sea_level():
    a = Atmosphere()
    assert a.density == pytest.approx(1.225, rel=1e-3)
    assert a.speed_of_sound == pytest.approx(340.3, rel=1e-3)


def test_atmosphere_altitude_decreases_density():
    assert Atmosphere(altitude_m=2000).density < Atmosphere().density


def test_humidity_reduces_density():
    dry = Atmosphere(temperature_k=303.15, humidity=0.0).density
    wet = Atmosphere(temperature_k=303.15, humidity=0.9).density
    assert wet < dry


# -------------------------------------------------------------------- 翼型
@pytest.mark.parametrize("foil", [CLARK_Y, NACA0012])
def test_airfoil_360_is_finite_and_continuous(foil):
    a = np.linspace(-np.pi, np.pi, 3601)
    cl, cd, _ = foil.coefficients(a)
    assert np.all(np.isfinite(cl)) and np.all(np.isfinite(cd))
    assert np.all(cd > 0.0)
    # 隣接点で飛びがない (1 deg 刻みで Cl の変化 < 0.15)
    assert np.max(np.abs(np.diff(cl))) < 0.15


def test_linear_airfoil_slope_and_zero_lift():
    foil = LinearAirfoil(cl_alpha=6.0, alpha0_deg=-2.0, cl_max=1.2)
    a = np.deg2rad(np.array([-2.0, 0.0, 2.0]))
    cl, _, _ = foil.coefficients(a)
    assert cl[0] == pytest.approx(0.0, abs=1e-9)
    slope = (cl[2] - cl[0]) / np.deg2rad(4.0)
    assert slope == pytest.approx(6.0, rel=0.02)


def test_symmetric_airfoil_is_antisymmetric():
    a = np.deg2rad(np.linspace(-60, 60, 121))
    cl, cd, _ = NACA0012.coefficients(a)
    cl_m, cd_m, _ = NACA0012.coefficients(-a)
    assert np.allclose(cl, -cl_m, atol=1e-9)
    assert np.allclose(cd, cd_m, atol=1e-9)


def test_airfoil_stall_limits_lift():
    foil = LinearAirfoil(cl_max=1.1)
    cl, _, _ = foil.coefficients(np.deg2rad(np.linspace(0, 40, 100)))
    assert cl.max() <= 1.2


def test_reynolds_correction_increases_drag_at_low_re():
    a = np.deg2rad(np.array([4.0]))
    _, cd_low, _ = CLARK_Y.coefficients(a, reynolds=np.array([3e4]))
    _, cd_high, _ = CLARK_Y.coefficients(a, reynolds=np.array([1e6]))
    assert cd_low[0] > cd_high[0]


def test_tabulated_airfoil_matches_table_inside_range():
    alpha = np.linspace(-8, 12, 21)
    cl = 0.1 * alpha + 0.2
    cd = 0.01 + 0.0005 * alpha**2
    foil = TabulatedAirfoil(alpha_deg=alpha, cl=cl, cd=cd)
    got, _, _ = foil.coefficients(np.deg2rad(np.array([5.0])))
    assert got[0] == pytest.approx(0.7, abs=0.05)


# -------------------------------------------------------------------- 形状
def test_from_diameter_pitch_recovers_pitch():
    geo = from_diameter_pitch(10.0, 4.7)
    assert geo.diameter == pytest.approx(10 * INCH)
    assert geo.pitch_at(0.75) / INCH == pytest.approx(4.7, rel=0.02)


def test_geometry_validates_monotonic_radius():
    from prop_sim.geometry import PropellerGeometry

    with pytest.raises(ValueError):
        PropellerGeometry(0.1, 2, [0.5, 0.2], [0.1, 0.1], [10.0, 5.0])


def test_solidity_scales_with_blade_count():
    two = from_diameter_pitch(10, 4.7, n_blades=2).solidity
    three = from_diameter_pitch(10, 4.7, n_blades=3).solidity
    assert three == pytest.approx(1.5 * two, rel=1e-6)
