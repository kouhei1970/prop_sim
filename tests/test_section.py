"""翼断面 -> 翼型モデル (薄翼理論 + 低 Re 補正) のテスト."""

import numpy as np
import pytest

from prop_sim import STAMPFLY_1209_SECTION, stampfly_1209_airfoil
from prop_sim.section import SectionShape, airfoil_from_section


def _shape(camber_fn, thickness=0.10, n=201) -> SectionShape:
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    t = 4.0 * thickness * np.sqrt(np.clip(x, 0, 1)) * (1.0 - x)
    return SectionShape(x, camber_fn(x), t)


# ------------------------------------------------------------ SectionShape
def test_section_validates_inputs():
    with pytest.raises(ValueError):
        SectionShape([0.0, 1.0], [0.0], [0.1, 0.1])
    with pytest.raises(ValueError):
        SectionShape([1.0, 0.0], [0.0, 0.0], [0.1, 0.1])


def test_upper_and_lower_surfaces():
    s = SectionShape([0.0, 0.5, 1.0], [0.0, 0.05, 0.0], [0.0, 0.10, 0.0])
    assert np.allclose(s.upper, [0.0, 0.10, 0.0])
    assert np.allclose(s.lower, [0.0, 0.00, 0.0])


def test_scaled_changes_camber_and_thickness():
    s = STAMPFLY_1209_SECTION.scaled(thickness=0.5, camber=2.0)
    a, b = s.thin_airfoil_properties(), STAMPFLY_1209_SECTION.thin_airfoil_properties()
    assert a.thickness_max == pytest.approx(0.5 * b.thickness_max, rel=1e-6)
    assert a.camber_max == pytest.approx(2.0 * b.camber_max, rel=1e-6)


# --------------------------------------------------------------- 薄翼理論
def test_symmetric_section_has_zero_lift_at_zero_alpha():
    p = _shape(lambda x: np.zeros_like(x)).thin_airfoil_properties()
    assert p.alpha0_rad == pytest.approx(0.0, abs=1e-6)
    assert p.cm_ac == pytest.approx(0.0, abs=1e-6)
    assert p.cl_ideal == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("m", [0.02, 0.04, 0.06])
def test_circular_arc_camber_matches_analytic_alpha0(m):
    """放物線キャンバ z = 4 m x(1-x) の理論値は alpha0 = -2 m [rad]."""
    p = _shape(lambda x: 4.0 * m * x * (1.0 - x)).thin_airfoil_properties()
    assert p.alpha0_rad == pytest.approx(-2.0 * m, rel=0.03)
    # 放物線キャンバの理想迎角は 0, cm_ac = -pi*m
    assert p.alpha_ideal_rad == pytest.approx(0.0, abs=2e-3)
    assert p.cm_ac == pytest.approx(-np.pi * m, rel=0.05)


def test_more_camber_gives_more_negative_alpha0():
    a0 = [
        _shape(lambda x, m=m: 4.0 * m * x * (1 - x)).thin_airfoil_properties().alpha0_rad
        for m in (0.0, 0.02, 0.05, 0.08)
    ]
    assert all(np.diff(a0) < 0.0)


def test_camber_position_affects_ideal_angle():
    """キャンバが前寄りだと前縁の傾きが急になり, 理想迎角が正側へ動く.

    (キャンバ位置が中央の放物線では理想迎角がちょうど 0 になる.)
    """
    front = _shape(lambda x: 0.06 * np.sin(np.pi * x**0.6)).thin_airfoil_properties()
    rear = _shape(lambda x: 0.06 * np.sin(np.pi * x**1.6)).thin_airfoil_properties()
    assert front.camber_max_x < rear.camber_max_x
    assert front.alpha_ideal_rad > rear.alpha_ideal_rad


# --------------------------------------------------- airfoil_from_section
def test_generated_airfoil_reproduces_zero_lift_angle():
    sec = _shape(lambda x: 4.0 * 0.05 * x * (1 - x))
    foil = airfoil_from_section(sec, camber_efficiency=1.0)
    cl, _, _ = foil.coefficients(np.array([foil.alpha0]))
    assert cl[0] == pytest.approx(0.0, abs=1e-6)
    p = sec.thin_airfoil_properties()
    assert np.deg2rad(foil.alpha0_deg) == pytest.approx(p.alpha0_rad, rel=1e-6)


def test_camber_efficiency_scales_alpha0():
    sec = _shape(lambda x: 4.0 * 0.05 * x * (1 - x))
    full = airfoil_from_section(sec, camber_efficiency=1.0)
    part = airfoil_from_section(sec, camber_efficiency=0.5)
    assert part.alpha0_deg == pytest.approx(0.5 * full.alpha0_deg, rel=1e-6)


def test_lift_slope_grows_with_reynolds():
    sec = _shape(lambda x: 4.0 * 0.04 * x * (1 - x))
    a = airfoil_from_section(sec, reynolds_ref=1e4).cl_alpha
    b = airfoil_from_section(sec, reynolds_ref=1e6).cl_alpha
    assert a < b < 2 * np.pi


def test_drag_falls_with_reynolds_and_grows_with_thickness():
    thin = _shape(lambda x: np.zeros_like(x), thickness=0.06)
    thick = _shape(lambda x: np.zeros_like(x), thickness=0.18)
    assert airfoil_from_section(thin, reynolds_ref=1e4).cd0 > airfoil_from_section(
        thin, reynolds_ref=1e6
    ).cd0
    assert airfoil_from_section(thick).cd0 > airfoil_from_section(thin).cd0


def test_bubble_penalty_scales_drag():
    sec = _shape(lambda x: np.zeros_like(x))
    a = airfoil_from_section(sec, bubble_penalty=1.0).cd0
    b = airfoil_from_section(sec, bubble_penalty=2.0).cd0
    assert b == pytest.approx(2.0 * a, rel=1e-6)


# ------------------------------------------------------- StampFly の断面
def test_stampfly_section_matches_photo_measurement():
    p = STAMPFLY_1209_SECTION.thin_airfoil_properties()
    assert p.thickness_max == pytest.approx(0.095, abs=0.006)
    assert p.camber_max == pytest.approx(0.071, abs=0.006)
    assert 0.42 < p.camber_max_x < 0.58
    assert 0.15 < p.thickness_max_x < 0.28
    assert p.alpha0_deg == pytest.approx(-8.3, abs=0.8)
    assert 0.85 < p.cl_ideal < 1.15
    assert -0.30 < p.cm_ac < -0.12


def test_stampfly_section_lower_surface_hugs_chord_near_leading_edge():
    """前縁付近の下面はほぼ翼弦線に乗る (影を断面と誤検出しないこと)."""
    s = STAMPFLY_1209_SECTION.resample(200)
    lower = np.interp([0.05, 0.10, 0.20], s.x, s.lower)
    assert np.all(lower > -0.02)
    assert np.all(np.abs(lower) < 0.02)
    # 上面は逆にしっかり盛り上がっている
    upper = np.interp([0.05, 0.10, 0.20], s.x, s.upper)
    assert np.all(upper > 0.04)


def test_stampfly_airfoil_is_reasonable_at_low_reynolds():
    f = stampfly_1209_airfoil()
    assert 3.5 < f.cl_alpha < 4.5           # 2 pi の 6 - 7 割
    assert -8.5 < f.alpha0_deg < -5.5
    assert 0.9 < f.cl_max < 1.2
    assert 0.03 < f.cd0 < 0.08
    a = np.deg2rad(np.linspace(-5, 15, 200))
    cl, cd, _ = f.coefficients(a, np.full_like(a, 1.2e4), None)
    ld = (cl / cd).max()
    assert 8.0 < ld < 25.0                  # Re ~ 1e4 の妥当な最大揚抗比


def test_measured_section_gives_more_thrust_than_generic_thin_airfoil():
    """実測断面はキャンバが強いので, 汎用の低 Re 薄翼より推力が出る."""
    from prop_sim import BEMT, LOW_RE_THIN, OperatingPoint, Rotor, stampfly_1209

    model = BEMT()
    op = OperatingPoint(rpm=30000)
    measured = model.solve(Rotor(stampfly_1209()), op).thrust
    generic = model.solve(Rotor(stampfly_1209(airfoil=LOW_RE_THIN)), op).thrust
    assert measured > generic
    assert measured < 1.6 * generic


def test_section_derived_model_is_less_sensitive_than_guessing():
    """断面が分かると仮定の幅が狭まる (ホバー回転数のばらつきが小さい)."""
    from scipy.optimize import brentq

    from prop_sim import BEMT, OperatingPoint, Rotor, stampfly_1209

    model = BEMT()

    def hover(foil):
        r = Rotor(stampfly_1209(airfoil=foil))
        return brentq(
            lambda n: model.solve(r, OperatingPoint(rpm=n)).thrust / 9.80665 * 1e3 - 9.2,
            5000.0, 80000.0, xtol=10.0,
        )

    rpms = [
        hover(stampfly_1209_airfoil(camber_efficiency=c, bubble_penalty=b))
        for c in (0.70, 0.85, 1.00) for b in (1.2, 1.6, 2.2)
    ]
    assert (max(rpms) - min(rpms)) / np.mean(rpms) < 0.08
