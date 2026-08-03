"""CFD モジュール (格子生成・polyMesh 入出力・荷重積分) のテスト.

OpenFOAM の実行は要らない。ケースの生成と, 自前で書いた polyMesh を
自前で読み戻して幾何量・積分値が理論値と合うかを検証する。
"""

import numpy as np
import pytest

import prop_sim as ps
from prop_sim.cfd import (
    AirfoilCaseConfig,
    OpenFoamCase,
    PolyMesh,
    build_blockmesh_dict,
    cell_areas,
    integrate_coefficients,
    ogrid_nodes,
    patch_wrench,
    polygon_centroid,
    read_volume_field,
    surface_points,
    write_polymesh,
)
from prop_sim.cfd.mesh import _geometric, _node_normals
from prop_sim.section import SectionShape


def circle_section(n=120, name="cylinder"):
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    t = 2.0 * np.sqrt(np.maximum(0.25 - (x - 0.5) ** 2, 0.0))
    return SectionShape(x=x, camber=np.zeros_like(x), thickness=t, name=name)


@pytest.fixture(scope="module")
def cyl_points():
    return surface_points(circle_section(), 80)


@pytest.fixture(scope="module")
def foil_points():
    return surface_points(ps.STAMPFLY_1209_SECTION, 120)


# --------------------------------------------------------------- surface_points
def test_surface_points_is_counter_clockwise_and_closed(foil_points):
    p = foil_points
    q = np.roll(p, -1, axis=0)
    assert 0.5 * np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]) > 0.0
    assert not np.any(np.all(np.isclose(p, np.roll(p, -1, axis=0)), axis=1))


def test_surface_points_area_matches_the_section(foil_points):
    """多角形面積が sum(t dx) と一致する."""
    s = ps.STAMPFLY_1209_SECTION
    p, q = foil_points, np.roll(foil_points, -1, axis=0)
    poly = 0.5 * np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])
    assert poly == pytest.approx(np.trapezoid(s.thickness, s.x), rel=0.02)


def test_sharp_trailing_edge_is_not_duplicated():
    x = np.linspace(0.0, 1.0, 60)
    sec = SectionShape(x=x, camber=np.zeros_like(x),
                       thickness=0.12 * np.sin(np.pi * x), name="sharp")
    p = surface_points(sec, 60)
    assert np.linalg.norm(p[0] - p[-1]) > 1e-6


# ------------------------------------------------------------ polygon_centroid
def test_polygon_centroid_of_a_circle():
    a = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    p = np.stack([2.0 + np.cos(a), -1.0 + np.sin(a)], axis=1)
    assert polygon_centroid(p) == pytest.approx([2.0, -1.0], abs=1e-9)


def test_polygon_centroid_falls_back_for_degenerate_input():
    p = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert polygon_centroid(p) == pytest.approx([1.0, 0.0])


# -------------------------------------------------------------------- 径方向分布
def test_geometric_distribution_total_and_ratio():
    n, total, grading = 40, 12.0, 500.0
    s = _geometric(n, total, grading)
    h = np.diff(s)
    assert s[0] == 0.0 and s[-1] == pytest.approx(total)
    assert h[-1] / h[0] == pytest.approx(grading)
    assert np.all(h > 0)


def test_geometric_reduces_to_uniform():
    s = _geometric(5, 1.0, 1.0)
    assert np.allclose(np.diff(s), 0.2)


def test_node_normals_point_outward_on_a_circle(cyl_points):
    n = _node_normals(cyl_points)
    r = cyl_points - np.array([0.5, 0.0])
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    assert np.min(np.einsum("ij,ij->i", n, r)) > 0.99


# ------------------------------------------------------------------- ogrid_nodes
def test_ogrid_cells_are_all_positive_for_the_stampfly_section(foil_points):
    x = ogrid_nodes(foil_points, n_radial=60, far_field_radius=30.0)
    assert cell_areas(x).min() > 0.0
    assert x.shape == (foil_points.shape[0], 61, 2)


def test_ogrid_outer_layer_lies_on_the_far_field_circle(foil_points):
    r_far = 25.0
    x = ogrid_nodes(foil_points, n_radial=40, far_field_radius=r_far)
    r = np.linalg.norm(x[:, -1, :] - foil_points.mean(axis=0), axis=1)
    assert r.max() - r.min() < 1e-6 * r_far
    assert r.mean() == pytest.approx(r_far, rel=1e-6)


def test_ogrid_first_layer_is_normal_to_the_wall(cyl_points):
    """円柱なら最初の層は厳密に半径方向を向く."""
    x = ogrid_nodes(cyl_points, n_radial=50, far_field_radius=30.0)
    step = x[:, 1, :] - x[:, 0, :]
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    r = cyl_points - np.array([0.5, 0.0])
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    assert np.min(np.einsum("ij,ij->i", step, r)) > 0.999


def test_ogrid_respects_the_radial_grading(foil_points):
    x = ogrid_nodes(foil_points, n_radial=60, far_field_radius=30.0,
                    radial_grading=500.0)
    h = np.linalg.norm(np.diff(x, axis=1), axis=2)
    # 壁の第 1 層は遠方より圧倒的に薄い
    assert h[:, -1].mean() / h[:, 0].mean() > 100.0


def test_ogrid_rejects_bad_input():
    with pytest.raises(ValueError):
        ogrid_nodes(np.zeros((5, 3)))


def test_ogrid_reports_degenerate_cells(foil_points):
    """押し出し距離を無茶にすると検出して例外にする (黙って壊れない)."""
    with pytest.raises(ValueError, match="面積"):
        ogrid_nodes(foil_points, n_radial=60, far_field_radius=30.0,
                    normal_length=20.0, smooth_passes=0)


# -------------------------------------------------------- polyMesh 書き出し/読み
@pytest.fixture(scope="module")
def written_mesh(tmp_path_factory, foil_points):
    d = tmp_path_factory.mktemp("mesh")
    nodes = ogrid_nodes(foil_points, n_radial=30, far_field_radius=20.0)
    write_polymesh(d / "constant/polyMesh", nodes, span=0.1)
    return d, nodes


def test_polymesh_round_trip_counts(written_mesh, foil_points):
    d, nodes = written_mesh
    m = PolyMesh(d / "constant/polyMesh")
    nt, nr = foil_points.shape[0], nodes.shape[1] - 1
    assert m.n_cells == nt * nr
    assert len(m.points) == 2 * nt * (nr + 1)
    assert len(m.faces) == nt * (2 * nr - 1) + 2 * nt + 2 * nt * nr
    assert m.boundary["airfoil"]["n_faces"] == nt
    assert m.boundary["farfield"]["n_faces"] == nt
    assert m.boundary["frontAndBack"]["n_faces"] == 2 * nt * nr
    assert m.boundary["airfoil"]["type"] == "wall"
    assert m.boundary["frontAndBack"]["type"] == "empty"


def test_polymesh_is_in_upper_triangular_order(written_mesh):
    d, _ = written_mesh
    m = PolyMesh(d / "constant/polyMesh")
    n = m.neighbour.size
    assert np.all(m.owner[:n] < m.neighbour)
    key = m.owner[:n].astype(np.int64) * (m.n_cells + 1) + m.neighbour
    assert np.all(np.diff(key) > 0)


def test_every_cell_is_closed(written_mesh):
    """各セルの面積ベクトルの総和はゼロ (閉じた多面体)."""
    d, _ = written_mesh
    m = PolyMesh(d / "constant/polyMesh")
    s = np.zeros((m.n_cells, 3))
    sf = m.face_areas
    np.add.at(s, m.owner, sf)
    np.add.at(s, m.neighbour, -sf[: m.neighbour.size])
    assert np.abs(s).max() < 1e-9 * np.linalg.norm(sf, axis=1).max()


def test_total_volume_matches_the_grid_area(written_mesh):
    d, nodes = written_mesh
    m = PolyMesh(d / "constant/polyMesh")
    fc, sf = m.face_centres, m.face_areas
    v = np.zeros(m.n_cells)
    np.add.at(v, m.owner, np.einsum("ij,ij->i", fc, sf) / 3.0)
    np.add.at(v, m.neighbour, -np.einsum("ij,ij->i", fc, sf)[: m.neighbour.size] / 3.0)
    assert v.min() > 0.0
    assert v.sum() == pytest.approx(cell_areas(nodes).sum() * 0.1, rel=1e-9)


def test_cell_centres_are_inside_their_cells(written_mesh):
    d, nodes = written_mesh
    m = PolyMesh(d / "constant/polyMesh")
    c = m.cell_centres
    assert np.all(np.abs(c[:, 2] - 0.05) < 1e-12)      # スパン中央
    assert np.isfinite(c).all()


def test_wall_normals_point_out_of_the_fluid(written_mesh, foil_points):
    """壁パッチの Sf は流体から見て外向き = 物体の中へ向く."""
    d, _ = written_mesh
    m = PolyMesh(d / "constant/polyMesh")
    f = m.patch_faces("airfoil")
    sf = m.face_areas[f]
    n_out = _node_normals(foil_points)                 # 物体から外向き
    assert np.all(np.einsum("ij,ij->i", sf[:, :2], n_out) < 0.0)


def test_far_field_normals_point_away(written_mesh):
    d, _ = written_mesh
    m = PolyMesh(d / "constant/polyMesh")
    f = m.patch_faces("farfield")
    sf, fc = m.face_areas[f], m.face_centres[f]
    r = fc[:, :2] - fc[:, :2].mean(axis=0)
    assert np.all(np.einsum("ij,ij->i", sf[:, :2], r) > 0.0)


# ------------------------------------------------------------------ 場の読み書き
def _write_field(path, cls, obj, internal, boundary):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n"
        f"    object {obj};\n}}\n// * * * //\n\n"
        f"dimensions [0 0 0 0 0 0 0];\n\ninternalField   {internal};\n\n"
        f"boundaryField\n{{\n{boundary}}}\n"
    )


def test_read_uniform_scalar_field(tmp_path):
    _write_field(tmp_path / "p", "volScalarField", "p", "uniform 3.5",
                 "    airfoil { type zeroGradient; }\n")
    f = read_volume_field(tmp_path / "p")
    assert f["n_components"] == 1
    assert f["internal"] == pytest.approx(3.5)
    assert f["boundary"]["airfoil"]["type"] == "zeroGradient"
    assert f["boundary"]["airfoil"]["value"] is None


def test_read_uniform_vector_field(tmp_path):
    _write_field(tmp_path / "U", "volVectorField", "U", "uniform (1 2 3)",
                 "    airfoil { type noSlip; }\n")
    f = read_volume_field(tmp_path / "U")
    assert f["n_components"] == 3
    assert f["internal"] == pytest.approx([1.0, 2.0, 3.0])


def test_read_nonuniform_field_with_boundary_values(tmp_path):
    _write_field(
        tmp_path / "p", "volScalarField", "p",
        "nonuniform List<scalar>\n3\n(\n1\n2\n3\n)",
        "    airfoil\n    {\n        type calculated;\n"
        "        value nonuniform List<scalar>\n2\n(\n7\n8\n)\n;\n    }\n",
    )
    f = read_volume_field(tmp_path / "p")
    assert f["internal"] == pytest.approx([1.0, 2.0, 3.0])
    assert f["boundary"]["airfoil"]["value"] == pytest.approx([7.0, 8.0])


# ------------------------------------------------------------------ 荷重の積分
@pytest.fixture(scope="module")
def synthetic_case(tmp_path_factory, foil_points):
    """一様な圧力場と静止流体の合成ケース."""
    d = tmp_path_factory.mktemp("case")
    nodes = ogrid_nodes(foil_points, n_radial=25, far_field_radius=20.0)
    write_polymesh(d / "constant/polyMesh", nodes, span=0.1)
    nc = foil_points.shape[0] * 25
    (d / "constant").mkdir(exist_ok=True)
    (d / "constant/transportProperties").write_text(
        "FoamFile\n{\n class dictionary;\n}\nnu   1e-4;\n")
    _write_field(d / "0/p", "volScalarField", "p", "uniform 5",
                 "    airfoil { type zeroGradient; }\n"
                 "    farfield { type fixedValue; value uniform 0; }\n"
                 "    frontAndBack { type empty; }\n")
    _write_field(d / "0/U", "volVectorField", "U", "uniform (0 0 0)",
                 "    airfoil { type noSlip; }\n"
                 "    farfield { type fixedValue; value uniform (0 0 0); }\n"
                 "    frontAndBack { type empty; }\n")
    return d, nc


def test_uniform_pressure_gives_zero_net_force(synthetic_case):
    """閉じた面上で sum(Sf) = 0 なので, 一様圧力の合力は厳密にゼロ."""
    d, _ = synthetic_case
    w = patch_wrench(d, "airfoil", time="0", rho=1.0)
    scale = w["area"] * 5.0
    assert np.abs(w["force_pressure"]).max() < 1e-10 * scale


def test_still_fluid_gives_zero_viscous_force(synthetic_case):
    d, _ = synthetic_case
    w = patch_wrench(d, "airfoil", time="0", rho=1.0)
    assert np.abs(w["force_viscous"]).max() < 1e-12


def test_patch_wrench_reads_nu_from_transport_properties(synthetic_case):
    d, _ = synthetic_case
    w = patch_wrench(d, "airfoil", time="0")
    assert w["time"] == "0"
    assert w["area"] > 0.0


def test_wall_area_matches_the_perimeter(synthetic_case, foil_points):
    d, _ = synthetic_case
    w = patch_wrench(d, "airfoil", time="0")
    per = np.linalg.norm(np.roll(foil_points, -1, axis=0) - foil_points,
                         axis=1).sum()
    assert w["area"] == pytest.approx(per * 0.1, rel=1e-9)


def test_integrate_coefficients_on_the_synthetic_case(synthetic_case):
    d, _ = synthetic_case
    out = integrate_coefficients(d, AirfoilCaseConfig(alpha_deg=0.0, span=0.1))
    assert out["Cd"] == pytest.approx(0.0, abs=1e-9)
    assert out["Cl"] == pytest.approx(0.0, abs=1e-9)
    assert out["n_samples"] == 1.0


def _case_with_wall_pressure(directory, points, cp_of_face):
    """壁面に任意の圧力分布を与えた合成ケースを作る."""
    nr = 20
    write_polymesh(directory / "constant/polyMesh",
                   ogrid_nodes(points, n_radial=nr, far_field_radius=20.0),
                   span=0.1)
    (directory / "constant/transportProperties").write_text(
        "FoamFile\n{\n class dictionary;\n}\nnu   1e-4;\n")
    m = PolyMesh(directory / "constant/polyMesh")
    f = m.patch_faces("airfoil")
    p_wall = cp_of_face(m.face_centres[f], m.face_areas[f]) * 0.5
    vals = "\n".join(f"{v:.10g}" for v in p_wall)
    _write_field(
        directory / "0/p", "volScalarField", "p", "uniform 0",
        "    airfoil\n    {\n        type calculated;\n"
        f"        value nonuniform List<scalar>\n{len(p_wall)}\n(\n{vals}\n)\n;\n    }}\n"
        "    farfield { type fixedValue; value uniform 0; }\n"
        "    frontAndBack { type empty; }\n")
    _write_field(directory / "0/U", "volVectorField", "U", "uniform (0 0 0)",
                 "    airfoil { type noSlip; }\n"
                 "    farfield { type fixedValue; value uniform (0 0 0); }\n"
                 "    frontAndBack { type empty; }\n")
    return m


def test_lift_behind_the_quarter_chord_is_nose_down(tmp_path, foil_points):
    """後ろ寄りに上向き荷重 -> CmPitch < 0 (頭下げ).

    ``CmPitch`` の符号が空力の慣用 (頭上げが正, `Airfoil.cm` と同じ) に
    なっていることを固定する。幾何的な (r x F)_z とは符号が逆になる。
    """
    def cp(fc, sf):
        # x > 0.6 の上面だけ吸い上げる (Sf は流体から外向き = 上面では下向き)
        return np.where((fc[:, 0] > 0.6) & (sf[:, 1] < 0.0), -1.0, 0.0)

    _case_with_wall_pressure(tmp_path, foil_points, cp)
    out = integrate_coefficients(tmp_path, AirfoilCaseConfig(alpha_deg=0.0,
                                                             span=0.1))
    assert out["Cl"] > 0.0
    assert out["CmPitch"] < 0.0


def test_lift_ahead_of_the_quarter_chord_is_nose_up(tmp_path, foil_points):
    def cp(fc, sf):
        return np.where((fc[:, 0] < 0.15) & (sf[:, 1] < 0.0), -1.0, 0.0)

    _case_with_wall_pressure(tmp_path, foil_points, cp)
    out = integrate_coefficients(tmp_path, AirfoilCaseConfig(alpha_deg=0.0,
                                                             span=0.1))
    assert out["Cl"] > 0.0
    assert out["CmPitch"] > 0.0


# ------------------------------------------------------------------ ケース生成
def test_case_write_produces_everything_without_openfoam(tmp_path):
    cfg = AirfoilCaseConfig(reynolds=8000.0, alpha_deg=5.0, n_surface=80,
                            n_radial=30, far_field_radius=20.0)
    case = OpenFoamCase(tmp_path / "c", ps.STAMPFLY_1209_SECTION, cfg)
    d = case.write()
    for f in ("system/controlDict", "system/fvSchemes", "system/fvSolution",
              "constant/transportProperties", "constant/turbulenceProperties",
              "0/U", "0/p", "constant/polyMesh/points",
              "constant/polyMesh/faces", "constant/polyMesh/owner",
              "constant/polyMesh/neighbour", "constant/polyMesh/boundary"):
        assert (d / f).is_file(), f
    assert not (d / "system/blockMeshDict").exists()   # ogrid なので不要


def test_nu_follows_the_reynolds_number(tmp_path):
    cfg = AirfoilCaseConfig(reynolds=5000.0, u_inf=1.0, n_surface=60, n_radial=20)
    d = OpenFoamCase(tmp_path / "c", circle_section(), cfg).write()
    txt = (d / "constant/transportProperties").read_text()
    assert "0.0002" in txt


def test_inflow_direction_follows_alpha(tmp_path):
    cfg = AirfoilCaseConfig(alpha_deg=30.0, n_surface=60, n_radial=20)
    d = OpenFoamCase(tmp_path / "c", circle_section(), cfg).write()
    txt = (d / "0/U").read_text()
    assert "0.8660254" in txt and "0.5" in txt


def test_function_objects_are_off_by_default(tmp_path):
    """この環境の OpenFOAM は functionObject が使えないので既定は無効."""
    d = OpenFoamCase(tmp_path / "c", circle_section(),
                     AirfoilCaseConfig(n_surface=60, n_radial=20)).write()
    assert "forceCoeffs" not in (d / "system/controlDict").read_text()


def test_blockmesh_backend_still_available_for_star_shaped_sections(tmp_path):
    cfg = AirfoilCaseConfig(mesh="blockmesh", n_surface=60, n_radial=20)
    d = OpenFoamCase(tmp_path / "c", circle_section(), cfg).write()
    assert (d / "system/blockMeshDict").is_file()


def test_unknown_mesh_backend_is_rejected(tmp_path):
    cfg = AirfoilCaseConfig(mesh="nonsense", n_surface=60, n_radial=20)
    with pytest.raises(ValueError):
        OpenFoamCase(tmp_path / "c", circle_section(), cfg).write()


def test_blockmesh_refuses_the_thin_cambered_section(foil_points):
    """1/4 コード中心の O 型格子は StampFly 断面では成立しない.

    黙って裏返った格子を吐くのではなく例外にする.
    """
    with pytest.raises(ValueError):
        build_blockmesh_dict(foil_points, centre=(0.25, 0.0))


def test_blockmesh_dict_is_well_formed_for_a_cylinder(cyl_points):
    txt = build_blockmesh_dict(cyl_points, n_radial=20, far_field_radius=20.0)
    assert txt.count("hex (") == cyl_points.shape[0]
    assert txt.count("(") == txt.count(")")
    for patch in ("airfoil", "farfield", "frontAndBack"):
        assert patch in txt
