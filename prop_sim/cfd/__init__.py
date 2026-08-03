"""CFD 連携 (OpenFOAM).

2 次元翼型ポーラの計算を担当する。3 次元ロータ CFD は本パッケージでは
回さず、外部で得た 6 分力を
:meth:`prop_sim.SurrogateModel.from_conditions` で取り込む方針。
"""

from .foam_io import PolyMesh, latest_time, patch_wrench, read_volume_field
from .mesh import cell_areas, ogrid_nodes, write_polymesh
from .openfoam import (
    AirfoilCaseConfig,
    OpenFoamCase,
    build_blockmesh_dict,
    find_openfoam,
    integrate_coefficients,
    polygon_centroid,
    read_force_coefficients,
    run_polar,
    solver_converged,
    surface_points,
)

__all__ = [
    "OpenFoamCase",
    "AirfoilCaseConfig",
    "find_openfoam",
    "surface_points",
    "polygon_centroid",
    "ogrid_nodes",
    "write_polymesh",
    "cell_areas",
    "build_blockmesh_dict",
    "read_force_coefficients",
    "integrate_coefficients",
    "solver_converged",
    "run_polar",
    "PolyMesh",
    "read_volume_field",
    "patch_wrench",
    "latest_time",
]
