"""OpenFOAM の ASCII ファイルを読んで壁面荷重を積分する.

なぜ自前で積分するのか
----------------------
``forceCoeffs`` などの functionObject を使うのが本来の筋だが, 環境に入る
OpenFOAM のビルドによっては **functionObject が一切使えない**ことがある.
たとえば Debian/Ubuntu の openfoam 1912 パッケージでは
``functionObjectList::read()`` が各 function の辞書の SHA1 を取る際に

    error in IOstream "sha1" for operation operator<<(Ostream&, const word&)

で落ちる (``sha1`` ストリームの状態バグ). *どの* functionObject でも
再現するので, ソルバ側では functions を一切使わず, 書き出された場から
Python で荷重を積分する方針にしてある. 副次的に

* OpenFOAM のバージョン依存が無くなる
* 圧力成分と粘性成分を分離して見られる
* テストできる (合成メッシュで幾何量を検算できる)

という利点がある.

積分の内容
----------
壁パッチ上で

    F_p    = sum  rho * p_f * Sf                (圧力)
    F_tau  = sum  rho * nu * A_f * (U_c - U_f)_t / d_perp    (粘性)

``Sf`` は流体領域から外向き (= 物体に向かう) 面積ベクトルなので, 上式は
そのまま **物体が受ける力**になる. 粘性項は境界の ``snGrad(U)`` の 1 次
近似で, OpenFOAM の ``forces`` が使う ``dev2`` 版と非圧縮・層流では
壁面上で一致する.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

__all__ = ["PolyMesh", "read_volume_field", "patch_wrench", "latest_time"]

_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
_FOAMFILE = re.compile(r"FoamFile\s*\{.*?\}", re.S)


def _strip(text: str) -> str:
    return _FOAMFILE.sub("", _COMMENT.sub("", text), count=1)


def _outer_block(text: str, start: int = 0) -> str:
    """``start`` 以降で最初に開く括弧に対応する中身を返す."""
    i = text.index("(", start)
    depth = 0
    for k in range(i, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return text[i + 1:k]
    raise ValueError("括弧が閉じていません")


def _floats(block: str, ncol: int) -> np.ndarray:
    a = np.fromstring(block.replace("(", " ").replace(")", " "), sep=" ")
    return a.reshape(-1, ncol) if ncol > 1 else a


# ------------------------------------------------------------------ メッシュ
class PolyMesh:
    """``constant/polyMesh`` の ASCII 版を読む最小限のリーダ."""

    def __init__(self, poly_dir):
        d = Path(poly_dir)
        self.points = _floats(_outer_block(_strip((d / "points").read_text())), 3)
        faces_txt = _strip((d / "faces").read_text())
        self.faces = [
            np.fromstring(m.group(2), dtype=int, sep=" ")
            for m in re.finditer(r"(\d+)\s*\(([^)]*)\)", _outer_block(faces_txt))
        ]
        self.owner = _floats(
            _outer_block(_strip((d / "owner").read_text())), 1).astype(int)
        nb = d / "neighbour"
        self.neighbour = (
            _floats(_outer_block(_strip(nb.read_text())), 1).astype(int)
            if nb.is_file() else np.empty(0, int)
        )
        self.boundary = self._read_boundary(d / "boundary")
        self._face_geom: tuple[np.ndarray, np.ndarray] | None = None
        self._cell_centres: np.ndarray | None = None

    @staticmethod
    def _read_boundary(path) -> dict[str, dict]:
        txt = _outer_block(_strip(Path(path).read_text()))
        out: dict[str, dict] = {}
        for m in re.finditer(r"(\w+)\s*\{(.*?)\}", txt, re.S):
            body = m.group(2)
            e = dict(re.findall(r"(\w+)\s+([^;{}]+);", body))
            out[m.group(1)] = {
                "type": e.get("type", "patch").strip(),
                "n_faces": int(e["nFaces"]),
                "start_face": int(e["startFace"]),
            }
        return out

    # ------------------------------------------------------------ 幾何
    @property
    def n_cells(self) -> int:
        return int(max(self.owner.max(),
                       self.neighbour.max() if self.neighbour.size else -1)) + 1

    def _compute_face_geom(self) -> None:
        """OpenFOAM と同じ三角形分割で面心と面積ベクトルを出す."""
        p = self.points
        ctr = np.empty((len(self.faces), 3))
        sf = np.empty((len(self.faces), 3))
        for k, f in enumerate(self.faces):
            q = p[f]
            if q.shape[0] == 3:
                sf[k] = 0.5 * np.cross(q[1] - q[0], q[2] - q[0])
                ctr[k] = q.mean(axis=0)
                continue
            c = q.mean(axis=0)
            a = q
            b = np.roll(q, -1, axis=0)
            n = np.cross(b - a, c - a)                    # 2 x 三角形面積
            area = 0.5 * np.linalg.norm(n, axis=1)
            tri_c = (a + b + c) / 3.0
            tot = area.sum()
            ctr[k] = (tri_c * area[:, None]).sum(axis=0) / tot if tot > 0 else c
            sf[k] = 0.5 * n.sum(axis=0)
        self._face_geom = (ctr, sf)

    @property
    def face_centres(self) -> np.ndarray:
        if self._face_geom is None:
            self._compute_face_geom()
        return self._face_geom[0]

    @property
    def face_areas(self) -> np.ndarray:
        """面積ベクトル ``Sf`` (owner から neighbour に向く)."""
        if self._face_geom is None:
            self._compute_face_geom()
        return self._face_geom[1]

    @property
    def cell_centres(self) -> np.ndarray:
        """ピラミッド分割による重心 (OpenFOAM の primitiveMesh と同じ手順)."""
        if self._cell_centres is not None:
            return self._cell_centres
        nc = self.n_cells
        fc, sf = self.face_centres, self.face_areas
        n_faces = np.zeros(nc)
        est = np.zeros((nc, 3))
        np.add.at(est, self.owner, fc)
        np.add.at(n_faces, self.owner, 1.0)
        if self.neighbour.size:
            np.add.at(est, self.neighbour, fc[: self.neighbour.size])
            np.add.at(n_faces, self.neighbour, 1.0)
        est /= n_faces[:, None]

        vol = np.zeros(nc)
        num = np.zeros((nc, 3))

        def accumulate(cells, faces, sign):
            d = fc[faces] - est[cells]
            pyr_v = sign * np.einsum("ij,ij->i", d, sf[faces]) / 3.0
            pyr_c = 0.75 * fc[faces] + 0.25 * est[cells]
            np.add.at(vol, cells, pyr_v)
            np.add.at(num, cells, pyr_v[:, None] * pyr_c)

        allf = np.arange(len(self.faces))
        accumulate(self.owner, allf, +1.0)
        if self.neighbour.size:
            nf = np.arange(self.neighbour.size)
            accumulate(self.neighbour, nf, -1.0)
        self._cell_centres = num / vol[:, None]
        return self._cell_centres

    def patch_faces(self, name: str) -> np.ndarray:
        b = self.boundary[name]
        return np.arange(b["start_face"], b["start_face"] + b["n_faces"])


# -------------------------------------------------------------------- 場
def read_volume_field(path) -> dict:
    """``volScalarField`` / ``volVectorField`` を読む.

    Returns
    -------
    dict
        ``internal`` (``(N,)`` または ``(N,3)``) と ``boundary``
        (パッチ名 -> ``{"type": str, "value": ndarray | None}``).
    """
    txt = _strip(Path(path).read_text())
    ncol = 3 if re.search(r"\bvolVectorField\b",
                          Path(path).read_text()) else 1

    def parse_value(s: str, tag: str) -> np.ndarray | None:
        m = re.search(rf"{tag}\s+uniform\s+(\([^)]*\)|[-\d.eE+]+)\s*;", s)
        if m:
            v = _floats(m.group(1), ncol)
            # uniform はスカラなら 0 次元, ベクトルなら (3,) にする
            return v[0] if ncol > 1 else float(v[0])
        m = re.search(rf"{tag}\s+nonuniform", s)
        if m:
            if re.match(r"\s*List<\w+>\s*0\s*\(\s*\)", s[m.end():]):
                return np.empty((0, ncol) if ncol > 1 else 0)
            return _floats(_outer_block(s, m.end()), ncol)
        return None

    internal = parse_value(txt, "internalField")
    bnd: dict[str, dict] = {}
    m = re.search(r"boundaryField\s*\{", txt)
    if m:
        body = _brace_block(txt, m.end() - 1)
        for name, sub in _sub_blocks(body):
            t = re.search(r"type\s+([\w.]+)\s*;", sub)
            bnd[name] = {
                "type": t.group(1) if t else "",
                "value": parse_value(sub, "value"),
            }
    return {"internal": internal, "boundary": bnd, "n_components": ncol}


def _brace_block(text: str, open_idx: int) -> str:
    depth = 0
    for k in range(open_idx, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:k]
    raise ValueError("波括弧が閉じていません")


def _sub_blocks(body: str):
    i = 0
    while True:
        m = re.compile(r"([\w.\"|]+)\s*\{").search(body, i)
        if not m:
            return
        blk = _brace_block(body, m.end() - 1)
        yield m.group(1), blk
        i = m.end() + len(blk) + 1


def _cell_values(field: dict, idx: np.ndarray) -> np.ndarray:
    """指定セルの内部値. ``uniform`` の場合は展開する."""
    ncol = field["n_components"]
    v = np.asarray(field["internal"], float)
    if v.ndim == (1 if ncol > 1 else 0):          # uniform
        return np.broadcast_to(
            v, (idx.size, ncol) if ncol > 1 else (idx.size,)).copy()
    return v[idx]


def _boundary_array(field: dict, patch: str, mesh: PolyMesh) -> np.ndarray:
    """パッチ上の面値. ``value`` が無い型は境界条件から復元する."""
    ncol = field["n_components"]
    faces = mesh.patch_faces(patch)
    own = mesh.owner[faces]
    b = field["boundary"].get(patch, {"type": "zeroGradient", "value": None})
    v = b["value"]
    if v is not None and np.asarray(v).size:
        v = np.asarray(v, float)
        if v.ndim == (1 if ncol > 1 else 0):        # uniform
            return np.broadcast_to(v, (faces.size, ncol) if ncol > 1
                                   else (faces.size,)).copy()
        return v
    t = b["type"]
    if t in ("noSlip",):
        return np.zeros((faces.size, ncol) if ncol > 1 else faces.size)
    if t in ("zeroGradient", "calculated", "extrapolatedCalculated",
             "fixedFluxExtrapolatedPressure", ""):
        return _cell_values(field, own)
    raise ValueError(f"境界条件 {t!r} の面値を復元できません ({patch})")


def latest_time(case_dir) -> str:
    """時間ディレクトリのうち最大のもの (``0`` を含む)."""
    times = []
    for p in Path(case_dir).iterdir():
        if p.is_dir():
            try:
                times.append((float(p.name), p.name))
            except ValueError:
                pass
    if not times:
        raise FileNotFoundError(f"{case_dir} に時間ディレクトリがありません")
    return max(times)[1]


def patch_wrench(
    case_dir,
    patch: str = "airfoil",
    *,
    time: str | None = None,
    rho: float = 1.0,
    nu: float | None = None,
    centre=(0.25, 0.0, 0.0),
    mesh: PolyMesh | None = None,
) -> dict:
    """壁パッチ上の圧力・粘性荷重を積分する.

    Returns
    -------
    dict
        ``force`` / ``force_pressure`` / ``force_viscous`` (N),
        ``moment`` (``centre`` まわり), ``time``, ``area``.
    """
    case_dir = Path(case_dir)
    mesh = mesh or PolyMesh(case_dir / "constant/polyMesh")
    t = time or latest_time(case_dir)
    if nu is None:
        nu = float(re.search(
            r"\bnu\s+([-\d.eE+]+)\s*;",
            _strip((case_dir / "constant/transportProperties").read_text())
        ).group(1))

    p_field = read_volume_field(case_dir / t / "p")
    u_field = read_volume_field(case_dir / t / "U")

    faces = mesh.patch_faces(patch)
    own = mesh.owner[faces]
    sf = mesh.face_areas[faces]                  # 流体から外向き = 物体向き
    area = np.linalg.norm(sf, axis=1)
    nvec = sf / area[:, None]

    p_f = _boundary_array(p_field, patch, mesh)
    f_p = rho * (p_f[:, None] * sf)

    u_f = _boundary_array(u_field, patch, mesh)
    u_c = _cell_values(u_field, own)
    d = mesh.cell_centres[own] - mesh.face_centres[faces]
    d_perp = np.abs(np.einsum("ij,ij->i", d, nvec))
    du = u_c - u_f
    du_t = du - np.einsum("ij,ij->i", du, nvec)[:, None] * nvec
    f_t = rho * nu * area[:, None] * du_t / d_perp[:, None]

    f = f_p + f_t
    r = mesh.face_centres[faces] - np.asarray(centre, float)
    return {
        "time": t,
        "force": f.sum(axis=0),
        "force_pressure": f_p.sum(axis=0),
        "force_viscous": f_t.sum(axis=0),
        "moment": np.cross(r, f).sum(axis=0),
        "area": float(area.sum()),
        "mesh": mesh,
    }
