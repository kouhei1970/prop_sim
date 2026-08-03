"""例 12: 制御実装むけの推力・トルク係数テーブルを作る.

    python examples/12_thrust_torque_table.py

何を作るか
----------
飛行制御では推力とトルクを ``T = C_T Omega^2``、``Q = C_Q Omega^2`` と
置くのが普通だが、機体が動くと係数が変わる。本スクリプトは

    T(Omega, V) = C_T0 * f_T(lambda, mu) * Omega^2
    Q(Omega, V) = C_Q0 * f_Q(lambda, mu) * Omega^2

    lambda = V_axial / (Omega R)   ... シャフト軸方向の速度比 (正 = 上昇)
    mu     = V_edge  / (Omega R)   ... ディスク面内の速度比

の補正関数 ``f_T``, ``f_Q`` を表にする。``lambda``、``mu`` は
**回転数によらない**ので、2 次元テーブル 1 枚で全回転数を覆える
(その検証も本スクリプトが行う)。

出力

    docs/report/data/kt_table.csv     f_T の表
    docs/report/data/kq_table.csv     f_Q の表
    docs/report/data/kt_table.h       C の配列 (そのまま貼れる)
    docs/report/data/kt_table.json    メタデータ + 検証結果
    docs/report/assets/kt_table.svg   図

注意
----
* **降下 (lambda < 0) は入れていない。** ボルテックスリング状態で運動量理論が
  成立せず、BEMT の値が信用できないため。
* 未較正のモデルなので ``C_T0`` の絶対値には ±20 % の幅がある。
  一方 ``f_T`` は比なので、絶対値の不確かさは打ち消し合って残らない。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # リポジトリ直下を参照

import numpy as np

import prop_sim as ps

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "report" / "data"
ASSETS = ROOT / "docs" / "report" / "assets"

LAMBDA = np.arange(0.0, 0.2501, 0.025)      # 軸方向速度比
MU = np.arange(0.0, 0.1501, 0.025)          # 面内速度比
RPM_REF = 27310.0                            # 表を作る基準回転数
RPM_CHECK = (18000.0, 27310.0, 38000.0)      # 回転数非依存性の検証用

rotor = ps.Rotor(ps.stampfly_1209())
model = ps.BEMT()
R = rotor.geometry.diameter / 2


def solve(rpm, lam, mu):
    """(lambda, mu) を OperatingPoint に直して解く."""
    om = rpm * 2 * np.pi / 60.0
    v_ax, v_ed = lam * om * R, mu * om * R
    op = ps.OperatingPoint(
        rpm=rpm, v_inf=float(np.hypot(v_ax, v_ed)),
        inflow_angle_deg=float(np.degrees(np.arctan2(v_ed, v_ax))),
    )
    sol = model.solve(rotor, op)
    return float(sol.thrust), float(sol.torque())


def build_table(rpm):
    t0, q0 = solve(rpm, 0.0, 0.0)
    ft = np.empty((LAMBDA.size, MU.size))
    fq = np.empty_like(ft)
    for i, lam in enumerate(LAMBDA):
        for j, mu in enumerate(MU):
            t, q = solve(rpm, float(lam), float(mu))
            ft[i, j], fq[i, j] = t / t0, q / q0
    return ft, fq, t0, q0


def check_rpm_independence():
    """複数の回転数で表を作り、ばらつきを測る."""
    tables = [build_table(n)[0] for n in RPM_CHECK]
    stack = np.stack(tables)
    spread = 100.0 * (stack.max(axis=0) / stack.min(axis=0) - 1.0)
    return float(spread.max()), spread


def to_c_array(name, arr):
    rows = ",\n".join(
        "  {" + ", ".join(f"{v:.5f}f" for v in row) + "}" for row in arr)
    return (f"static const float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n"
            f"{rows}\n}};\n")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    print("推力・トルク補正テーブルを作成中 ...")
    ft, fq, t0, q0 = build_table(RPM_REF)
    om0 = RPM_REF * 2 * np.pi / 60.0
    ct0, cq0 = t0 / om0**2, q0 / om0**2

    print("回転数非依存性を検証中 ...")
    max_spread, spread = check_rpm_independence()
    print(f"  {RPM_CHECK[0]:.0f}-{RPM_CHECK[-1]:.0f} rpm でのばらつき "
          f"最大 {max_spread:.2f} %")

    # ---- CSV
    for name, arr in (("kt_table", ft), ("kq_table", fq)):
        lines = ["lambda\\mu," + ",".join(f"{m:.3f}" for m in MU)]
        for lam, row in zip(LAMBDA, arr):
            lines.append(f"{lam:.3f}," + ",".join(f"{v:.5f}" for v in row))
        (DATA / f"{name}.csv").write_text("\n".join(lines) + "\n")
        print(f"  -> data/{name}.csv")

    # ---- C ヘッダ
    header = f"""/* StampFly 1209 推力・トルク補正テーブル (prop_sim, 未較正)
 *
 *   T = CT0 * f_T(lambda, mu) * omega^2      [N],   omega [rad/s]
 *   Q = CQ0 * f_Q(lambda, mu) * omega^2      [N m]
 *
 *   lambda = V_axial / (omega * R)   (正 = 上昇方向)
 *   mu     = V_edge  / (omega * R)
 *   R = {R:.6f} m
 *
 * lambda, mu は回転数によらないので本表 1 枚で全回転数を覆える
 * ({RPM_CHECK[0]:.0f}-{RPM_CHECK[-1]:.0f} rpm でばらつき最大 {max_spread:.2f} %)。
 * 降下 (lambda < 0) は未収録 — ボルテックスリング状態で理論が成立しない。
 */
#define PROP_R           {R:.6f}f
#define PROP_CT0         {ct0:.6e}f   /* N s^2   */
#define PROP_CQ0         {cq0:.6e}f   /* N m s^2 */
#define PROP_N_LAMBDA    {LAMBDA.size}
#define PROP_N_MU        {MU.size}
#define PROP_LAMBDA_STEP {LAMBDA[1] - LAMBDA[0]:.4f}f
#define PROP_MU_STEP     {MU[1] - MU[0]:.4f}f

{to_c_array("prop_ft", ft)}
{to_c_array("prop_fq", fq)}"""
    (DATA / "kt_table.h").write_text(header)
    print("  -> data/kt_table.h")

    meta = {
        "ct0_n_s2": ct0, "cq0_nm_s2": cq0, "radius_m": R,
        "lambda": LAMBDA.tolist(), "mu": MU.tolist(),
        "rpm_ref": RPM_REF, "rpm_checked": list(RPM_CHECK),
        "rpm_spread_max_pct": max_spread,
        "ft_at_lambda0_mu0": float(ft[0, 0]),
        "ft_min": float(ft.min()), "ft_max": float(ft.max()),
        "fq_min": float(fq.min()), "fq_max": float(fq.max()),
        "calibrated": False,
        "note": "降下 (lambda<0) は未収録。未較正なので CT0 には ±20% の幅がある。",
    }
    (DATA / "kt_table.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print("  -> data/kt_table.json")

    # ---- 図
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("IPAGothic", "Noto Sans CJK JP"):
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "legend.frameon": False,
                         "savefig.transparent": True})
    C = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for j, mu in enumerate(MU):
        if j % 2:
            continue
        ax[0].plot(LAMBDA, ft[:, j], color=C[j // 2], lw=1.8, label=f"{mu:.2f}")
        ax[1].plot(LAMBDA, fq[:, j], color=C[j // 2], lw=1.8)
    ax[0].axhline(1.0, color="0.6", lw=0.8)
    ax[0].set(xlabel="λ = V軸 / (Ω R)", ylabel=r"$f_T$", title="推力の補正係数")
    ax[0].legend(title="μ = V面内 / (Ω R)", fontsize=8)
    ax[1].axhline(1.0, color="0.6", lw=0.8)
    ax[1].set(xlabel="λ = V軸 / (Ω R)", ylabel=r"$f_Q$", title="トルクの補正係数")
    cs = ax[2].contourf(MU, LAMBDA, ft, levels=14, cmap="RdBu_r")
    ln = ax[2].contour(MU, LAMBDA, ft, levels=8, colors="k", linewidths=0.5,
                       alpha=0.55)
    ax[2].clabel(ln, fmt="%.2f", fontsize=7)
    fig.colorbar(cs, ax=ax[2], pad=0.02)
    ax[2].set(xlabel="μ", ylabel="λ", title=r"$f_T(\lambda, \mu)$")
    ax[2].grid(False)

    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / "kt_table.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    svg = path.read_text()
    for src in ("#000000", "#808080", "#999999"):
        svg = svg.replace(f"stroke: {src}", "stroke: currentColor")
    svg = svg.replace("stroke: #b0b0b0", "stroke: currentColor; stroke-opacity: 0.25")
    svg = svg.replace("<svg ", '<svg fill="currentColor" ', 1)
    i = svg.index(">", svg.index("<svg")) + 1
    svg = (svg[:i] + "<style>svg{color:#16202b}"
           "@media(prefers-color-scheme:dark){svg{color:#e7edf4}}</style>"
           + svg[i:])
    path.write_text(svg)
    print("  -> assets/kt_table.svg")

    print(f"\nCT0 = {ct0:.4e} N·s²,  CQ0 = {cq0:.4e} N·m·s²  (λ=μ=0)")
    print(f"f_T の範囲 {ft.min():.3f} - {ft.max():.3f}, "
          f"f_Q の範囲 {fq.min():.3f} - {fq.max():.3f}")


if __name__ == "__main__":
    main()
