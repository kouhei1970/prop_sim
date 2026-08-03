"""例 11: StampFly 1209 の仮想風洞試験 — 制御系設計のためのデータ生成.

`docs/report/` の報告ページに載せる図と数値を全部ここで作る。

    python examples/11_virtual_wind_tunnel.py

出力

    docs/report/assets/*.svg   図
    docs/report/data/*.csv     生データ
    docs/report/data/summary.json  本文に埋め込む代表値

内容

    A. 静止特性 (推力・トルク・効率) と k_T, k_Q の同定
    B. 前進飛行マップ (速度 x 流入角) — 推力・面内力・ハブモーメント
    C. ホバー点まわりの線形微係数 (制御則の設計に直接使う量)
    D. 過渡応答 (動的インフロー + 回転慣性)
    E. 6 分力の変動成分 (BPF) と IMU / ノッチフィルタへの示唆
    F. 4 発機体レベルへの換算
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # リポジトリ直下を参照

import numpy as np

import prop_sim as ps
from prop_sim.experiments import dynamic_run, static_sweep

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "report"
ASSETS = OUT / "assets"
DATA = OUT / "data"

G = 9.80665
MASS_G = 36.8            # StampFly の機体重量 [g]
N_ROTOR = 4
DIAGONAL_M = 0.065       # モータ間の対角距離 [m] (StampFly 実寸)
ARM_M = DIAGONAL_M / 2   # ロータ中心〜機体中心 [m]
#: 機体構成. StampFly は X 配置で、上から見て 前右/後左 が反時計回り (CCW)、
#: 前左/後右 が時計回り (CW)。ハブ座標系は z が上向きなので CCW = spin +1。
LAYOUT = "X"
#: プロペラガード (ダクト) の **入口 (上端) 内径** [m]。外形図の φ36.2。
#: ダクトは上面図で二重円に見え、**下に向かって細くなるテーパ形状**なので、
#: プロペラ回転面での実効ボアはこれより小さい。その値は図に寸法が無い。
#: したがって φ36.2 から出す翼端すきまは **上限** にしかならない。
DUCT_INLET_BORE_M = 0.0362
DUCT_BORE_M = DUCT_INLET_BORE_M
SPIN_BY_ARM = {"front_right": +1, "rear_left": +1,
               "front_left": -1, "rear_right": -1}
HOVER_GF = MASS_G / N_ROTOR

rotor = ps.Rotor(ps.stampfly_1209())
model = ps.BEMT()
summary: dict = {}


# ------------------------------------------------------------------ 図の体裁
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("IPAGothic", "Noto Sans CJK JP", "TakaoGothic"):
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "savefig.transparent": True,
    })
    return plt


plt = setup_matplotlib()
C = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]

#: <img> 経由でもライト/ダークに追従させるために SVG 内に差し込む配色
_SVG_THEME = (
    "<style>svg{color:#16202b}"
    "@media(prefers-color-scheme:dark){svg{color:#e7edf4}}</style>"
)


def save(fig, name):
    """SVG で保存し, 文字と軸を ``currentColor`` に置き換える.

    報告ページはライト/ダークの両方で表示されるので, 図の文字が黒に
    焼き付いていると片方で読めなくなる. matplotlib が出す黒を
    ``currentColor`` にしておけば, CSS 側の文字色に自動で追従する
    (データ系列の色は明示指定なのでそのまま残る).
    """
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    fig.savefig(path, format="svg")
    plt.close(fig)
    svg = path.read_text()
    for src in ("#000000", "#808080", "#999999"):
        svg = svg.replace(f"stroke: {src}", "stroke: currentColor")
    svg = svg.replace("stroke: #b0b0b0", "stroke: currentColor; stroke-opacity: 0.25")
    # 文字はグリフのパス参照で fill を持たないので, ルートで currentColor にする
    svg = svg.replace("<svg ", '<svg fill="currentColor" ', 1)
    # <img> で読み込まれると外側のページの色は届かないので, SVG 自身にも
    # 配色を持たせる (ページに直接インライン展開する場合は build.py が外す)
    i = svg.index(">", svg.index("<svg")) + 1
    svg = svg[:i] + _SVG_THEME + svg[i:]
    path.write_text(svg)
    print(f"  -> assets/{name}")


def find_rpm(thrust_gf, **kw):
    """指定推力になる回転数を二分法で求める."""
    lo, hi = 2000.0, 60000.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        t = model.solve(rotor, ps.OperatingPoint(rpm=mid, **kw)).thrust / G * 1e3
        lo, hi = (mid, hi) if t < thrust_gf else (lo, mid)
    return 0.5 * (lo + hi)


# =========================================================== A. 静止特性
def part_a():
    print("A. 静止特性")
    rpm = np.arange(6000.0, 45001.0, 1000.0)
    res = static_sweep(rotor, model, rpm=rpm)
    c = res.columns
    t_gf = c["thrust"] / G * 1e3
    omega = rpm * 2 * np.pi / 60.0

    # T = k_T omega^2 を高回転側 (20k-45k) で同定する
    hi = rpm >= 20000.0
    k_t = float(np.sum(c["thrust"][hi] * omega[hi] ** 2) / np.sum(omega[hi] ** 4))
    k_q = float(np.sum(c["torque"][hi] * omega[hi] ** 2) / np.sum(omega[hi] ** 4))
    t_fit = k_t * omega**2
    err = 100.0 * (c["thrust"] / t_fit - 1.0)

    hover_rpm = find_rpm(HOVER_GF)
    op_h = ps.OperatingPoint(rpm=hover_rpm)
    sol_h = model.solve(rotor, op_h)

    summary["static"] = {
        "k_t": k_t, "k_q": k_q, "k_q_over_k_t": k_q / k_t,
        "hover_rpm": hover_rpm,
        "hover_omega": hover_rpm * 2 * np.pi / 60.0,
        "hover_thrust_gf": HOVER_GF,
        "hover_power_w": float(sol_h.power(op_h)),
        "hover_torque_mnm": float(sol_h.torque() * 1e3),
        "hover_fm": float(sol_h.coefficients(rotor, op_h)["FM"]),
        "hover_gf_per_w": float(HOVER_GF / sol_h.power(op_h)),
        "quad_law_error_pct_at_8000": float(err[rpm == 8000.0][0]),
        "quad_law_error_pct_at_15000": float(err[rpm == 15000.0][0]),
        "max_gf_per_w": float(np.max(t_gf / c["power_shaft"])),
        "rpm_at_max_gf_per_w": float(rpm[np.argmax(t_gf / c["power_shaft"])]),
        "re_075_hover": float(0.75 * rotor.geometry.diameter / 2
                              * hover_rpm * 2 * np.pi / 60.0
                              * float(rotor.geometry.chord(np.array(0.75)))
                              / 1.46e-5),
    }

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    ax[0].plot(rpm / 1e3, t_gf, color=C[0], lw=1.8, label="BEMT")
    ax[0].plot(rpm / 1e3, t_fit / G * 1e3, "--", color=C[1], lw=1.2,
               label=r"$k_T\,\Omega^2$ 近似")
    ax[0].axhline(HOVER_GF, color="0.5", lw=0.8, ls=":")
    ax[0].axvline(hover_rpm / 1e3, color="0.5", lw=0.8, ls=":")
    ax[0].set(xlabel="回転数 [krpm]", ylabel="推力 [gf]", title="推力")
    ax[0].legend()

    ax[1].plot(rpm / 1e3, err, color=C[2], lw=1.8)
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].fill_between([20, 45], -1, 1, color=C[2], alpha=0.10)
    ax[1].set(xlabel="回転数 [krpm]", ylabel="誤差 [%]",
              title=r"$\Omega^2$ 則からのずれ")

    ax[2].plot(rpm / 1e3, t_gf / c["power_shaft"], color=C[0], lw=1.8)
    ax2 = ax[2].twinx()
    ax2.plot(rpm / 1e3, c["FM"], color=C[3], lw=1.8)
    ax2.set_ylabel("FM", color=C[3])
    ax2.grid(False)
    ax[2].axvline(hover_rpm / 1e3, color="0.5", lw=0.8, ls=":")
    ax[2].set(xlabel="回転数 [krpm]", ylabel="推力効率 [gf/W]",
              title="効率 (軸動力基準)")
    save(fig, "static.svg")
    res.to_csv(DATA / "static_sweep.csv")
    return hover_rpm


# ================================================== B. 前進飛行マップ
def part_b(hover_rpm):
    print("B. 前進飛行マップ")
    v = np.arange(0.0, 10.01, 0.5)
    angles = [0.0, 30.0, 60.0, 90.0]
    out = {}
    for a in angles:
        r = static_sweep(rotor, model, rpm=hover_rpm, v_inf=v,
                         inflow_angle_deg=a)
        out[a] = r.columns
    t0 = out[0.0]["thrust"][0]

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for a, col in zip(angles, C):
        d = out[a]
        ax[0].plot(v, d["thrust"] / t0, color=col, lw=1.8, label=f"{a:.0f}°")
        ax[1].plot(v, d["Fx_true"] / G * 1e3, color=col, lw=1.8)
        ax[2].plot(v, d["My_true"] * 1e6, color=col, lw=1.8)
    ax[0].axhline(1.0, color="0.6", lw=0.8)
    ax[0].set(xlabel="対気速度 [m/s]", ylabel=r"$F_z / F_{z,0}$",
              title="推力 (ホバー回転数固定)")
    ax[0].legend(title="流入角")
    ax[1].set(xlabel="対気速度 [m/s]", ylabel=r"$F_x$ [gf]",
              title="面内力 (H 力)")
    ax[2].set(xlabel="対気速度 [m/s]", ylabel=r"$M_y$ [μN·m]",
              title="ハブモーメント")
    save(fig, "forward.svg")

    # 2 次元マップ (等高線).
    # static_sweep は引数の**直積**を掃引するので, meshgrid を渡してはいけない
    # (399 点のつもりが 399 x 399 になる). 1 次元の軸をそのまま渡して
    # 結果を (v, angle) の順で reshape する.
    vv = np.arange(0.0, 10.01, 0.5)
    aa = np.arange(0.0, 90.01, 5.0)
    V, A = np.meshgrid(vv, aa)
    r = static_sweep(rotor, model, rpm=hover_rpm, v_inf=vv,
                     inflow_angle_deg=aa, n_average=64)

    def grid(col, scale=1.0):
        return (r.columns[col] * scale).reshape(vv.size, aa.size).T

    fz = grid("thrust", 1.0 / t0)
    fx = grid("Fx_true", 1e3 / G)
    my = grid("My_true", 1e6)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
    for k, (z, ttl, fmt) in enumerate([
            (fz, r"推力比 $F_z/F_{z,0}$", "%.2f"),
            (fx, r"面内力 $F_x$ [gf]", "%.2f"),
            (my, r"ハブモーメント $M_y$ [μN·m]", "%.0f")]):
        cs = ax[k].contourf(V, A, z, levels=14, cmap="RdBu_r" if k == 0 else "viridis")
        ln = ax[k].contour(V, A, z, levels=7, colors="k", linewidths=0.5, alpha=0.55)
        ax[k].clabel(ln, fmt=fmt, fontsize=7)
        fig.colorbar(cs, ax=ax[k], pad=0.02)
        ax[k].set(xlabel="対気速度 [m/s]",
                  ylabel="流入角 [deg]" if k == 0 else "", title=ttl)
        ax[k].grid(False)
    save(fig, "map.svg")

    def at(a, vel):
        d = out[a]
        i = int(np.argmin(np.abs(v - vel)))
        return {"fz_ratio": float(d["thrust"][i] / t0),
                "fx_gf": float(d["Fx_true"][i] / G * 1e3),
                "my_unm": float(d["My_true"][i] * 1e6),
                "mx_unm": float(d["Mx_true"][i] * 1e6),
                "j": float(d["J"][i])}

    summary["forward"] = {
        "axial_4": at(0.0, 4.0), "axial_8": at(0.0, 8.0),
        "edge_4": at(90.0, 4.0), "edge_8": at(90.0, 8.0),
        "tilt60_4": at(60.0, 4.0), "tilt60_8": at(60.0, 8.0),
    }
    for a in angles:
        static_sweep(rotor, model, rpm=hover_rpm, v_inf=v,
                     inflow_angle_deg=a).to_csv(DATA / f"forward_{a:.0f}deg.csv")


# ============================================ C. ホバー点の線形微係数
def part_c(hover_rpm):
    print("C. 線形微係数")
    om0 = hover_rpm * 2 * np.pi / 60.0
    base = model.solve(rotor, ps.OperatingPoint(rpm=hover_rpm))
    t0 = base.thrust

    def wrench(**kw):
        """ハブに立つ全レンチ (空力 + 慣性).

        ``RotorSolution.wrench`` は**空力だけ**なので, これだけで機体角速度に
        対する微分を取るとジャイロモーメントが丸ごと落ちる. 慣性項を足して
        static_sweep と同じ「試験台が受けるレンチ」に揃える.
        """
        op = ps.OperatingPoint(rpm=hover_rpm, **kw)
        aero = model.solve(rotor, op).wrench
        inertial = rotor.inertial_wrench(op, psi1=0.0, omega_dot=0.0,
                                         include_unbalance=False)
        return aero + inertial, op

    # 回転数微分
    d = 200.0
    tp = model.solve(rotor, ps.OperatingPoint(rpm=hover_rpm + d)).thrust
    tm = model.solve(rotor, ps.OperatingPoint(rpm=hover_rpm - d)).thrust
    qp = model.solve(rotor, ps.OperatingPoint(rpm=hover_rpm + d)).torque()
    qm = model.solve(rotor, ps.OperatingPoint(rpm=hover_rpm - d)).torque()
    dom = 2 * d * 2 * np.pi / 60.0
    dt_dom = (tp - tm) / dom
    dq_dom = (qp - qm) / dom

    # 上昇速度 (軸流) 微分: 流入角 0, v_inf = +w は下降流 = 上昇に相当
    dv = 0.5
    tw = model.solve(rotor, ps.OperatingPoint(rpm=hover_rpm, v_inf=dv,
                                              inflow_angle_deg=0.0)).thrust
    dt_dw = (tw - t0) / dv

    # 水平速度 (横流れ) 微分
    we, _ = wrench(v_inf=dv, inflow_angle_deg=90.0)
    w0, _ = wrench(v_inf=0.0)
    dfx_du = (we.force[0] - w0.force[0]) / dv
    dfz_du = (we.force[2] - w0.force[2]) / dv
    dmy_du = (we.moment[1] - w0.moment[1]) / dv

    # 機体角速度 (ピッチレート q) 微分. ジャイロは Mx 側に出る:
    #   -omega x (Jz Omega ez) = -(0,q,0) x (0,0,JzOmega) = (-q Jz Omega, 0, 0)
    dq = 2.0
    wq, _ = wrench(body_rate=np.array([0.0, dq, 0.0]))
    dmy_dq = (wq.moment[1] - w0.moment[1]) / dq
    dmx_dq = (wq.moment[0] - w0.moment[0]) / dq
    # 空力だけの寄与も分けて出す (どちらが効いているか示すため)
    aq = model.solve(rotor, ps.OperatingPoint(
        rpm=hover_rpm, body_rate=np.array([0.0, dq, 0.0]))).wrench
    a0 = model.solve(rotor, ps.OperatingPoint(rpm=hover_rpm)).wrench
    dmy_dq_aero = (aq.moment[1] - a0.moment[1]) / dq
    dmx_dq_aero = (aq.moment[0] - a0.moment[0]) / dq

    # 動的インフローの時定数
    a_disk = rotor.geometry.disk_area
    vh = float(np.sqrt(t0 / (2 * 1.225 * a_disk)))
    nu = vh / (om0 * (rotor.geometry.diameter / 2))
    tau = 0.85 / (4.0 * nu * om0)

    summary["derivatives"] = {
        "dT_domega": float(dt_dom),
        "dT_domega_norm": float(2 * t0 / om0),
        "dQ_domega": float(dq_dom),
        "dT_dw": float(dt_dw),
        "dT_dw_per_rotor_g_per_ms": float(dt_dw / G * 1e3),
        "heave_damping_4rotor": float(N_ROTOR * dt_dw / (MASS_G * 1e-3)),
        "dFx_du": float(dfx_du),
        "dFx_du_4rotor": float(N_ROTOR * dfx_du / (MASS_G * 1e-3)),
        "dFz_du": float(dfz_du),
        "dMy_du": float(dmy_du),
        "dMy_du_4rotor_unm": float(N_ROTOR * dmy_du * 1e6),
        "dMy_dq": float(dmy_dq),
        "dMx_dq": float(dmx_dq),
        "dMy_dq_aero": float(dmy_dq_aero),
        "dMx_dq_aero": float(dmx_dq_aero),
        "v_hover_induced": vh,
        "inflow_tau_ms": float(tau * 1e3),
        "inflow_bw_hz": float(1.0 / (2 * np.pi * tau)),
        "polar_inertia": float(rotor.geometry.polar_inertia),
        "spin_momentum": float(rotor.geometry.polar_inertia * om0),
    }
    # 操作力 (X 配置). 4 発すべてがロール・ピッチに寄与し, 腕の
    # ロール/ピッチ軸まわりの有効長は L*sin(45°) = L/sqrt(2) になる。
    #   ロール = 2 * (T+ - T-) * L/sqrt(2) = sqrt(2) * (T+ - T-) * L
    # 同じ Δrpm なら + 配置 (2 発 x 腕 L) の sqrt(2) 倍になる。
    # ヨーは反トルクの差なので回転方向の割り付けで決まる。
    lever = ARM_M / np.sqrt(2.0)
    for dn in (500.0, 1000.0, 2000.0):
        sp = model.solve(rotor, ps.OperatingPoint(rpm=hover_rpm + dn))
        sm = model.solve(rotor, ps.OperatingPoint(rpm=hover_rpm - dn))
        d = summary["derivatives"]
        d[f"roll_moment_dn{dn:.0f}_mnm"] = float(2.0 * (sp.thrust - sm.thrust)
                                                 * lever * 1e3)
        d[f"roll_moment_plus_dn{dn:.0f}_mnm"] = float((sp.thrust - sm.thrust)
                                                      * ARM_M * 1e3)
        d[f"yaw_moment_dn{dn:.0f}_mnm"] = float(2.0 * abs(sp.torque() - sm.torque())
                                                * 1e3)
    summary["derivatives"]["lever_x_mm"] = float(lever * 1e3)


# ======================== C2. シュラウド (プロペラガード) の効き幅
def part_c2(hover_rpm):
    """翼端損失を切った場合との差を出す.

    StampFly はプロペラが薄い筒状のリングでガードされている。本パッケージは
    開放ロータのモデルなのでシュラウドは扱えないが、シュラウドの効果のうち
    **翼端渦の抑制**だけは「Prandtl の翼端損失を切る」ことで上限を見積もれる。
    実際のガードは翼端すきまが有限なのでこの一部しか回収できない。
    """
    print("C2. シュラウドの効き幅 (翼端損失の上限)")
    notip = ps.BEMT(tip_loss=False)
    op = ps.OperatingPoint(rpm=hover_rpm)
    a, b = model.solve(rotor, op), notip.solve(rotor, op)

    def find(m, gf=HOVER_GF):
        lo, hi = 2000.0, 60000.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            th = m.solve(rotor, ps.OperatingPoint(rpm=mid)).thrust / G * 1e3
            lo, hi = (mid, hi) if th < gf else (lo, mid)
        return 0.5 * (lo + hi)

    r_a, r_b = find(model), find(notip)
    op_a, op_b = ps.OperatingPoint(rpm=r_a), ps.OperatingPoint(rpm=r_b)
    summary["shroud"] = {
        "thrust_gain_pct": float(100.0 * (b.thrust / a.thrust - 1.0)),
        "fm_open": float(a.coefficients(rotor, op)["FM"]),
        "fm_no_tip_loss": float(b.coefficients(rotor, op)["FM"]),
        "hover_rpm_open": float(r_a),
        "hover_rpm_no_tip_loss": float(r_b),
        "hover_rpm_change_pct": float(100.0 * (r_b / r_a - 1.0)),
        "power_open_w": float(model.solve(rotor, op_a).power(op_a)),
        "power_no_tip_loss_w": float(notip.solve(rotor, op_b).power(op_b)),
        # 理想ダクト (収縮なし) の運動量理論による同動力での推力比 (2*sigma)^(1/3)
        "ideal_duct_thrust_gain_pct": float(100.0 * (2.0 ** (1.0 / 3.0) - 1.0)),
        # 外形図から読んだ翼端すきま。ダクトが翼端渦を抑えられるかは
        # すきま / 翼半径 で決まり、目安は 1 % 以下。
        "duct_bore_mm": DUCT_BORE_M * 1e3,
        # 入口径から出した翼端すきま = **上限**。テーパのぶん実際は小さい。
        "tip_gap_max_mm": float((DUCT_BORE_M - rotor.geometry.diameter) / 2 * 1e3),
        "tip_gap_over_radius_max_pct": float(
            100.0 * (DUCT_BORE_M - rotor.geometry.diameter)
            / rotor.geometry.diameter),
        # ダクトとして効き始める目安 (delta/R = 2%) を満たす実効ボア
        "bore_for_2pct_mm": float(rotor.geometry.diameter * 1.02 * 1e3),
        # 外形図の比例読み: 上面図でプロペラ翼端が内円のすぐ内側まで届いており、
        # 内円 / プロペラ径 は 1.02-1.05 と読める。目視なので幅を持たせる。
        "bore_read_lo_mm": float(rotor.geometry.diameter * 1.02 * 1e3),
        "bore_read_hi_mm": float(rotor.geometry.diameter * 1.05 * 1e3),
        "tip_gap_read_lo_mm": float(rotor.geometry.diameter * 0.01 * 1e3),
        "tip_gap_read_hi_mm": float(rotor.geometry.diameter * 0.025 * 1e3),
        "tip_gap_over_radius_read_lo_pct": 2.0,
        "tip_gap_over_radius_read_hi_pct": 5.0,
        # 隣り合うダクトどうしのすきま (肉厚は不明なので内径基準の上限)
        "duct_gap_mm": float((DIAGONAL_M / np.sqrt(2.0) - DUCT_BORE_M) * 1e3),
        "duct_spacing_ratio": float(DIAGONAL_M / np.sqrt(2.0) / DUCT_BORE_M),
        "footprint_mm": float((DIAGONAL_M / np.sqrt(2.0) + DUCT_BORE_M) * 1e3),
    }
    summary["shroud"]["power_change_pct"] = float(
        100.0 * (summary["shroud"]["power_no_tip_loss_w"]
                 / summary["shroud"]["power_open_w"] - 1.0))


# ============================ C3. ダクト断面の縮尺図 (外形図の比例読み)
def part_c3():
    """外形図から読んだダクト断面を縮尺どおりに描く.

    外形図 (M5Stack StampFly v1.1) で寸法線があるのは **外径 φ36.2** だけ。
    上面図ではダクトが二重円に見え、断面は **上が広がり (ベルマウス)、
    下はほぼ一定径の筒 (スロート)** になっている。
    プロペラ翼端は上面図で外円の 0.86 倍の位置にあり、実測比
    31.21/36.2 = 0.862 と一致するので図は縮尺どおりと確認できる。
    スロート径には寸法が無いので幅を持たせて図示する。
    """
    print("C3. ダクト断面の縮尺図")
    d_out = 36.2
    d_prop = rotor.geometry.diameter * 1e3
    h = 5.5                       # ダクト高さ (側面図からの比例読み)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    for d_throat, col, ls in ((32.0, C[3], "-"), (34.2, C[4], "--")):
        r_t, r_o = d_throat / 2, d_out / 2
        z_flare = 0.42 * h                     # ベルマウスの高さぶん
        zz = np.linspace(0, z_flare, 40)
        # ベルマウス: 上に向かって r_t -> r_o へ 1/4 円弧的に広がる
        rr = r_t + (r_o - r_t) * np.sin(0.5 * np.pi * zz / z_flare)
        for s in (+1, -1):
            ax.plot(s * np.concatenate([[r_t], rr]),
                    np.concatenate([[-h + z_flare], zz + (-h + z_flare)]),
                    ls, color=col, lw=1.8,
                    label=(f"ダクト内壁 (スロート φ{d_throat})"
                           if s > 0 else None))
            ax.plot([s * r_t, s * r_t], [-h, -h + z_flare], ls, color=col, lw=1.8)

    # プロペラ回転面
    ax.plot([-d_prop / 2, d_prop / 2], [-0.45 * h, -0.45 * h], color=C[1],
            lw=2.6, label=f"プロペラ回転面 φ{d_prop:.2f} (実測)")
    ax.annotate("", xy=(d_prop / 2, -0.45 * h), xytext=(16.0, -0.45 * h),
                arrowprops=dict(arrowstyle="<->", color=C[2], lw=1.1))
    ax.text(14.6, -0.45 * h + 0.35, "δ", color=C[2], fontsize=10, ha="center")
    ax.axhline(-h, color="0.6", lw=0.8, ls=":")
    ax.text(0, -h - 0.7, "↓ 流れ", ha="center", fontsize=8, color="0.5")
    ax.set_aspect("equal")
    ax.set(xlim=(-20, 20), ylim=(-7.2, 1.2), xlabel="半径方向 [mm]",
           ylabel="軸方向 [mm]",
           title="ダクト断面 (外形図の比例読み、縮尺どおり)")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.32),
              ncol=2)
    save(fig, "duct.svg")


# ================================================== D. 過渡応答
def part_d(hover_rpm):
    print("D. 過渡応答")
    step = 0.10 * hover_rpm

    def sched(t):
        return hover_rpm + (step if t >= 0.005 else 0.0)

    dyn = dynamic_run(rotor, model, duration=0.06, rpm=sched,
                      sim_rate=200000.0, measure=False)
    t = dyn.time
    fz = dyn.hub_wrench[:, 2] / G * 1e3
    mz = dyn.hub_wrench[:, 5] * 1e3

    # ブレード通過のリプルを落とす. 中心化移動平均だとステップの前後に
    # 応答が滲んで立ち上がり時刻が測れないので, **因果的**な 1 回転移動平均に
    # して群遅延 (n-1)/2 ぶんを戻す.
    n = max(int(round(200000.0 / (hover_rpm / 60.0 * 4))), 1)
    fz_s = np.convolve(fz, np.ones(n) / n)[: fz.size]
    lag = (n - 1) // 2
    fz_s = np.concatenate([fz_s[lag:], np.full(lag, fz_s[-1])])

    t0 = 0.005
    pre, post = t < t0, t >= t0
    f0 = fz[pre][n:].mean()                 # ステップ前の定常値
    f1 = fz_s[-n:].mean()                   # ステップ後の定常値
    tt, yy = t[post] - t0, fz_s[post]

    # 誘導速度がまだ増えていないので推力は一度行き過ぎ, その後 1 次遅れで
    # 落ち着く (動的インフローの効果). 立ち上がりではなく行き過ぎ量と
    # 整定時間で特徴づける.
    ipk = int(np.argmax(yy))
    peak, t_peak = float(yy[ipk]), float(tt[ipk] * 1e3)
    over = 100.0 * (peak / f1 - 1.0)
    band = np.abs(yy - f1) > 0.02 * abs(f1 - f0)
    t_settle = float(tt[np.max(np.nonzero(band))] * 1e3) if band.any() else 0.0

    # 行き過ぎのあとの減衰から時定数を読む (ln|y - f1| の傾き)
    dec = (tt > tt[ipk]) & (np.abs(yy - f1) > 1e-3 * abs(f1))
    tau_ms = float(
        -1e3 / np.polyfit(tt[dec], np.log(np.abs(yy[dec] - f1)), 1)[0]
    ) if dec.sum() > 20 else float("nan")

    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
    ax[0].plot((t - t0) * 1e3, fz, color="0.78", lw=0.7, label="瞬時値")
    ax[0].plot((t - t0) * 1e3, fz_s, color=C[0], lw=1.8, label="1 回転平均")
    ax[0].axhline(f1, color=C[1], lw=0.9, ls="--", label="整定値")
    ax[0].plot(t_peak, peak, "o", color=C[1], ms=4)
    ax[0].annotate(f"行き過ぎ {over:+.1f} %", (t_peak, peak),
                   textcoords="offset points", xytext=(8, 4), fontsize=8,
                   color=C[1])
    ax[0].set(xlabel="時間 [ms]", ylabel="推力 [gf]",
              title="回転数 +10 % ステップ (推力)", xlim=(-1, 12))
    ax[0].legend(loc="lower right")
    ax[1].plot((t - t0) * 1e3, mz, color=C[2], lw=1.2)
    ax[1].set(xlabel="時間 [ms]", ylabel=r"$M_z$ [mN·m]",
              title="反トルク (慣性トルクを含む)", xlim=(-1, 12))
    save(fig, "step.svg")

    summary["dynamic"] = {
        "step_pct": 10.0,
        "overshoot_pct": float(over),
        "t_peak_ms": t_peak,
        "t_settle_2pct_ms": t_settle,
        "tau_ms": tau_ms,
        "thrust_before_gf": float(f0),
        "thrust_after_gf": float(f1),
        "thrust_peak_gf": peak,
    }
    dyn.to_csv(DATA / "step_response.csv")


# ================================== E. 6 分力の変動 (BPF) と振動
def part_e(hover_rpm):
    print("E. 変動成分")
    dyn = dynamic_run(rotor, model, duration=0.05, rpm=hover_rpm,
                      v_inf=6.0, inflow_angle_deg=75.0,
                      sim_rate=400000.0, measure=False)
    t = dyn.time
    f_rev = hover_rpm / 60.0

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.0))
    lbl = [r"$F_z$", r"$F_x$", r"$M_x$"]
    for k, (col, c) in enumerate(zip((2, 0, 3), C)):
        y = dyn.hub_wrench[:, col]
        y = y - y.mean()
        sc = 1e3 / G if col < 3 else 1e6
        ax[0].plot(t * 1e3, y * sc, color=c, lw=1.0, label=lbl[k])
    ax[0].set(xlabel="時間 [ms]", ylabel="変動分 [gf] / [μN·m]",
              title="6 m/s・流入角 75° での変動", xlim=(0, 10))
    ax[0].legend(ncol=3)

    # 次数スペクトル
    y = dyn.hub_wrench[:, 2]
    n = len(y)
    win = np.hanning(n)
    sp = np.abs(np.fft.rfft((y - y.mean()) * win)) * 2 / win.sum()
    fr = np.fft.rfftfreq(n, t[1] - t[0])
    ax[1].semilogy(fr / f_rev, np.maximum(sp / G * 1e3, 1e-9), color=C[0], lw=1.0)
    for k in (1, 4, 8, 12):
        ax[1].axvline(k, color="0.6", lw=0.7, ls=":")
        ax[1].text(k, ax[1].get_ylim()[1], f"{k}/rev", fontsize=7,
                   ha="center", va="bottom")
    ax[1].set(xlabel="次数 [/rev]", ylabel=r"$F_z$ 振幅 [gf]", xlim=(0, 14),
              title="ハブ荷重の次数スペクトル")
    save(fig, "vibration.svg")

    def order_amp(col, order):
        y = dyn.hub_wrench[:, col]
        i = int(np.argmin(np.abs(fr / f_rev - order)))
        w = np.hanning(len(y))
        s = np.abs(np.fft.rfft((y - y.mean()) * w)) * 2 / w.sum()
        return float(s[i])

    summary["vibration"] = {
        "f_rev_hz": float(f_rev),
        "bpf_hz": float(4 * f_rev),
        "fz_4rev_gf": order_amp(2, 4) / G * 1e3,
        "fz_mean_gf": float(dyn.hub_wrench[:, 2].mean() / G * 1e3),
        "mx_4rev_unm": order_amp(3, 4) * 1e6,
        "imbalance_1rev_mn_per_um": float(
            rotor.geometry.mass * 1e-6 * (hover_rpm * 2 * np.pi / 60.0) ** 2 * 1e3),
    }


# ================================== F. 機体レベル (4 発)
def part_f(hover_rpm):
    print("F. 機体レベル")
    # 定常水平飛行: 機体傾斜 theta で推力の水平成分が抗力とつり合う
    cd_a = 0.0035        # 機体の CdA [m^2] (36 mm 級の粗い見積り)
    rows = []
    for u in (0.0, 2.0, 4.0, 6.0, 8.0):
        drag = 0.5 * 1.225 * cd_a * u**2
        theta = np.degrees(np.arctan2(drag, MASS_G * 1e-3 * G))
        need_gf = MASS_G / N_ROTOR / np.cos(np.radians(theta))
        inflow = 90.0 - theta          # 流入角: 90 = 真横
        rpm = find_rpm(need_gf, v_inf=u, inflow_angle_deg=inflow)
        op = ps.OperatingPoint(rpm=rpm, v_inf=u, inflow_angle_deg=inflow)
        sol = model.solve(rotor, op)
        w = sol.wrench
        rows.append({
            "u": u, "theta_deg": float(theta), "rpm": float(rpm),
            "power_w": float(sol.power(op) * N_ROTOR),
            "fx_gf": float(w.force[0] / G * 1e3),
            "my_unm": float(w.moment[1] * 1e6),
            "my_total_mnm": float(N_ROTOR * w.moment[1] * 1e3),
        })
    summary["vehicle"] = {"cd_a": cd_a, "arm_m": ARM_M, "rows": rows}

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
    u = [r["u"] for r in rows]
    ax[0].plot(u, [r["rpm"] / 1e3 for r in rows], "o-", color=C[0], lw=1.8)
    ax[0].set(xlabel="前進速度 [m/s]", ylabel="回転数 [krpm]",
              title="つり合い回転数 (4 発)")
    ax[1].plot(u, [r["power_w"] for r in rows], "o-", color=C[1], lw=1.8)
    ax[1].set(xlabel="前進速度 [m/s]", ylabel="軸動力 合計 [W]", title="必要動力")
    ax[2].plot(u, [r["my_total_mnm"] for r in rows], "o-", color=C[2], lw=1.8)
    ax[2].set(xlabel="前進速度 [m/s]", ylabel=r"$\sum M_y$ [mN·m]",
              title="ハブモーメントの合計 (機体ピッチ)")
    save(fig, "vehicle.svg")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    hover_rpm = part_a()
    part_b(hover_rpm)
    part_c(hover_rpm)
    part_c2(hover_rpm)
    part_c3()
    part_d(hover_rpm)
    part_e(hover_rpm)
    part_f(hover_rpm)

    summary["meta"] = {
        "mass_g": MASS_G, "n_rotor": N_ROTOR, "arm_m": ARM_M,
        "diameter_mm": float(rotor.geometry.diameter * 1e3),
        "n_blades": int(rotor.geometry.n_blades),
        "model": "BEMT",
        "calibrated": False,
        "layout": LAYOUT,
        "diagonal_mm": DIAGONAL_M * 1e3,
        "arm_mm": ARM_M * 1e3,
        # 隣り合うロータの中心間距離と、直径で割った間隔比。
        # 2 を下回ると干渉が効き始めるとされる。
        "adjacent_spacing_mm": DIAGONAL_M / np.sqrt(2.0) * 1e3,
        "spacing_over_diameter": DIAGONAL_M / np.sqrt(2.0)
                                 / rotor.geometry.diameter,
        "tip_gap_mm": (DIAGONAL_M / np.sqrt(2.0)
                       - rotor.geometry.diameter) * 1e3,
        "spin_by_arm": SPIN_BY_ARM,
    }
    (DATA / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n-> {DATA / 'summary.json'}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
