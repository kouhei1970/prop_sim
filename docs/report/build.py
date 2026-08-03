"""`data/summary.json` と `template.html` から報告ページを組み立てる.

    python examples/11_virtual_wind_tunnel.py   # 先にデータと図を作る
    python docs/report/build.py                 # index.html を生成

テンプレート中の記法

    {{a.b.c|.2f}}   summary.json の値を書式付きで埋める
    {{fig:name.svg}} assets/name.svg を <figure> の中に埋め込む
    {{rows:vehicle}} 表の行を組み立てる

``--fragment PATH`` を付けると <body> の中身だけ (図は SVG をインライン展開)
を書き出す。単一ファイルでプレビューしたいとき用。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ASSETS = HERE / "assets"

PLACEHOLDER = re.compile(r"\{\{([^}|]+)(?:\|([^}]*))?\}\}")

TITLE = "StampFly 1209 プロペラ 仮想風洞試験レポート"
DESCRIPTION = (
    "31 mm 4 枚マイクロプロペラの 6 分力を静止・前進飛行・機体運動について"
    "数値掃引し、飛行制御系の設計に使える係数・微係数・周波数にまとめた報告。"
)


def load_summary() -> dict:
    path = DATA / "summary.json"
    if not path.is_file():
        raise SystemExit(
            f"{path} がありません。先に examples/11_virtual_wind_tunnel.py を実行してください")
    s = json.loads(path.read_text())
    return derive(s)


def derive(s: dict) -> dict:
    """テンプレートから参照しやすい派生量を足す."""
    st, dv, fw, vb = s["static"], s["derivatives"], s["forward"], s["vibration"]

    def eng(x: float, digits: int = 3) -> str:
        """1.23e-08 のような指数表記を読みやすくする."""
        return f"{x:.{digits}e}".replace("e-0", "e-").replace("e+0", "e+")

    st["k_t_e"] = eng(st["k_t"])
    st["k_q_e"] = eng(st["k_q"])
    st["k_q_over_k_t_mm"] = st["k_q_over_k_t"] * 1e3

    dv["dT_domega_e"] = eng(dv["dT_domega"])
    dv["dT_domega_norm_e"] = eng(dv["dT_domega_norm"])
    dv["dQ_domega_e"] = eng(dv["dQ_domega"])
    dv["dMy_du_e"] = eng(dv["dMy_du"])
    dv["dMy_dq_e"] = eng(dv["dMy_dq"])
    dv["spin_momentum_e"] = eng(dv["spin_momentum"])
    dv["minus_spin_momentum_e"] = eng(-dv["spin_momentum"])
    dv["dMx_dq_e"] = eng(dv["dMx_dq"])
    dv["dMx_dq_aero_e"] = eng(dv["dMx_dq_aero"])
    dv["gyro_dominance"] = abs(dv["dMx_dq"]) / abs(dv["dMy_dq"])

    # --- StampFly のファーム系 (FRD) への換算 -------------------------------
    # ハブ系 (FLU) とは x 軸まわり 180 度の関係なので y, z 成分と
    # 角速度 q, r が反転する。微係数は反転が 2 回起きて相殺する場合がある。
    dv["dMy_du_frd"] = -dv["dMy_du"]              # My だけ反転
    dv["dMy_du_frd_e"] = eng(dv["dMy_du_frd"])
    dv["dMy_du_4rotor_frd_unm"] = -dv["dMy_du_4rotor_unm"]
    dv["dMx_dq_frd"] = -dv["dMx_dq"]              # q だけ反転
    dv["dMx_dq_frd_e"] = eng(dv["dMx_dq_frd"])
    dv["dMy_dq_frd_e"] = eng(dv["dMy_dq"])        # 2 回反転して不変
    dv["dFx_du_frd"] = dv["dFx_du"]               # 不変
    dv["dT_dw_frd"] = dv["dT_dw"]                 # 2 回反転して不変
    dv["spin_momentum_frd_e"] = eng(dv["spin_momentum"])
    st["hover_torque_frd_mnm"] = -st["hover_torque_mnm"]
    dv["heave_tau_s"] = 1.0 / abs(dv["heave_damping_4rotor"])
    dv["gyro_at_10"] = abs(dv["spin_momentum"]) * 10.0 * 1e6
    dv["gyro_at_10_pct"] = 100.0 * dv["gyro_at_10"] / (st["hover_torque_mnm"] * 1e3)

    for k, d in fw.items():
        # 軸流では対称性から面内力・ハブモーメントは厳密にゼロ。
        # 1e-14 のような数値誤差を "-0" と表示しないよう丸める。
        for key in ("fx_gf", "mx_unm", "my_unm"):
            if abs(d[key]) < 1e-6:
                d[key] = 0.0
        d["fz_loss_pct"] = 100.0 * (1.0 - d["fz_ratio"])
        d["fz_gain_pct"] = 100.0 * (d["fz_ratio"] - 1.0)
        d["my_over_torque_pct"] = 100.0 * abs(d["my_unm"]) / (
            st["hover_torque_mnm"] * 1e3)
        d["mx_over_my"] = abs(d["mx_unm"]) / max(abs(d["my_unm"]), 1e-12)
        d["my_4rotor_unm"] = 4.0 * abs(d["my_unm"])
        d["mx_frd_unm"] = d["mx_unm"] or 0.0      # ロールは不変
        d["my_frd_unm"] = -d["my_unm"] or 0.0     # ピッチは反転 (-0.0 を潰す)
        d["my_4rotor_frd_unm"] = -4.0 * d["my_unm"] or 0.0

    vb["fz_4rev_pct"] = 100.0 * vb["fz_4rev_gf"] / vb["fz_mean_gf"]
    f, fs = vb["bpf_hz"], 1000.0
    vb["alias_1khz"] = abs(f - round(f / fs) * fs)

    s["meta"]["arm_mm"] = s["vehicle"]["arm_m"] * 1e3
    s["meta"]["generated"] = _dt.date.today().isoformat()
    return s


def lookup(s: dict, path: str):
    cur = s
    for part in path.split("."):
        cur = cur[part]
    return cur


def _z(x: float, tol: float = 1e-9) -> float:
    """符号つきゼロ (-0.000 表示) を潰す."""
    return 0.0 if abs(x) < tol else x


def vehicle_rows(s: dict) -> str:
    out = []
    for r in s["vehicle"]["rows"]:
        out.append(
            "<tr>"
            f'<td class="num">{r["u"]:.0f}</td>'
            f'<td class="num">{r["theta_deg"]:.1f}</td>'
            f'<td class="num">{r["rpm"]:,.0f}</td>'
            f'<td class="num">{r["power_w"]:.2f}</td>'
            # FRD では My の符号が反転する (正 = 機首上げ)
            f'<td class="num">{_z(-r["my_total_mnm"]):.3f}</td>'
            "</tr>"
        )
    return "\n".join(out)


def render(template: str, s: dict, *, inline_svg: bool) -> str:
    def sub(m: re.Match) -> str:
        key, fmt = m.group(1).strip(), (m.group(2) or "s").strip()
        if key.startswith("fig:"):
            name = key[4:]
            if inline_svg:
                svg = (ASSETS / name).read_text()
                svg = svg[svg.index("<svg"):]           # XML 宣言と DOCTYPE を落とす
                # ページに直接展開するときは SVG 自前の配色を外し,
                # 周囲の color (テーマ切替に追従する) を継承させる
                svg = re.sub(r"<style>svg\{color:.*?</style>", "", svg, count=1)
                return f'<div class="fig-box">{svg}</div>'
            return (f'<div class="fig-box"><img src="assets/{name}" alt="" '
                    f'loading="lazy"></div>')
        if key.startswith("rows:"):
            return {"vehicle": vehicle_rows}[key[5:]](s)
        v = lookup(s, key)
        return format(v, fmt) if fmt != "s" else str(v)

    html = PLACEHOLDER.sub(sub, template)
    # 幅の広い表は横スクロールさせる (本文が横に溢れないように)
    return re.sub(r"(<table class=\"data[^\"]*\">.*?</table>)",
                  r'<div class="table-scroll">\1</div>', html, flags=re.S)


HEAD = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:type" content="article">
<link rel="stylesheet" href="style.css">
</head>
<body>
<main class="wrap">
"""

FOOT = """</main>
</body>
</html>
"""

LANDING = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>prop_sim — 無人機プロペラ 6 分力試験シミュレータ</title>
<link rel="stylesheet" href="report/style.css">
</head>
<body>
<main class="wrap">
<header class="masthead">
  <p class="eyebrow">prop_sim</p>
  <h1>無人機プロペラ<br>6 分力試験シミュレータ</h1>
  <p class="deck">
    回転するプロペラの 6 分力 (F<sub>x</sub>, F<sub>y</sub>, F<sub>z</sub>,
    M<sub>x</sub>, M<sub>y</sub>, M<sub>z</sub>) を計測する実験を、
    空力から 6 分力計の出力までまるごとシミュレートする Python パッケージ。
  </p>
</header>
<section class="panel">
  <h2>レポート</h2>
  <table class="data wide">
    <tbody>
      <tr><th><a href="report/">{title}</a></th>
          <td>静止・前進飛行・機体運動での 6 分力を掃引し、飛行制御系の設計に
              使える係数・微係数・周波数にまとめたもの</td></tr>
    </tbody>
  </table>
  <h2>ドキュメント</h2>
  <table class="data wide">
    <tbody>
      <tr><th><a href="https://github.com/kouhei1970/prop_sim/blob/main/docs/STAMPFLY_1209.md">STAMPFLY_1209.md</a></th>
          <td>StampFly 1209 の形状実測・性能推定・計測要求仕様</td></tr>
      <tr><th><a href="https://github.com/kouhei1970/prop_sim/blob/main/docs/MODELS.md">MODELS.md</a></th>
          <td>空力モデル Lv0–Lv4 の比較と限界</td></tr>
      <tr><th><a href="https://github.com/kouhei1970/prop_sim/blob/main/docs/CFD.md">CFD.md</a></th>
          <td>OpenFOAM による 2 次元翼型 CFD と検証</td></tr>
    </tbody>
  </table>
</section>
<footer class="colophon">
  <p><a href="https://github.com/kouhei1970/prop_sim">github.com/kouhei1970/prop_sim</a> — MIT License</p>
</footer>
</main>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", help="body の中身だけを書き出す先 (SVG をインライン化)")
    args = ap.parse_args()

    s = load_summary()
    template = (HERE / "template.html").read_text()

    html = render(template, s, inline_svg=False)
    out = HERE / "index.html"
    out.write_text(HEAD.format(title=TITLE, description=DESCRIPTION) + html + FOOT)
    print(f"-> {out}")

    # GitHub Pages を docs/ から配信する前提。Jekyll を通さない。
    (HERE.parent / ".nojekyll").write_text("")
    landing = HERE.parent / "index.html"
    landing.write_text(LANDING.format(title=TITLE))
    print(f"-> {landing}")

    if args.fragment:
        frag = Path(args.fragment)
        css = (HERE / "style.css").read_text()
        body = render(template, s, inline_svg=True)
        frag.write_text(
            f"<title>{TITLE}</title>\n<style>\n{css}\n</style>\n"
            f'<main class="wrap">\n{body}\n</main>\n')
        print(f"-> {frag}")


if __name__ == "__main__":
    main()
