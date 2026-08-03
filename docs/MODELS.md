# プロペラ空力モデルの選択肢

6 分力 (Fx, Fy, Fz, Mx, My, Mz) を再現するために選べるモデルを, 忠実度と
計算コストの順に整理する。**Lv0〜Lv3 と LvS を実装済み**、Lv4 (CFD) は
外部ソルバの結果を取り込む口を用意してある。

実測比較は `examples/08_algorithm_comparison.py` で再現できる。

## 一覧

| Lv | 手法 | 実装 | 1 点あたり | 斜め流入 | 主な用途 |
|----|------|------|-----------|---------|---------|
| 0 | Ct/Cq 多項式 | `QuadraticModel` | ~0.2 ms | 1 次項のみ | 制御シミュレータ, HILS |
| 1 | 翼素理論 + 一様インフロー | `BET` | ~25 ms | ○ | 翼端損失モデルの切り分け |
| **2** | **翼素運動量理論 (BEMT)** | **`BEMT`** | **~22 ms / ~100 ms** | **○** | **本命。設計検討・試験計画** |
| 2.5 | BEMT + 動的インフロー | `BEMT.step` | ~1.2 ms/step | ○ | 動的データ, 過渡応答 |
| 3 | 揚力線 + 渦後流 | `LiftingLine` | ~0.5 s | **×**(軸流のみ) | BEMT の検証, 翼端渦・旋回 |
| 4 | CFD (外部) | `SurrogateModel.from_conditions` | ~0.2 ms | データ次第 | 外部 CFD/実測の取り込み |
| S | 応答曲面 (代理モデル) | `SurrogateModel` | ~0.2 ms | ○ | 大量試行, リアルタイム |

## 実測した比較結果 (10x4.7 2 枚, 6000 rpm ホバー)

| モデル | T [N] | ΔT vs BEMT | Q [mN·m] | FM | コスト |
|--------|-------|-----------|----------|-----|-------|
| Lv0 quadratic | 4.329 | +0.8 % | 62.47 | 0.651 | 216 µs |
| Lv1 BET | 4.542 | +5.8 % | 63.23 | 0.692 | 26 ms |
| Lv2 BEMT | 4.293 | — | 63.81 | 0.630 | 22 ms |
| Lv3 揚力線 (旋回込み) | 4.096 | **−4.6 %** | 62.21 | 0.602 | 0.56 s |
| Lv3 揚力線 (旋回無視) | 4.267 | **−0.6 %** | 64.44 | 0.618 | 1.1 s |
| LvS 応答曲面 | 4.266 | −0.6 % | 62.74 | 0.634 | 260 µs |

**この 3 行が BEMT の誤差分解になっている。**

- 旋回 (swirl) を無視する効果 … 推力 **+4.0 %**, トルク +3.5 %
  → BEMT は後流の旋回を考えないので推力を 4 % 過大評価する
- アニュラス独立 + Prandtl 翼端損失の誤差 … **0.6 %**
  → BEMT のこの 2 つの近似は思ったより良い
- BET (翼端損失なし) は +5.8 % — 翼端損失を無視した分そのまま

半径方向の荷重分布では, 渦法は r/R = 0.45〜0.75 で BEMT より 8〜10 % 低く,
翼端 (r/R > 0.85) では逆に高い。Prandtl 損失は翼端の落ち方を
実際よりきつく見積もっている。

## 各レベルの中身と限界

### Lv0 — Ct/Cq 多項式 (`QuadraticModel`)

```
T = rho n^2 D^4 (ct0 + ct1 J + ct2 J^2)
Q = rho n^2 D^5 (cq0 + cq1 J + cq2 J^2)
```

に、面内前進率 `mu_e = V_edge/(nD)` の 1 次項で法線力とハブモーメントを
足したもの。係数は `QuadraticModel.fit(rotor, BEMT())` で高忠実度モデル
（または実測データ）から最小二乗同定する。

- 長所: 圧倒的に速い。飛行制御シミュレータにそのまま組み込める。
- 短所: 同定した範囲の外は保証されない。方位角依存性 (1/rev, B/rev) を
  一切持たないので、動的な 6 分力の再現には使えない。

### Lv1 — 翼素理論 + 一様インフロー (`BET`)

ディスク全体で 1 つの運動量収支 `T = 2 rho A v sqrt(V_e^2 + (V_a+v)^2)` を
解き、その一様な誘導速度で翼素荷重を積分する。

- 長所: 半径方向のインフロー分布を仮定しないので、BEMT のアニュラス独立
  仮定や翼端損失モデルの影響を切り分けられる。
- 短所: 翼端の荷重を過大評価するため推力が 10〜30 % 高く出る。

### Lv2 — 翼素運動量理論 (`BEMT`) ← 既定

各アニュラスで

```
翼素:   dT/dr = sum_b <dFz/dr>_psi
運動量: dT/dr = 4 pi r rho F v0 sqrt(V_edge^2 + (V_ax + v0)^2)
```

を釣り合わせて `v0(r)` を求める。実装上の要点:

- **Prandtl の翼端・ハブ損失** `F`（`loss_floor` で特異点を回避）
- **Glauert の一般化運動量式**で斜め流入に対応
- **線形インフロー分布**（Drees / Pitt / Coleman）で方位角方向の偏りを表現
  → ディスク前縁側で流入が小さく後縁側で大きくなり、1/rev の翼素荷重変動、
  すなわち **ハブモーメント Mx, My と面内力 Fx, Fy** が生じる
- 翼型は **±180 deg の 360 deg ポーラ**（線形 + Viterna 外挿）。静止推力や
  逆流域でも破綻しない
- Reynolds 補正・Prandtl-Glauert 圧縮性補正・抵抗発散を含む
- 数値解法は**アニュラスごとの二分法（ベクトル化）**。反復が発散しない

**BEMT が苦手な条件**（結果に注意が必要）

| 条件 | 何が起きるか | 対処 |
|------|-------------|------|
| ボルテックスリング状態（降下率 ≈ 誘導速度） | 運動量理論が成立しない | `RotorSolution.warnings` で警告。渦法/実験式が必要 |
| 大きな流入角 (>60 deg) | 後流スキューの線形近似が粗い | Lv3 で検証 |
| 多ロータの干渉 | そもそもモデル外 | Lv3/4 |
| 極端な低 Re (<3e4) | 翼型ポーラの信頼性が落ちる | `LOW_RE_THIN` を使う。実測ポーラがあれば `TabulatedAirfoil` |
| 深い失速 | Viterna 外挿は定常値のみ、動的失速なし | Lv3/4、または非定常翼型モデルを追加 |

### Lv2.5 — 動的インフロー

誘導速度を状態変数にして 1 次遅れで追従させる。

```
dv0/dt = (v0_qs - v0) / tau,   tau = 0.85 / (4 nu Omega),  nu = v_i/(Omega R)
```

`v0_qs` はその瞬間の翼素荷重に対応する運動量理論の解。時定数は
Pitt–Peters / Carpenter–Fridovich の見かけ質量に基づく。定常状態は BEMT の
解に厳密に一致する（テストで検証済み）。

これにより、スロットルステップで**推力が定常値を数 % 行き過ぎてから
収束する**といった、実測でよく見える挙動が再現される。

### Lv3 — 渦法（未実装 / 拡張ポイント）

- **揚力線 + 規定後流**: ブレードを束縛渦、後流を螺旋渦として Biot–Savart で
  誘導速度を計算。BEMT のアニュラス独立仮定と Prandtl 損失を置き換えられる。
- **自由渦後流 (FVW)**: 後流形状も時間発展させる。斜め流入・機体干渉・
  ロータ間干渉に強い。
- 実装するなら `AeroModel` を継承して `solve()` で `InflowField` を返すだけで
  よい（`Rotor` 側は変更不要）。方位角ごとの誘導速度が必要なら
  `InflowField` を継承して `induced(r_R, psi)` を任意関数にする。

### Lv4 — CFD（未実装 / 外部連携）

RANS（滑りメッシュ）や アクチュエータライン LES。実務では「BEMT で広範囲を
掃引 → 数点だけ CFD で検証 → 差分を補正係数にする」使い方が現実的。
`AeroModel` のラッパとして外部ソルバの結果テーブルを読み込む形にすれば、
本シミュレータの試験シナリオ・センサモデルをそのまま流用できる。

## 微小プロペラ (Re ~ 10^4) の注意

30 mm 級のマイクロドローン用プロペラは、翼端でも Reynolds 数が 5×10^3 〜 2×10^4
にしかならない。この領域では層流剥離のせいで

- 揚力傾斜が 2π の 6 割程度まで落ちる
- 最大揚力係数が 0.8 前後に下がる
- 抗力係数が通常の翼型の 3〜5 倍になる

ため、通常の翼型ポーラ（Re ~ 5×10^5 相当）をそのまま使うと**推力を 20〜25 %、
ホバー効率 FM を 0.1 以上過大評価する**。汎用の `LOW_RE_THIN` はこの領域に
合わせた薄翼モデルで、`reynolds_ref = 1e4`、`reynolds_exponent = 0.4` として
Re 依存性も強めてある。

### 翼断面が実測できる場合

ブレードを切って断面形状（キャンバ線 + 厚み分布）が取れるなら、
`prop_sim.section` で翼型モデルを作れる。

```python
from prop_sim import SectionShape, airfoil_from_section

sec = SectionShape(x=..., camber=..., thickness=...)   # x/c で正規化
print(sec.thin_airfoil_properties().summary())          # alpha0, cl_i, cm_ac
foil = airfoil_from_section(sec, reynolds_ref=1.2e4)
```

処理は 2 段:

1. **薄翼理論**（非粘性）でキャンバ線から
   `alpha0 = -(1/pi)∫(dz/dx)(cos θ - 1)dθ`、理想迎角、`cm_ac` を求める
2. **低 Re の粘性補正**で `LinearAirfoil` のパラメータに落とす
   - 揚力傾斜 = 2π × η(Re)（Re = 10⁴ で η ≈ 0.63）
   - `alpha0` × キャンバ効率（既定 0.85。境界層によるデキャンバ）
   - `cd_min` = 層流平板摩擦 × 厚み形状係数 × 剥離泡ペナルティ（既定 1.6）

補正係数はすべて引数で上書きできる。StampFly 1209 の実例では、断面を実測した
ことでホバー回転数の推定幅が **±8 % → ±2 %** に縮んだ
（[docs/STAMPFLY_1209.md](STAMPFLY_1209.md)）。

`examples/07_stampfly_1209.py` が翼型仮定の感度を定量的に出力する。

## 独自モデルの追加方法

```python
from prop_sim.models import AeroModel, RotorSolution, register_model
from prop_sim.inflow import InflowField


class MyVortexModel(AeroModel):
    name = "my-vortex"

    def solve(self, rotor, op):
        r_R = rotor.radial_grid(32)
        v0 = ...                       # 何らかの方法で誘導速度を求める
        inflow = InflowField(r_R, v0)
        wrench = rotor.mean_aero_wrench(inflow, op, r_R, n_azimuth=36)
        return RotorSolution(wrench, inflow)


register_model("my-vortex", MyVortexModel)
```

`static_sweep` / `dynamic_run` / センサモデル / 後処理はモデルに依存しない
ので、これだけで既存の試験シナリオがすべて新モデルで走る。

## モデル選択の指針

- **試験計画を立てたい / どの条件を測るか決めたい** → `BEMT`
- **計測系の誤差を見積もりたい** → `BEMT` + `TestRig`（例 6）
- **制御則を回したい** → `SurrogateModel.fit(rotor, BEMT())`
- **BEMT の結果を疑いたい** → 軸流なら `LiftingLine` と突き合わせる（例 8）。
  旋回の分だけ BEMT が推力を 4 % 高く出すことを織り込んで判断する
- **CFD や風洞データがある** → `SurrogateModel.from_conditions` で取り込めば
  試験シナリオ・センサモデル・後処理をそのまま使える
- **実測データで較正したい** → 翼型を実測断面から作り（`airfoil_from_section`）、
  静止推力 1 点で `collective_deg` を合わせるのが最短
