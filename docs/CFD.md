# OpenFOAM による 2 次元翼型 CFD

`prop_sim.cfd` は OpenFOAM を呼んで **断面まわりの 2 次元流れ**を解き、
`cl(alpha, Re)` / `cd(alpha, Re)` を `TabulatedAirfoil` として返す。

```python
from prop_sim.cfd import run_polar
import prop_sim as ps

out = run_polar(ps.STAMPFLY_1209_SECTION, alphas_deg=[-4, 0, 4, 8],
                reynolds=1.2e4)
rotor = ps.Rotor(ps.stampfly_1209(airfoil=out["airfoil"]))
```

## なぜ 2 次元翼型なのか

本パッケージに残っている最大の不確かさは **Re = 2,000〜20,000 の翼型
ポーラ**であって、3 次元の渦構造ではない。渦法 (Lv3) が BEMT と 0.6 % で
一致していることが、後者はもう十分だと示している
([docs/MODELS.md](MODELS.md) の誤差分解)。

一方で `airfoil_from_section()` の低 Re 補正は経験式で、
`camber_efficiency = 0.85` / `bubble_penalty = 1.6` /
`reynolds_lift_slope = 0.06` はいずれも当てずっぽうに近い。
ここを物理計算に置き換えるのが費用対効果としていちばん大きい。

3 次元ロータ CFD（滑りメッシュ RANS）は数百万セル・数時間〜数日で本
パッケージの想定外。外部で回した 6 分力は
`SurrogateModel.from_conditions()` で取り込める。

## 検証（円柱、既知解との比較）

格子生成・ソルバ設定・荷重積分をまとめて検証するために、まず円柱の
低 Reynolds 数流れを解いた。文献値は Tritton (1959) ほか。

| ケース | Cd | Cd 圧力 | Cd 摩擦 | Cl | 文献値 (Cd) |
|--------|-----|--------|--------|-----|------------|
| Re = 20 | 2.031 | 1.221 | 0.809 | −1e−9 | 2.00〜2.09 |
| Re = 40 | 1.522 | 0.996 | 0.526 | −2e−9 | 1.50〜1.55 |
| Re = 40（格子細分） | 1.514 | 0.991 | 0.523 | −7e−9 | — |
| Re = 40（遠方 80c） | 1.509 | 0.988 | 0.521 | −8e−10 | — |

- 圧力/摩擦の内訳も文献の 1.00 / 0.53 とよく合う
- 格子・領域依存性は 0.6 % 以内
- Re = 40 は渦放出開始 (Re ≈ 47) の手前なので定常。Cl が 1e−9 で
  出ることが、**対称形状で横力が出ない = 積分の非対称誤差が無い**ことの
  確認になっている

## 格子: 自前の O 型格子

`blockMesh` は使っていない。理由:

`blockMesh` で O 型格子を組むには断面の各点を 1 点から放射状に外へ
飛ばすしかなく、これは断面がその点について **星形 (star-shaped)** で
ないと成立しない。StampFly 1209 の断面（キャンバ 7.1 %、厚み 9.5 %）では

- 1/4 コード点 `(0.25, 0)` は断面の**外**に出る（下面が持ち上がるため）
- 図心を使っても後半下面で偏角が逆行する（放射線が表面をかすめる）

ので、どこに中心を置いても格子が裏返る。実際 `blockMesh` は
`is inside-out` / `has inward-pointing faces` で落ちる。

そこで `prop_sim.cfd.mesh` で

```
A_i(k) = p_i + n_i * L_k                (壁法線方向の押し出し)
S_i(k) = p_i + (O_i - p_i) * t_k        (遠方円へ向かう直線)
X_i(k) = (1 - b_k) A_i(k) + b_k S_i(k)
```

という代数格子を作って `constant/polyMesh` を直接書いている。`b_k` は
壁からの**物理距離**の smoothstep で、`normal_length` までは法線押し出し、
それ以降は遠方円へ滑らかに移る。パラメータ `t` ではなく物理距離で切るのは、
押し出し距離が断面の曲率半径を超えると法線どうしが交差してセルが潰れる
から。`O_i` は弧長で割り付けた円周上の点なので最外層はちょうど円になる。

生成後に**全セルの符号つき面積を検査**し、非正なら例外にする
（黙って裏返った格子を渡さない）。

円柱で自前格子と `blockMesh` 版を比べると Cd は 1.522 対 1.517 (0.3 %) で、
移行によって精度は落ちていない。`mesh="blockmesh"` で従来の経路も残してある
（星形の断面でのみ有効）。

## 荷重積分: functionObject を使わない

`forceCoeffs` functionObject を使うのが本来の筋だが、この環境の OpenFOAM
(Debian/Ubuntu の `openfoam` 1912 パッケージ) では **functionObject が
一切使えない**。`functionObjectList::read()` が各 function の辞書の SHA1 を
取るところで

```
error in IOstream "sha1" for operation operator<<(Ostream&, const word&)
```

と落ちる。`forceCoeffs` でも `solverInfo` でも `Q` でも同じなので、
特定の functionObject ではなく `sha1` ストリームの状態バグ。

そこで `prop_sim.cfd.foam_io` で書き出された場から直接積分する:

```
F_p   = sum rho * p_f * Sf                                (圧力)
F_tau = sum rho * nu * A_f * (U_c - U_f)_t / d_perp       (粘性)
```

`Sf` は流体領域から外向き（= 物体に向かう）なので、そのまま
**物体が受ける力**になる。粘性項は境界の `snGrad(U)` の 1 次近似で、
非圧縮・層流では OpenFOAM の `dev2` 版と壁面上で一致する。

副次的な利点:

- OpenFOAM のバージョン依存が無くなる
- 圧力成分と粘性成分を分離して見られる（上の円柱の表）
- テストできる（合成メッシュで幾何量と積分値を検算; `tests/test_cfd.py`）

`use_function_objects=True` にすれば従来の `forceCoeffs` 経路も使える。

## 解法

Re が 10^4 前後なので**層流**（乱流モデル無し）。

- `simpleFoam`（定常）… 剥離が無い低迎角・低 Re ならこれで足りる
- `pimpleFoam`（非定常）… 残差が落ちない場合。Courant 数で刻みを自動調整し、
  `purgeWrite` で残った複数時刻を平均して `Cl_std` / `Cd_std` も返す

**StampFly 1209 の断面は Re = 12,000 では定常解が存在しない。**
`simpleFoam` の残差は 1.5e−3 で頭打ちになり振動する（剥離渦の放出）。
したがってこの断面のポーラを取るには `pimpleFoam` が必要で、
1 迎角あたり 10〜30 分かかる。

## 限界

- Re < 10^4 の**層流剥離泡は 2 次元計算でも難しい**。2 次元では剥離せん断層の
  3 次元不安定が表現できないので、剥離泡が実際より大きく・剥離が早く出る
  傾向がある。結果は「経験式より根拠がある推定」であって真値ではない
- 遷移モデルを入れていない（全域層流）。実機のプロペラは表面粗さと
  流入乱れで遷移が早まる可能性がある
- 2 次元なので翼端・回転効果（遠心力によるスパン方向の境界層流れ、
  失速遅れ）は入らない。これらは 3 次元の効果で BEMT 側の話

## API

| 関数 / クラス | 役割 |
|--------------|------|
| `AirfoilCaseConfig` | Re, 迎角, 格子, ソルバ設定 |
| `OpenFoamCase` | ケースの生成 (`write`)・実行 (`run`)・係数 (`coefficients`) |
| `run_polar` | 迎角を振って `TabulatedAirfoil` を作る |
| `ogrid_nodes` / `write_polymesh` | O 型格子の生成と polyMesh 出力 |
| `PolyMesh` / `read_volume_field` | polyMesh と場の読み込み |
| `patch_wrench` | 壁パッチ上の荷重積分（圧力/粘性の内訳つき） |
| `find_openfoam` | `etc/bashrc` の探索（`PROP_SIM_OPENFOAM_BASHRC` で上書き可） |

`OpenFoamCase.write()` は OpenFOAM が無くても動く（格子生成も含めて
すべて Python 側）。テストもそれを利用して OpenFOAM 非依存にしてある。
