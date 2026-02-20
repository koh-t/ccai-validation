# ccai-validation 仕様書

**Empirical Validation of Sharded Citizens' Assembly on CCAI Polis Data**

| 項目 | 内容 |
|------|------|
| バージョン | 0.1.0 |
| 対応論文 | "Convexity Kills Arrow but Not Coalitions" §6 |
| 最終更新 | 2026-02-20 |

---

## 1. 概要

本パッケージは論文 §6 (Empirical Validation) の全実験を再現するコードである。
Anthropic の Collective Constitutional AI (CCAI) プロジェクトで収集された
Polis 投票データ（1,002 名 × 275 声明）を用い、以下の理論的結果を実データ上で検証する。

| 実験 | 検証対象 | 論文定理 |
|------|---------|---------|
| PAC 非空性 | 部分合意錐 $C_{ij}(\pi^*)$ が非空 | Lemma 4.3 |
| ペアワイズ・ログローリング | 2 エージェント連合の操作成功率 | Theorem 4.8 (Example 4.10) |
| 集約ルール比較 | 各ルール下での連合操作のシフト量 | Theorem 4.8 |
| メカニズム特性 | 軸ごとの SP・パネル独立性 | Theorem 5.2 |
| シャーディング効果 | 少数派連合の軸キャプチャ確率 | Proposition 5.3 |
| 憲法レベル操作 | 戦略的投票による原則入替え | §6.7 |

---

## 2. 環境構築

### 2.1 前提条件

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) ≥ 0.5

### 2.2 セットアップ

```bash
# リポジトリのクローン
git clone <repo-url>
cd ccai-validation

# uv による依存解決・仮想環境構築（lockfile 利用）
uv sync

# データの配置
# https://github.com/saffronh/ccai からダウンロード
mkdir -p data
cp /path/to/clean_votes.csv data/
cp /path/to/clean_comments.csv data/   # optional
```

### 2.3 依存パッケージ

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| numpy | ≥1.24 | 数値計算全般 |
| pandas | ≥2.0 | データ読込 |
| scipy | ≥1.11 | softmax, trim_mean |
| scikit-learn | ≥1.3 | PCA, SimpleImputer |
| pytest | ≥8.0 | テスト (dev) |
| pytest-cov | ≥5.0 | カバレッジ (dev) |

---

## 3. プロジェクト構成

```
ccai-validation/
├── pyproject.toml              # パッケージ定義 + uv/pytest 設定
├── uv.lock                     # ロックファイル（uv sync で生成）
├── README.md
├── SPEC.md                     # 本仕様書
├── data/                       # CCAI データ配置先
│   ├── clean_votes.csv
│   └── clean_comments.csv
├── src/
│   └── ccai_validation/
│       ├── __init__.py
│       ├── core.py             # 数学プリミティブ
│       ├── mechanism.py        # Citizens' Assembly メカニズム
│       ├── data.py             # データ読込・前処理
│       ├── experiments.py      # 6 つの実験関数
│       └── cli.py              # CLI エントリポイント
└── tests/
    ├── conftest.py             # 共通フィクスチャ
    ├── test_core.py            # core.py のテスト (36 件)
    ├── test_mechanism.py       # mechanism.py のテスト (20 件)
    ├── test_experiments.py     # experiments.py のテスト (23 件)
    └── test_data.py            # data.py のテスト (5 件)
```

---

## 4. モジュール仕様

### 4.1 `core.py` — 数学プリミティブ

論文の数式に直接対応する基本関数群。

#### 定数

| 名前 | 値 | 説明 |
|------|-----|------|
| `EPS` | `1e-15` | ゼロ除算防止の微小定数 |

#### シンプレックス操作

| 関数 | シグネチャ | 対応 |
|------|-----------|------|
| `to_simplex(x)` | `Vec → Vec` | $x_j \leftarrow \max(x_j, \varepsilon) / \sum$ |
| `is_simplex(x, atol)` | `Vec → bool` | $x \in \Delta^{m-1}$ の判定 |

#### 埋め込み

| 関数 | シグネチャ | 対応 |
|------|-----------|------|
| `embed_simplex(V, m)` | `(n,d) Mat × int → (n,m) Mat, PCA` | PCA + softmax → $\Delta^{m-1}$ |
| `impute_votes(V, fill)` | `Mat → Mat` | NaN → `fill` (default 0) |

**embed_simplex の処理フロー:**

1. `PCA(n_components=m).fit_transform(V)` → `(n, m)` スコア行列
2. 行ごとの softmax → 各行が $\Delta^{m-1}$ 上の点

#### ℓ₂ 効用・勾配（論文 Eq. 1–2）

| 関数 | 数式 | 制約 |
|------|------|------|
| `l2_sq(a, b)` | $\|a - b\|^2$ | — |
| `l2_utility(π, p_i)` | $u_i(\pi) = -\|\pi - p_i\|^2$ | $u_i(p_i) = 0$（最大値） |
| `l2_gradient(π, p_i)` | $\nabla_\pi u_i = 2(p_i - \pi)$ | $\sum_j g_j = 0$（$T\Delta$ 上） |

#### 集約ルール（§6.4）

全ルールの出力は $\Delta^{m-1}$ 上（`to_simplex` で正規化）。

| 関数 | アルゴリズム |
|------|------------|
| `cw_median(ideals)` | 座標ごとメディアン → 正規化 |
| `cw_mean(ideals)` | 座標ごと平均 → 正規化 |
| `cw_trimmed_mean(ideals, trim)` | 座標ごとトリム平均（上下 `trim` %除去） → 正規化 |

#### 部分合意錐（Definition 4.1）

| 関数 | 返り値 | 判定条件 |
|------|-------|---------|
| `pac_nonempty(g_i, g_j)` | `bool` | $\cos\theta > -1 + 10^{-10}$（非反平行） |
| `gradient_cos(g_i, g_j)` | `float` | $\cos\theta = \langle g_i, g_j \rangle / (\|g_i\| \|g_j\|)$ |

#### 連合操作ヘルパー

| 関数 | 説明 |
|------|------|
| `coalition_target(ideals, idx, π*, s)` | 連合 $S$ の目標点 $\pi^* + s \cdot d^*$（$d^*$ は平均勾配方向） |
| `utility_gain(π_new, π_old, ideals, idx)` | 平均 $\Delta u$ および利得割合 |
| `jaccard(A, B)` | $\|A \cap B\| / \|A \cup B\|$ |

---

### 4.2 `mechanism.py` — Sharded Citizens' Assembly

論文 Definition 5.1 の完全実装。

#### `CAResult` データクラス

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `pi` | `Vec` | 集約結果 $\pi^* \in \Delta^{m-1}$ |
| `A_idx` | `ndarray[intp]` | Panel A のグローバルインデックス |
| `B_idx` | `ndarray[intp]` | Panel B のグローバルインデックス |
| `axes` | `(K, m) Mat` | PCA 抽出軸（正規直交） |
| `mu_A` | `Vec` | Panel A 重心 |
| `projections` | `(|B|, K) Mat` | Panel B の軸射影 |
| `medians` | `(K,) Vec` | 軸ごとのメディアン |
| `subpanels` | `list[ndarray] \| None` | シャード時の各サブパネルの局所インデックス |

#### `citizens_assembly()` 関数

```python
def citizens_assembly(
    ideals: Mat,
    *,
    K: int | None = None,        # 軸数（default: min(m-1, 5)）
    panel_a_frac: float = 0.5,   # Panel A 割合
    seed: int = 42,              # 乱数シード
    shard: bool = True,          # シャーディングの有無
) -> CAResult
```

**処理ステップ:**

1. **Partition**: `rng.permutation(n)` で無作為に Panel A / B に分割
2. **Axis Extraction**: Panel A の理想点を PCA → 上位 $K$ 主成分軸
3. **Voting**:
   - `shard=True`: Panel B を $K$ サブパネルに分割、各サブパネルが 1 軸のみ投票（メディアン）
   - `shard=False`: Panel B 全員が全軸に投票
4. **Reconstruction**: $\hat\pi = \bar x_A + \sum_k \mu_k e_k$ → シンプレックスに射影

**不変条件:**
- 同一 seed → 同一出力（決定性）
- `A_idx ∩ B_idx = ∅`、`|A_idx| + |B_idx| = n`
- `axes @ axes.T ≈ I_K`（正規直交）
- `shard=True` のとき、サブパネルは互いに素で Panel B を被覆

---

### 4.3 `data.py` — データ読込

#### `CCAIData` データクラス

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `V_raw` | `(n, d) Mat` | 生投票行列（NaN あり） |
| `V_filled` | `(n, d) Mat` | 補完済み投票行列 |
| `groups` | `ndarray[intp]` | Polis グループ ID |
| `vote_cols` | `list[str]` | 声明 ID（列名） |
| `comments_df` | `DataFrame \| None` | 声明テキスト（任意） |

計算プロパティ: `n`, `d`, `g0_idx`, `g1_idx`, `sparsity`

#### `load_ccai(data_dir)` 関数

- `clean_votes.csv` が存在しなければ `FileNotFoundError`
- メタ列（participant, xid, group-id, ...）を除外
- NaN → 0 で補完

**期待するファイル形式:**

`clean_votes.csv`:
- 各行 = 1 参加者
- `group-id` 列: 0 または 1
- 声明列: -1（反対）, 0（パス）, 1（賛成）, NaN（未投票）

---

### 4.4 `experiments.py` — 6 つの実験

各実験関数は純粋関数（副作用なし）。入力は前処理済みデータ、出力はデータクラス。

#### Experiment 1: `exp_pac()`

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `ideals` | — | `(n, m)` 理想点 |
| `groups` | — | グループ ID |
| `n_samples` | 2000 | サンプリングするペア数 |
| `seed` | 42 | 乱数シード |

**返り値** `PACResult`:
- `rates`: `dict[str, float]` — `{all, intra_g0, intra_g1, cross}` ごとの PAC 非空率
- `antiparallel_frac`: $\cos\theta < -0.9$ の割合
- `mean_cos`: 平均 $\cos\theta$

**検証論理**: ランダムにエージェント対 $(i, j)$ をサンプルし、$\pi^*$（座標メディアン）での
勾配 $g_i, g_j$ の角度から PAC 非空性を判定。

#### Experiment 2: `exp_aggregation()`

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `ideals` | — | `(n, m)` 理想点 |
| `groups` | — | グループ ID |
| `strength` | 0.01 | 操作強度 $s$ |
| `n_ca_trials` | 50 | CA のランダム試行数 |

**返り値** `list[AggRow]` (5 行: Mean, Trimmed mean, Median, CA-unsharded, CA-sharded)

**操作プロトコル**: G0 全員が目標 $\pi^* + s \cdot d^*$ を報告。

#### Experiment 3: `exp_mechanism_properties()`

**返り値** `MechResult`:
- `sp_violation`: 軸ごと SP 違反率（Theorem 5.2 により理論的に 0）
- `panel_independence`: $\|\text{axes}_{honest} - \text{axes}_{manip}\|$（理論的に 0）

**テスト方法**:
- SP: 各サブパネルの各エージェントについて、最適な嘘（メディアンの反対側に極端な値）を試行。
  新メディアンが真の射影に近づけば違反。
- 独立性: Panel B を操作しても Panel A から抽出される軸は不変。

#### Experiment 4: `exp_sharding()`

**返り値** `ShardResult`:
- `shifts_sharded / unsharded`: $\|\pi_{manip} - \pi_{honest}\|^2$ のリスト
- `gains_sharded / unsharded`: G0 利得割合のリスト
- `captures`: シャード時に連合が過半数を占めた軸の数
- `rho`: 連合比率

`use_minority=True` のとき、G0 ではなくランダムな $\rho n$ 人の連合を生成。
Proposition 5.3 の適用条件（$\rho < 1/2$）を満たす設定で検証可能。

#### Experiment 5: `exp_pairwise()`

**返り値** `PairwiseResult`:
- `success`: ログローリング成功数
- `total`: 試行数
- `gains`: 成功した各ペアの `(Δu_i, Δu_j)` リスト

**方法**: ランダムペア $(i,j)$ について、$d = g_i + g_j$ が PAC 内にあれば
$\pi' = \pi^* + 0.001 \cdot d / \|d\|$ を計算し、両者の効用増加を確認。

#### Experiment 6: `exp_constitution()`

**返り値** `ConstitutionResult`:
- `n_contentious`: $|\text{gap}| > \text{threshold}$ の声明数
- `J_honest_g0`, `J_honest_g1`, `J_g0_g1`, `J_honest_manip`: Jaccard 指数
- `n_entered`, `n_exited`: 操作後に上位 15 に入った/出た原則の数

**操作プロトコル**: G0 が contentious 声明で戦略的投票
（G0 有利 → 全員 +1、G1 有利 → 全員 -1）。

---

### 4.5 `cli.py` — CLI

```bash
# 全実験実行
uv run ccai-run --data_dir ./data

# 特定実験のみ
uv run ccai-run --data_dir ./data --experiments pac,pairwise

# 次元指定
uv run ccai-run --data_dir ./data --dims 3,5,10
```

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--data_dir` | `./data` | CCAI データディレクトリ |
| `--experiments` | 全 6 実験 | カンマ区切りで選択 |
| `--dims` | `3,5,10` | シンプレックス次元 |

---

## 5. テスト仕様

### 5.1 実行方法

```bash
# 全テスト（実データ不要のもの）
uv run pytest -m "not slow"

# 実データを使うテストも含む
uv run pytest

# カバレッジ
uv run pytest --cov=ccai_validation --cov-report=term-missing
```

### 5.2 テスト一覧

#### `test_core.py` (35 件)

| クラス | テスト | 検証内容 |
|--------|-------|---------|
| `TestSimplex` | `test_to_simplex_sums_to_one` | $\sum \pi_j = 1$ |
| | `test_to_simplex_positive` | 負入力でも $\pi_j > 0$ |
| | `test_is_simplex_valid` | 正常入力の判定 |
| | `test_is_simplex_invalid_sum` | 和が 1 でない場合の棄却 |
| | `test_is_simplex_negative` | 負成分の棄却 |
| `TestEmbedding` | `test_output_on_simplex` | 全 100 エージェントが $\Delta$ 上 |
| | `test_pca_components` | PCA 次元数の一致 |
| | `test_impute_votes_no_nan` | NaN 完全除去 |
| | `test_impute_preserves_shape` | 形状保存 |
| `TestL2` | `test_l2_sq_zero` | $\|a-a\|^2 = 0$ |
| | `test_l2_sq_symmetric` | $\|a-b\|^2 = \|b-a\|^2$ |
| | `test_utility_maximised_at_ideal` | $u_i(p_i) = 0$ |
| | `test_utility_negative_elsewhere` | $u_i(\pi) < 0$ for $\pi \ne p_i$ |
| | `test_gradient_in_tangent_space` | $\sum_j g_j = 0$ |
| | `test_gradient_direction` | 勾配が $p_i - \pi$ 方向 |
| | `test_gradient_zero_at_ideal` | $\nabla u_i(p_i) = 0$ |
| `TestAggregation` | `test_median_on_simplex` | 出力が $\Delta$ 上 |
| | `test_mean_on_simplex` | 出力が $\Delta$ 上 |
| | `test_trimmed_mean_on_simplex` | 出力が $\Delta$ 上 |
| | `test_unanimous_median` | 全員一致 → median = ideal |
| | `test_unanimous_mean` | 全員一致 → mean = ideal |
| | `test_mean_equals_centroid` | Example 4.10 で mean ≈ 重心 |
| `TestPAC` | `test_parallel_nonempty` | 平行 → 非空 |
| | `test_antiparallel_empty` | 反平行 → 空 |
| | `test_orthogonal_nonempty` | 直交 → 非空 |
| | `test_zero_gradient_empty` | ゼロ勾配 → 空 |
| | `test_1d_straddling_empty` | 1D 反対側 → 空 |
| | `test_example_4_10` | $\cos\theta = -1/(m-1) = -0.5$ の検証 |
| `TestCoalition` | `test_target_on_simplex` | 目標点が $\Delta$ 上 |
| | `test_target_differs_from_median` | 目標 ≠ メディアン |
| | `test_utility_gain_unanimous` | 全員に有利な移動 |
| `TestJaccard` | `test_identical` | $J(A,A) = 1$ |
| | `test_disjoint` | $J(A,B) = 0$ for disjoint |
| | `test_partial` | $J(\{1,2,3\}, \{2,3,4\}) = 0.5$ |
| | `test_empty` | $J(\emptyset, \emptyset) = 0$ |

#### `test_mechanism.py` (16 件)

| クラス | テスト | 検証内容 |
|--------|-------|---------|
| `TestCABasics` | `test_output_on_simplex` | $\pi^* \in \Delta$ |
| | `test_partition_disjoint` | $A \cap B = \emptyset$ |
| | `test_partition_complete` | $\|A\| + \|B\| = n$ |
| | `test_axes_orthonormal` | $E E^T = I_K$ |
| | `test_projections_shape` | $(|B|, K)$ |
| | `test_medians_length` | $|\mu| = K$ |
| `TestSharding` | `test_subpanels_exist_when_sharded` | shard=True → subpanels ≠ None |
| | `test_subpanels_none_when_unsharded` | shard=False → subpanels = None |
| | `test_subpanels_disjoint` | サブパネル互いに素 |
| | `test_subpanels_cover_panel_b` | 和集合 = Panel B |
| `TestDeterminism` | `test_same_seed_same_output` | 決定性 |
| | `test_different_seed_different_partition` | 異なる seed → 異なる分割 |
| `TestPanelIndependence` | `test_axes_unchanged_under_b_manipulation` | Panel B 操作 → 軸不変 |
| `TestPerAxisSP` | `test_single_agent_cannot_benefit` | 1D メディアン SP（200 ケース） |
| `TestUnanimity` | `test_unanimous_recovery` | 全員一致 → 理想点復元 |
| | `test_k_defaults` | K=None → min(m-1, 5) |

#### `test_experiments.py` (20 件)

| クラス | テスト | 検証内容 |
|--------|-------|---------|
| `TestExpPAC` | 4 件 | 返り値の型・範囲、高次元での非空率 > 90% |
| `TestExpAggregation` | 4 件 | 5 ルール返却、名前一致、shift ≥ 0、frac ∈ [0,1] |
| `TestExpMechanism` | 2 件 | SP 違反 = 0、パネル独立性 = 0 |
| `TestExpSharding` | 3 件 | 多数派キャプチャ > 0、少数派キャプチャ < 3、リスト長一致 |
| `TestExpPairwise` | 3 件 | triangle で成功 > 0、高次元で率 > 80%、利得正値 |
| `TestExpConstitution` | 4 件 | Jaccard ∈ [0,1]、exited ∈ [0,15]、contentious ≥ 0、閾値単調性 |

#### `test_data.py` (5 件)

| テスト | 検証内容 |
|--------|---------|
| `test_n_and_d` | プロパティの正確性 |
| `test_group_indices` | g0_idx / g1_idx の正確性 |
| `test_sparsity` | NaN 割合の計算 |
| `test_missing_file_raises` | FileNotFoundError |
| `test_real_data` (slow) | 実データの読込・形状検証 |

### 5.3 テストデータ（フィクスチャ）

| 名前 | 形状 | 生成方法 |
|------|------|---------|
| `triangle_ideals` | (3, 3) | Example 4.10 の固定値 |
| `symmetric_5d` | (50, 5) | Dirichlet(1,...,1) |
| `large_ideals` | (200, 10) | Dirichlet(2,...,2) |
| `vote_matrix` | (100, 30) | {-1, 0, 1} + 40% NaN |

全フィクスチャは `seed=0` で決定的。

---

## 6. 期待される出力

### 6.1 論文で確認された数値

| 実験 | 指標 | 期待値 |
|------|------|--------|
| PAC (m=3) | 非空率 | 100% |
| PAC (m=10) | 反平行率 | 0.0% |
| Pairwise (m=3) | 成功率 | ~73% |
| Pairwise (m=10) | 成功率 | ~98% |
| Aggregation (m=5) | Median G0%gain | ~57% |
| Mechanism (全 m) | SP 違反率 | 0.000% (exact) |
| Mechanism (全 m) | パネル独立性 | 0.0 (exact) |
| Sharding (ρ=0.30, m=5) | キャプチャ軸数 | 0 / 500 trials |
| Constitution | J(Honest, Manip) | 0.00 |
| Constitution | Contentious | 139 |

### 6.2 再現性

- 全関数が `seed` パラメータを受け取り、`np.random.RandomState(seed)` で制御
- `uv.lock` によりパッケージバージョン固定
- 浮動小数点演算の差異により $\pm 1\%$ 程度の変動あり

---

## 7. 設計原則

1. **論文との 1:1 対応**: 各関数・データクラスが論文の定義・定理に直接対応。
   docstring に該当する式番号を記載。
2. **純粋関数**: 実験関数は副作用なし（print なし）。CLI が表示を担当。
3. **合成データでテスト可能**: 実 CCAI データなしで 75/76 テストが通る。
4. **型ヒント完備**: `NDArray[np.floating]` ベースの型エイリアス。
5. **uv ファースト**: `pyproject.toml` + `uv.lock` で再現可能な環境。
