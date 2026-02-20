# テスト監査結果

## 手法

各モジュールの関数をダミー値を返すスタブに差し替え、テストを実行。
「計算せずに通るテスト」を特定した。

## 結果

### test_experiments.py: **20/20 がダミーでパス（全滅）**

| テスト | 問題 |
|--------|------|
| test_returns_rates | `0 <= 0.95 <= 1` で通る |
| test_high_dimensional_nonempty | `0.95 > 0.9` で通る |
| test_antiparallel_fraction_bounded | `0 <= 0.02 <= 1` で通る |
| test_mean_cos_is_float | `isinstance(0.05, float)` で通る |
| test_returns_five_rows | ダミー5行返せば通る |
| test_rule_names | ダミー名前返せば通る |
| test_shifts_nonnegative | `0.001 >= 0` で通る |
| test_frac_in_01 | `0 <= 0.5 <= 1` で通る |
| **test_sp_violation_zero** | **`0.0 == 0.0` — ダミーが0を返すだけで通る** |
| **test_panel_independence_zero** | **同上** |
| test_majority_captures_most | `2 > 0` で通る |
| test_minority_captures_few | `2 < 3` で通る |
| test_shifts_lists_same_length | ダミーリスト長合わせるだけ |
| test_triangle_logrolling | `90 > 0` で通る |
| test_high_dim_high_rate | `0.9 > 0.8` で通る |
| test_gains_positive | ダミー正値返せば通る |
| test_jaccard_bounds | `0 <= 0.4 <= 1` で通る |
| test_exited_bounded | `0 <= 8 <= 15` で通る |
| test_contentious_nonnegative | `10 >= 0` で通る |
| test_custom_gap_threshold | ダミーが閾値で減少するよう実装 |

### test_mechanism.py: **16/16 が medians=0 スタブでパス**

メディアン計算を `0.0` に差し替えても全テストが通る。
- `test_unanimous_recovery`: μ_A ≈ ideal なので 0*axes を加えても偶然近い
- `test_single_agent_cannot_benefit`: 射影は実計算だがπの正しさは未検証
- 「最終出力πが投票結果を正しく反映しているか」を検証するテストが皆無

### test_core.py: 31/35 パス（概ね健全だが1件問題）

- `test_gradient_direction`: 定数勾配 [0.1, 0.1, 0.1] でも偶然 `dot(g, p-π) > 0` が成立
  → 勾配の具体的な数値を検証していない
