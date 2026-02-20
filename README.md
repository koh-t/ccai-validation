# ccai-validation

Empirical validation of the **Sharded Citizens' Assembly** mechanism on [CCAI Polis data](https://github.com/saffronh/ccai).

Reproduces Section 6 of:
> *Convexity Kills Arrow but Not Coalitions: Democratic Alignment on LLM Policy Spaces*

## Quickstart

```bash
# Install (requires uv: https://docs.astral.sh/uv/)
uv sync

# Download CCAI data
mkdir -p data
# Place clean_votes.csv and clean_comments.csv from
# https://github.com/saffronh/ccai into data/

# Run all experiments
uv run ccai-run --data_dir ./data

# Run specific experiments
uv run ccai-run --data_dir ./data --experiments pac,pairwise --dims 3,5,10

# Run tests (no CCAI data needed)
uv run pytest -m "not slow"
```

## Experiments

| # | Experiment | Validates | Key Result |
|---|-----------|-----------|------------|
| 1 | PAC nonemptiness | Lemma 4.3 | 100% nonempty across all m |
| 2 | Pairwise logrolling | Example 4.10 | 73% (m=3) → 98% (m=10) success |
| 3 | Aggregation rules | Theorem 4.8 | All rules manipulable for m≥3 |
| 4 | Mechanism properties | Theorem 5.2 | SP violation = 0%, panel independence exact |
| 5 | Sharding effect | Proposition 5.3 | ρ=0.30 minority captures 0 axes / 500 trials |
| 6 | Constitution manipulation | §6.7 | J(honest, manipulated) = 0.00 |

## Project Structure

```
src/ccai_validation/
├── core.py          # ℓ₂ utility, gradients, aggregation rules, PAC
├── mechanism.py     # Sharded Citizens' Assembly (Definition 5.1)
├── data.py          # CCAI Polis data loading
├── experiments.py   # Six experiment functions (pure, no side effects)
└── cli.py           # CLI entry point
tests/               # 83 unit tests (synthetic data, no CCAI needed)
```

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) ≥ 0.5

See [SPEC.md](SPEC.md) for the full specification.

## License

MIT
