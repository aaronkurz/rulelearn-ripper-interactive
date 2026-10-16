# rulelearn-ripper-interactive

A rule induction toolkit for learning readable, editable rules from data. It is a fork of
[IBM's rulelearn](https://github.com/IBM/rulelearn) that adds **interactive, constrained
single-rule optimization** for RIPPER. You can grow and prune a single conjunctive rule
while forcing some features or predicates into it and keeping others out.

The package installs as `rulelearn_ri`, so it can be installed side by side with the
upstream `rulelearn` package.

## Installation

```bash
pip install rulelearn-ripper-interactive
```

or, to add it to a [uv](https://docs.astral.sh/uv/) project:

```bash
uv add rulelearn-ripper-interactive
```

The core install covers RIPPER and TRXF (rule interchange, metrics and PMML export).
The other algorithms need heavier dependencies, available as extras:

```bash
pip install "rulelearn-ripper-interactive[rbm]"   # BRCG / GLRM (cvxpy, ecos, matplotlib)
pip install "rulelearn-ripper-interactive[r2n]"   # Relational Rule Network (torch)
pip install "rulelearn-ripper-interactive[all]"   # everything
```

With uv, use `uv add "rulelearn-ripper-interactive[rbm]"` (or `[r2n]`, `[all]`) instead.

Requires Python 3.10 or newer.

## Single-rule optimization

`optimize_single_rule` learns one conjunctive rule for a target label and returns it as a
TRXF `Conjunction`:

```python
import pandas as pd
from rulelearn_ri.algorithms.ripper import optimize_single_rule

df = pd.DataFrame({
    'x': [1.0, 2.0, 3.0, 4.0],
    'target': ['True', 'True', 'False', 'False'],
})
X, y = df.drop(columns='target'), df['target']

rule = optimize_single_rule(X, y, target_label='True')
print(rule)
```

Constraints let you steer the rule:

```python
from rulelearn_ri.trxf.core import Feature, Predicate, Relation

# Force the feature `x` to appear in the rule (the boundary is still optimized)
optimize_single_rule(X, y, target_label='True', fixed_attributes=['x'])

# Force a specific predicate, with a fixed boundary, into the rule
optimize_single_rule(
    X, y, target_label='True',
    fixed_predicates=[Predicate(Feature('x'), Relation.LE, 2.5)],
)

# Keep a feature out of the rule (must not clash with any forced feature or predicate)
optimize_single_rule(X, y, target_label='True', forbidden_attributes=['x'])
```

Further options: `pruning_threshold` (minimum number of instances for the grow/prune
split, default 20), `random_state` (seed for the split, default 0) and `ratio`
(fraction of the data used for growing, default 2/3).

## Included algorithms

- Fast Effective Rule Induction (RIPPER) ([William W Cohen, 1995](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.2612&rep=rep1&type=pdf)),
  including the full rule-set learner `RipperExplainer`
- Boolean Decision Rules via Column Generation (Light Edition) ([Dash et al., 2018](https://papers.nips.cc/paper/7716-boolean-decision-rules-via-column-generation)), extra `rbm`
- Generalized Linear Rule Models ([Wei et al., 2019](http://proceedings.mlr.press/v97/wei19a.html)), extra `rbm`
- Relational Rule Network (R2N) ([Kusters et al., 2022](https://arxiv.org/abs/2201.06515)), extra `r2n`
- TRXF (Technical Rule Interchange Format): a common rule-set representation with evaluation tools
  and PMML export

Example notebooks are in the [`examples/`](https://github.com/aaronkurz/rulelearn-ripper-interactive/tree/main/examples) directory of the repository.

## Development

This project uses [uv](https://docs.astral.sh/uv/):

```bash
git clone git@github.com:aaronkurz/rulelearn-ripper-interactive.git
cd rulelearn-ripper-interactive
uv sync --all-extras
uv run pytest
```

### Releasing

1. Bump `version` in `pyproject.toml`.
2. Build and check the distributions:
   ```bash
   rm -rf dist && uv build
   uvx twine check --strict dist/*
   ```
3. Upload to TestPyPI first, then check it installs in a clean environment:
   ```bash
   uv publish --publish-url https://test.pypi.org/legacy/ --token <testpypi-token>
   ```
4. Upload to PyPI and tag the release:
   ```bash
   uv publish --token <pypi-token>
   git tag v<version> && git push origin v<version>
   ```

## Acknowledgements

This package is derived from [rulelearn](https://github.com/IBM/rulelearn) by IBM,
originally released as part of [AI Explainability 360](https://github.com/Trusted-AI/AIX360).
See [NOTICE](https://github.com/aaronkurz/rulelearn-ripper-interactive/blob/main/NOTICE) for details.

## License

Apache License 2.0. See [LICENSE](https://github.com/aaronkurz/rulelearn-ripper-interactive/blob/main/LICENSE).
