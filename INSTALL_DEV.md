# Installing the dev version (single-rule optimization)

This repository is a development fork of [`rulelearn`](https://github.com/Trusted-AI/AIX360)
that adds **single-rule optimization with constraints** — the `optimize_single_rule`
function, which grows and prunes a single conjunctive RIPPER rule, optionally forcing
certain features (`fixed_attributes`) or whole predicates (`fixed_predicates`) into the
rule.

## Install from source (recommended)

An editable install lets you tweak the code and pick up changes without reinstalling:

```bash
git clone git@github.com:aaronkurz/rulelearn-ripper-interactive.git
cd rulelearn-ripper-interactive
python -m venv venv && source venv/bin/activate
pip install -e .
```

## Alternative: install directly with pip

If you only want to use the feature (no local clone):

```bash
pip install "git+https://github.com/aaronkurz/rulelearn-ripper-interactive.git"
```

## Verify the install

```python
import pandas as pd
from rulelearn.algorithms.ripper import optimize_single_rule

df = pd.DataFrame({
    'x': [1.0, 2.0, 3.0, 4.0],
    'target': ['True', 'True', 'False', 'False'],
})

rule = optimize_single_rule(
    df.drop(columns='target'),
    df['target'],
    target_label='True',
)
print(rule)  # a TRXF Conjunction
```

You can constrain the rule to include specific features or predicates:

```python
from rulelearn.trxf.core import Feature, Predicate, Relation

# Force the feature `x` to appear in the rule
optimize_single_rule(X, y, target_label='True', fixed_attributes=['x'])

# Force a specific predicate (with fixed boundary) into the rule
optimize_single_rule(
    X, y, target_label='True',
    fixed_predicates=[Predicate(Feature('x'), Relation.LE, 2.5)],
)
```