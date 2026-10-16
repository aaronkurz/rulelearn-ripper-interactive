import logging
import sys
import unittest

import numpy as np
import pandas as pd

from rulelearn_ri.algorithms.ripper import RipperExplainer
from rulelearn_ri.algorithms.ripper.base import _encoding_for_parallel
from rulelearn_ri.algorithms.ripper.single_rule import (
    _grow_rule_standalone, _grow_rule_with_required_features, optimize_single_rule)
from rulelearn_ri.trxf.core.conjunction import Conjunction
from rulelearn_ri.trxf.core.feature import Feature
from rulelearn_ri.trxf.core.predicate import Predicate, Relation


def create_simple_df():
    """Data where pos = (int_col == 1) and neg = (int_col == 2)."""
    df = pd.DataFrame({
        'int_col': [1, 1, 1, 2, 2, 2],
        'float_col': [1.0, 1.5, 2.0, 3.0, 3.5, 4.0],
        'str_col': ['foo', 'foo', 'bar', 'bar', 'baz', 'baz'],
        'target': ['True', 'True', 'True', 'False', 'False', 'False']
    })
    df = df.astype({
        'int_col': 'int64', 'float_col': 'float64',
        'str_col': 'str', 'target': 'str'
    })
    return df


def create_mixed_nominal_df():
    """Data where pos = (str_col == 'foo') and neg = (str_col == 'bar')."""
    df = pd.DataFrame({
        'int_col': [1, 2, 3, 1, 2, 3],
        'float_col': [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        'str_col': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
        'target': ['True', 'True', 'True', 'False', 'False', 'False']
    })
    df = df.astype({
        'int_col': 'int64', 'float_col': 'float64',
        'str_col': 'str', 'target': 'str'
    })
    return df


def create_separable_float_df():
    """Data where pos = (float_col >= 3.0) and neg = (float_col <= 2.5)."""
    df = pd.DataFrame({
        'float_a': [3.0, 3.5, 4.0, 1.0, 1.5, 2.0],
        'float_b': [5.0, 5.5, 6.0, 1.0, 1.5, 2.0],
        'target': ['True', 'True', 'True', 'False', 'False', 'False']
    })
    df = df.astype({'float_a': 'float64', 'float_b': 'float64', 'target': 'str'})
    return df


def create_degenerate_df():
    """Data with no discriminating features."""
    df = pd.DataFrame({
        'float_a': [1.0, 1.0, 1.0],
        'float_b': [2.0, 2.0, 2.0],
        'target': ['True', 'False', 'False']
    })
    df = df.astype({'float_a': 'float64', 'float_b': 'float64', 'target': 'str'})
    return df


def create_xor_grow_input():
    """XOR data where no single literal has positive gain. The constant
    first column makes index 0 of the condition matrix a no-op literal."""
    df = pd.DataFrame({
        'c': [1.0, 1.0, 1.0, 1.0],
        'a': [0.0, 1.0, 0.0, 1.0],
        'b': [0.0, 1.0, 1.0, 0.0],
    })
    y = np.array([True, True, False, False])
    condition_mat = _encoding_for_parallel(
        df, {name: idx for idx, name in enumerate(df.columns)})
    data = df.to_numpy(dtype=float, copy=True)
    return data[y], data[~y], condition_mat


class TestOptimizeSingleRule(unittest.TestCase):

    def test_unconstrained_rule(self):
        """Grow a single rule without any constraints."""
        df = create_simple_df()
        X = df.drop(columns=['target'])
        y = df['target']

        conjunction = optimize_single_rule(X, y, target_label='True')
        self.assertIsInstance(conjunction, Conjunction)
        self.assertGreater(len(conjunction.predicates), 0)

        # The rule should correctly classify the training data
        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            if row['target'] == 'True':
                self.assertTrue(conjunction.evaluate(assignment),
                                'Rule should match positive instance')
            else:
                self.assertFalse(conjunction.evaluate(assignment),
                                 'Rule should not match negative instance')

    def test_fixed_attributes_numeric(self):
        """Force inclusion of a specific numeric column."""
        df = create_separable_float_df()
        X = df.drop(columns=['target'])
        y = df['target']

        conjunction = optimize_single_rule(
            X, y, target_label='True',
            fixed_attributes=['float_a']
        )

        # Verify float_a appears in the rule
        predicate_names = [p.feature.variable_names[0]
                           for p in conjunction.predicates]
        self.assertIn('float_a', predicate_names)

        # Rule should be correct
        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            if row['target'] == 'True':
                self.assertTrue(conjunction.evaluate(assignment))
            else:
                self.assertFalse(conjunction.evaluate(assignment))

    def test_fixed_attributes_multiple(self):
        """Force inclusion of multiple columns."""
        df = create_separable_float_df()
        X = df.drop(columns=['target'])
        y = df['target']

        conjunction = optimize_single_rule(
            X, y, target_label='True',
            fixed_attributes=['float_a', 'float_b']
        )

        predicate_names = [p.feature.variable_names[0]
                           for p in conjunction.predicates]
        self.assertIn('float_a', predicate_names)
        self.assertIn('float_b', predicate_names)

        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            if row['target'] == 'True':
                self.assertTrue(conjunction.evaluate(assignment))
            else:
                self.assertFalse(conjunction.evaluate(assignment))

    def test_fixed_attributes_nominal(self):
        """Force inclusion of a nominal (string) column."""
        df = create_mixed_nominal_df()
        X = df.drop(columns=['target'])
        y = df['target']

        conjunction = optimize_single_rule(
            X, y, target_label='True',
            fixed_attributes=['str_col']
        )

        predicate_names = [p.feature.variable_names[0]
                           for p in conjunction.predicates]
        self.assertIn('str_col', predicate_names)

        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            if row['target'] == 'True':
                self.assertTrue(conjunction.evaluate(assignment))
            else:
                self.assertFalse(conjunction.evaluate(assignment))

    def test_fixed_predicates_numeric(self):
        """Give a fixed predicate with boundary."""
        df = create_separable_float_df()
        X = df.drop(columns=['target'])
        y = df['target']

        fixed_pred = Predicate(Feature('float_a'), Relation.GE, 2.5)

        conjunction = optimize_single_rule(
            X, y, target_label='True',
            fixed_predicates=[fixed_pred]
        )

        # The fixed predicate should be preserved as-is
        found = False
        for p in conjunction.predicates:
            if (p.feature.variable_names[0] == 'float_a'
                    and p.relation == Relation.GE
                    and p.value == 2.5):
                found = True
                break
        self.assertTrue(found, 'Fixed predicate float_a >= 2.5 must be in the rule')

        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            if row['target'] == 'True':
                self.assertTrue(conjunction.evaluate(assignment))
            else:
                self.assertFalse(conjunction.evaluate(assignment))

    def test_fixed_predicates_nominal(self):
        """Give a fixed nominal predicate."""
        df = create_mixed_nominal_df()
        X = df.drop(columns=['target'])
        y = df['target']

        fixed_pred = Predicate(Feature('str_col'), Relation.EQ, 'foo')

        conjunction = optimize_single_rule(
            X, y, target_label='True',
            fixed_predicates=[fixed_pred]
        )

        found = False
        for p in conjunction.predicates:
            if (p.feature.variable_names[0] == 'str_col'
                    and p.relation == Relation.EQ
                    and p.value == 'foo'):
                found = True
                break
        self.assertTrue(found, 'Fixed predicate str_col == foo must be in the rule')

        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            if row['target'] == 'True':
                self.assertTrue(conjunction.evaluate(assignment))
            else:
                self.assertFalse(conjunction.evaluate(assignment))

    def test_fixed_predicates_exact_boundary(self):
        """Fixed predicate with an exact boundary that perfectly separates the data."""
        df = create_separable_float_df()
        X = df.drop(columns=['target'])
        y = df['target']

        fixed_pred = Predicate(Feature('float_a'), Relation.GE, 3.0)

        conjunction = optimize_single_rule(
            X, y, target_label='True',
            fixed_predicates=[fixed_pred]
        )
        self.assertIsInstance(conjunction, Conjunction)

        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            if row['target'] == 'True':
                self.assertTrue(conjunction.evaluate(assignment))
            else:
                self.assertFalse(conjunction.evaluate(assignment))

    def test_fixed_attributes_and_predicates_combined(self):
        """Both fixed_attributes and fixed_predicates can be used together.
        Attributes already covered by predicates are skipped."""
        df = create_separable_float_df()
        X = df.drop(columns=['target'])
        y = df['target']

        # float_a via predicate, float_b via attribute
        fixed_preds = [Predicate(Feature('float_a'), Relation.GE, 2.5)]
        fixed_attrs = ['float_b']

        conjunction = optimize_single_rule(
            X, y, target_label='True',
            fixed_predicates=fixed_preds,
            fixed_attributes=fixed_attrs
        )

        predicate_names = [p.feature.variable_names[0]
                           for p in conjunction.predicates]
        self.assertIn('float_a', predicate_names,
                      'float_a from fixed_predicates must appear')
        self.assertIn('float_b', predicate_names,
                      'float_b from fixed_attributes must appear')

        # The fixed predicate must be kept as-is
        found = False
        for p in conjunction.predicates:
            if (p.feature.variable_names[0] == 'float_a'
                    and p.relation == Relation.GE
                    and p.value == 2.5):
                found = True
                break
        self.assertTrue(found, 'Fixed predicate value must be unchanged')

        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            if row['target'] == 'True':
                self.assertTrue(conjunction.evaluate(assignment))
            else:
                self.assertFalse(conjunction.evaluate(assignment))

    def test_fixed_attributes_and_predicates_overlapping(self):
        """When fixed_attributes includes an attribute already covered
        by fixed_predicates, it is skipped (not duplicated)."""
        df = create_separable_float_df()
        X = df.drop(columns=['target'])
        y = df['target']

        # float_a appears in both, float_b only in attributes
        fixed_preds = [Predicate(Feature('float_a'), Relation.GE, 2.5)]
        fixed_attrs = ['float_a', 'float_b']

        conjunction = optimize_single_rule(
            X, y, target_label='True',
            fixed_predicates=fixed_preds,
            fixed_attributes=fixed_attrs
        )

        predicate_names = [p.feature.variable_names[0]
                           for p in conjunction.predicates]
        self.assertIn('float_a', predicate_names)
        self.assertIn('float_b', predicate_names)

        # float_a should appear exactly once via the fixed predicate
        float_a_count = predicate_names.count('float_a')
        self.assertEqual(float_a_count, 1,
                         'float_a should not be duplicated')

        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            if row['target'] == 'True':
                self.assertTrue(conjunction.evaluate(assignment))
            else:
                self.assertFalse(conjunction.evaluate(assignment))

    def test_forbidden_attributes_numeric(self):
        """Forbid a numeric column from appearing in the rule."""
        df = create_simple_df()
        X = df.drop(columns=['target'])
        y = df['target']

        conjunction = optimize_single_rule(
            X, y, target_label='True',
            forbidden_attributes=['int_col']
        )

        predicate_names = [p.feature.variable_names[0]
                           for p in conjunction.predicates]
        self.assertGreater(len(predicate_names), 0)
        self.assertNotIn('int_col', predicate_names)

        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            if row['target'] == 'True':
                self.assertTrue(conjunction.evaluate(assignment))
            else:
                self.assertFalse(conjunction.evaluate(assignment))

    def test_forbidden_attributes_nominal(self):
        """Forbid a nominal (string) column from appearing in the rule."""
        df = create_simple_df()
        X = df.drop(columns=['target'])
        y = df['target']

        conjunction = optimize_single_rule(
            X, y, target_label='True',
            forbidden_attributes=['str_col']
        )

        predicate_names = [p.feature.variable_names[0]
                           for p in conjunction.predicates]
        self.assertGreater(len(predicate_names), 0)
        self.assertNotIn('str_col', predicate_names)

        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            if row['target'] == 'True':
                self.assertTrue(conjunction.evaluate(assignment))
            else:
                self.assertFalse(conjunction.evaluate(assignment))

    def test_forbidden_attributes_multiple(self):
        """Forbid multiple columns at once."""
        df = create_separable_float_df()
        X = df.drop(columns=['target'])
        y = df['target']

        conjunction = optimize_single_rule(
            X, y, target_label='True',
            forbidden_attributes=['float_a', 'float_b']
        )

        predicate_names = [p.feature.variable_names[0]
                           for p in conjunction.predicates]
        self.assertNotIn('float_a', predicate_names)
        self.assertNotIn('float_b', predicate_names)

    def test_forbidden_with_fixed_attributes_combined(self):
        """Fixed and forbidden attributes can coexist without a clash."""
        df = create_simple_df()
        X = df.drop(columns=['target'])
        y = df['target']

        conjunction = optimize_single_rule(
            X, y, target_label='True',
            fixed_attributes=['float_col'],
            forbidden_attributes=['int_col']
        )

        predicate_names = [p.feature.variable_names[0]
                           for p in conjunction.predicates]
        self.assertIn('float_col', predicate_names)
        self.assertNotIn('int_col', predicate_names)

        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            if row['target'] == 'True':
                self.assertTrue(conjunction.evaluate(assignment))
            else:
                self.assertFalse(conjunction.evaluate(assignment))

    def test_forbidden_clash_with_fixed_attributes(self):
        """Forbidden and fixed attributes must not overlap."""
        df = create_simple_df()
        X = df.drop(columns=['target'])
        y = df['target']

        with self.assertRaises(ValueError):
            optimize_single_rule(
                X, y, target_label='True',
                fixed_attributes=['int_col'],
                forbidden_attributes=['int_col']
            )

    def test_forbidden_clash_with_fixed_predicates(self):
        """Forbidden attributes must not overlap with fixed predicate attributes."""
        df = create_simple_df()
        X = df.drop(columns=['target'])
        y = df['target']

        fixed_pred = Predicate(Feature('int_col'), Relation.EQ, 1)
        with self.assertRaises(ValueError):
            optimize_single_rule(
                X, y, target_label='True',
                fixed_predicates=[fixed_pred],
                forbidden_attributes=['int_col']
            )

    def test_forbidden_all_discriminating_features(self):
        """When forbidden attributes remove all separation, the rule is still
        returned (best-effort) without the forbidden attributes and without errors."""
        df = create_simple_df()
        X = df.drop(columns=['target'])
        y = df['target']

        conjunction = optimize_single_rule(
            X, y, target_label='True',
            forbidden_attributes=['int_col', 'float_col', 'str_col']
        )

        predicate_names = [p.feature.variable_names[0]
                           for p in conjunction.predicates]
        for name in ('int_col', 'float_col', 'str_col'):
            self.assertNotIn(name, predicate_names)

    def test_target_label_not_in_y_raises(self):
        """Like RipperExplainer.fit, a target_label absent from y (no positives) is an error."""
        df = pd.DataFrame({
            'float_a': [1.0, 2.0, 3.0],
            'target': ['False', 'False', 'False']
        }).astype({'float_a': 'float64', 'target': 'str'})

        X = df.drop(columns=['target'])
        y = df['target']

        with self.assertRaisesRegex(ValueError, r"target_label 'True' not found in y \(values: \['False'\]\)"):
            optimize_single_rule(
                X, y, target_label='True',
                fixed_attributes=['float_a']
            )

    def test_target_label_of_wrong_type_raises(self):
        """A string label for a boolean y matches nothing and must not silently yield an empty rule."""
        X = pd.DataFrame({'float_a': [1.0, 2.0, 3.0, 4.0]})
        y = pd.Series([True, True, False, False])

        with self.assertRaisesRegex(ValueError, r"target_label 'True' not found in y \(values: \[False, True\]\)"):
            optimize_single_rule(X, y, target_label='True')

    def test_constrained_rule_uses_ripper_compatibly(self):
        """Verify the standalone grow function classifies correctly
        and that a RipperExplainer can be built from the same data.
        The standalone rule may differ from the full RipperExplainer
        since Ripper uses multi-rule optimization.
        Note: RipperExplainer.fit mutates the DataFrame in place, so
        separate copies are used for each call."""
        df = create_simple_df()
        X = df.drop(columns=['target'])
        y = df['target']

        # Full RipperExplainer
        X_ripper = X.copy()
        y_ripper = y.copy()
        ripper = RipperExplainer()
        ripper.fit(X_ripper, y_ripper, target_label='True')
        ripper_ruleset = ripper.explain()
        ripper_conjunctions = ripper_ruleset.list_conjunctions()
        self.assertEqual(len(ripper_conjunctions), 1)

        # Verify Ripper classifies correctly
        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            ripper_result = ripper_conjunctions[0].evaluate(assignment)
            if row['target'] == 'True':
                self.assertTrue(ripper_result)
            else:
                self.assertFalse(ripper_result)

        # Single rule optimizer also classifies correctly
        X_single = X.copy()
        y_single = y.copy()
        single_conjunction = optimize_single_rule(
            X_single, y_single, target_label='True')
        for _, row in df.iterrows():
            assignment = row.drop('target').to_dict()
            single_result = single_conjunction.evaluate(assignment)
            if row['target'] == 'True':
                self.assertTrue(single_result)
            else:
                self.assertFalse(single_result)

    def test_grow_standalone_skips_noop_literal(self):
        """A literal that filters neither set must not end up in the rule."""
        pos, neg, condition_mat = create_xor_grow_input()

        rule = _grow_rule_standalone(pos, neg, condition_mat)
        self.assertEqual(rule, [])

    def test_grow_required_skips_noop_literal_in_free_phase(self):
        """The forced literal stays, but no no-op literal is added afterwards."""
        pos, neg, condition_mat = create_xor_grow_input()

        rule = _grow_rule_with_required_features(pos, neg, {1}, condition_mat)
        self.assertEqual([lit.name for lit in rule], [1])


if __name__ == '__main__':
    unittest.main(warnings='ignore')
