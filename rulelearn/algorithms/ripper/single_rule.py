import numpy as np
from pandas import DataFrame, Series

from rulelearn.algorithms.ripper.base import _encoding_for_parallel, init_encoder, encode_nominal, \
    _split_instances
from rulelearn.algorithms.ripper.binding import _find_best_literal, Literal, LE, GE, EQ, _bind_literal, \
    _filter_contradicted_instances, append_literal, _bind_rule
from rulelearn.algorithms.ripper.pruning import _pruning_irep

from rulelearn.trxf.core import Conjunction
from rulelearn.trxf.core import Feature
from rulelearn.trxf.core import Predicate, Relation


def optimize_single_rule(
        train: DataFrame,
        y: Series,
        target_label,
        fixed_attributes=None,
        fixed_predicates=None,
        forbidden_attributes=None,
        pruning_threshold: int = 20,
        random_state: int = 0,
        ratio: float = 2 / 3,
):
    """
    Optimize a single conjunctive rule for the given target label.

    Optionally, certain features can be forced to appear in the rule via ``fixed_attributes``,
    and/or certain whole predicates (with fixed boundaries) can be given via ``fixed_predicates``.
    Both can be specified together; attributes already covered by fixed_predicates are skipped.
    Conversely, ``forbidden_attributes`` lists features that must NOT appear in the rule; these
    must not overlap with the forced attributes or predicates, otherwise a ValueError is raised.
    The algorithm grows additional literals and prunes to produce a single optimized rule.

    Args:
        train (pd.DataFrame): Training features
        y (pd.Series): Training labels
        target_label: The positive class label for the rule
        fixed_attributes (list of str, optional): Column names that must appear in the rule
        fixed_predicates (list of Predicate, optional): Predicates that must be included as-is
        forbidden_attributes (list of str, optional): Column names that must not appear in the rule
        pruning_threshold (int): Minimum instances for grow/prune split. Defaults to 20.
        random_state (int): Random seed for data splitting. Defaults to 0.
        ratio (float): Fraction of data for growing. Defaults to 2/3.

    Returns:
        Conjunction: The optimized rule as a TRXF Conjunction
    """
    train = train.copy()
    y = y.copy()

    nominal_encoders = init_encoder(train)
    encode_nominal(nominal_encoders, train)
    column_name_list = train.columns.values
    column_name_index_map = {name: idx for idx, name in enumerate(column_name_list)}
    condition_mat = _encoding_for_parallel(train, column_name_index_map)

    # Build the unified forced feature set and optional predefined rule
    forced_literals = []
    forced_features = set()
    if fixed_predicates is not None:
        forced_literals = _predicates_to_literals(
            fixed_predicates, column_name_index_map, nominal_encoders)
        forced_features = set(lit.name for lit in forced_literals)

    required_features = set()
    if fixed_attributes is not None:
        required_features = set(column_name_index_map[attr]
                                for attr in fixed_attributes)

    forbidden_features = set()
    if forbidden_attributes is not None:
        forbidden_features = set(column_name_index_map[attr]
                                 for attr in forbidden_attributes)

    _validate_forbidden_clash(column_name_list, forbidden_features,
                              required_features | forced_features)

    train_arr = train.to_numpy(dtype=np.float64)
    y_arr = y.values if isinstance(y, Series) else np.array(y)

    pos = train_arr[y_arr == target_label]
    neg = train_arr[y_arr != target_label]

    neg = _filter_contradicted_instances(pos, neg)

    if len(pos) == 0:
        return Conjunction([])

    # fixed_predicates already cover some attributes -- skip those
    remaining_required_features = required_features - forced_features

    has_predicates = len(forced_literals) > 0
    has_remaining_attrs = len(remaining_required_features) > 0

    if len(pos) > pruning_threshold:
        pos_grow, pos_prune, neg_grow, neg_prune = _split_instances(
            pos, neg, ratio, random_state)
    else:
        pos_grow = pos_prune = pos
        neg_grow = neg_prune = neg

    if has_predicates:
        # Grow from fixed predicates first
        rule = _grow_rule_standalone(
            pos_grow, neg_grow, condition_mat, predefined_rule=forced_literals,
            forbidden_features=forbidden_features)

        if has_remaining_attrs:
            # Bind data to the rule so far, then force-include remaining attributes
            pos_bound = pos_grow
            neg_bound = neg_grow
            for lit in rule:
                pos_bound = _bind_literal(pos_bound, lit.name, lit.op,
                                          lit.num_val, lit.nom_val)
                neg_bound = _bind_literal(neg_bound, lit.name, lit.op,
                                          lit.num_val, lit.nom_val)
            rule = _grow_rule_with_required_features(
                pos_bound, neg_bound, remaining_required_features,
                condition_mat, learned_rule=rule,
                forbidden_features=forbidden_features)
    elif has_remaining_attrs:
        # Attributes only — grow from scratch forcing required features
        rule = _grow_rule_with_required_features(
            pos_grow, neg_grow, required_features, condition_mat,
            forbidden_features=forbidden_features)
    else:
        # Unconstrained — standard grow
        rule = _grow_rule_standalone(
            pos_grow, neg_grow, condition_mat,
            forbidden_features=forbidden_features)

    # All forced features (from both predicates and attributes)
    all_forced = forced_features | required_features

    if len(all_forced) > 0:
        rule = _prune_keeping_forced(pos_prune, neg_prune, rule, all_forced)
    else:
        rule = _pruning_irep(pos_prune, neg_prune, rule)

    return _rule_to_trxf_conjunction(rule, column_name_list, nominal_encoders)


def _predicates_to_literals(predicates, column_name_index_map, nominal_encoders):
    """
    Convert TRXF Predicate objects to internal Literal objects.
    """
    literals = []
    for pred in predicates:
        name = pred.feature.variable_names[0]
        col_idx = column_name_index_map[name]
        relation = pred.relation

        if relation == Relation.EQ:
            if name in nominal_encoders:
                label_map = {val: label for label, val in
                             enumerate(nominal_encoders[name].classes_)}
                nom_val = label_map[pred.value]
            else:
                nom_val = int(pred.value)
            literals.append(Literal(col_idx, EQ, -1, nom_val))
        elif relation == Relation.LE:
            literals.append(Literal(col_idx, LE, pred.value, -1))
        elif relation == Relation.GE:
            literals.append(Literal(col_idx, GE, pred.value, -1))
        else:
            raise ValueError(
                'Unsupported relation {} for fixed_predicates. '
                'Only EQ, LE, GE are supported.'.format(relation))
    return literals


def _validate_forbidden_clash(column_name_list, forbidden_features, forced_features):
    """
    Ensure that forbidden features do not overlap with features that are
    forced via fixed_attributes or fixed_predicates.
    """
    clash = forbidden_features & forced_features
    if clash:
        names = [column_name_list[idx] for idx in sorted(clash)]
        raise ValueError(
            'forbidden_attributes must not overlap with fixed_attributes or '
            'fixed_predicates. Conflicting column(s): {}'.format(', '.join(names)))


def _exclude_features(condition_mat, forbidden_features):
    """
    Return the condition matrices with literals of forbidden features removed.
    Returns None when no allowed literals remain.
    """
    if not forbidden_features:
        return condition_mat
    mask = ~np.isin(condition_mat[0], list(forbidden_features))
    if not mask.any():
        return None
    return [arr[mask] for arr in condition_mat]


def _grow_rule_standalone(pos, neg, condition_mat, predefined_rule=None,
                          forbidden_features=None):
    """
    Standalone version of RIPPER's grow rule, parameterized by condition_mat.
    """
    if len(pos) == 0:
        raise AssertionError('pos must not be empty')

    neg = _filter_contradicted_instances(pos, neg)

    if predefined_rule is not None:
        pos = _bind_rule(pos, predefined_rule)
        neg = _bind_rule(neg, predefined_rule)
        if len(pos) == 0 or len(neg) == 0:
            return list(predefined_rule)
        learned_rule = list(predefined_rule)
    else:
        learned_rule = []

    grow_mat = _exclude_features(condition_mat, forbidden_features)

    while len(neg) > 0:
        if grow_mat is None or len(pos) == 0:
            break
        pos = np.asfortranarray(pos)
        neg = np.asfortranarray(neg)

        best_literal_index = _find_best_literal(
            pos, neg,
            grow_mat[0], grow_mat[1],
            grow_mat[2], grow_mat[3]
        )
        literal = Literal(
            grow_mat[0][best_literal_index],
            grow_mat[1][best_literal_index],
            grow_mat[2][best_literal_index],
            grow_mat[3][best_literal_index]
        )
        new_pos = _bind_literal(pos, literal.name, literal.op,
                                literal.num_val, literal.nom_val)
        new_neg = _bind_literal(neg, literal.name, literal.op,
                                literal.num_val, literal.nom_val)

        # a literal that filters neither set only restricts unseen data
        if len(new_pos) == len(pos) and len(new_neg) == len(neg):
            break
        append_literal(learned_rule, literal)
        pos, neg = new_pos, new_neg

    return learned_rule


def _grow_rule_with_required_features(pos, neg, required_features, condition_mat,
                                     learned_rule=None, forbidden_features=None):
    """
    Grow a rule ensuring that every feature in ``required_features``
    (a set of column indices) appears in the rule.

    When ``learned_rule`` is given, literals are appended to it and
    filtering of contradicting instances is skipped (caller is responsible).

    Phase 1: restrict literal selection to remaining required features.
    Phase 2: grow freely to eliminate remaining negatives, excluding any
    forbidden features.
    """
    if len(pos) == 0:
        raise AssertionError('pos must not be empty')

    if learned_rule is None:
        neg = _filter_contradicted_instances(pos, neg)
        learned_rule = []
    remaining = set(required_features)

    # Phase 1: force-include required features (one literal each)
    while remaining and len(neg) > 0:
        mask = np.isin(condition_mat[0], list(remaining))
        if not mask.any():
            break
        filtered_mat = [arr[mask] for arr in condition_mat]

        pos_f = np.asfortranarray(pos)
        neg_f = np.asfortranarray(neg)
        best_idx = _find_best_literal(
            pos_f, neg_f,
            filtered_mat[0], filtered_mat[1],
            filtered_mat[2], filtered_mat[3]
        )
        literal = Literal(
            filtered_mat[0][best_idx],
            filtered_mat[1][best_idx],
            filtered_mat[2][best_idx],
            filtered_mat[3][best_idx]
        )
        append_literal(learned_rule, literal)
        remaining.discard(literal.name)

        pos = _bind_literal(pos, literal.name, literal.op,
                            literal.num_val, literal.nom_val)
        neg = _bind_literal(neg, literal.name, literal.op,
                            literal.num_val, literal.nom_val)

        if len(pos) == 0:
            break

    # Phase 1b: force-include any remaining required features even if
    # negatives are already eliminated (user explicitly requested them)
    for feat_idx in sorted(remaining):
        mask = condition_mat[0] == feat_idx
        if not mask.any():
            continue
        filtered_mat = [arr[mask] for arr in condition_mat]

        pos_f = np.asfortranarray(pos)
        neg_f = np.asfortranarray(neg)
        best_idx = _find_best_literal(
            pos_f, neg_f,
            filtered_mat[0], filtered_mat[1],
            filtered_mat[2], filtered_mat[3]
        )
        literal = Literal(
            filtered_mat[0][best_idx],
            filtered_mat[1][best_idx],
            filtered_mat[2][best_idx],
            filtered_mat[3][best_idx]
        )
        append_literal(learned_rule, literal)

        pos = _bind_literal(pos, literal.name, literal.op,
                            literal.num_val, literal.nom_val)
        neg = _bind_literal(neg, literal.name, literal.op,
                            literal.num_val, literal.nom_val)

    # Phase 2: grow freely
    grow_mat = _exclude_features(condition_mat, forbidden_features)

    while len(neg) > 0:
        if grow_mat is None or len(pos) == 0:
            break
        pos = np.asfortranarray(pos)
        neg = np.asfortranarray(neg)

        best_idx = _find_best_literal(
            pos, neg,
            grow_mat[0], grow_mat[1],
            grow_mat[2], grow_mat[3]
        )
        literal = Literal(
            grow_mat[0][best_idx],
            grow_mat[1][best_idx],
            grow_mat[2][best_idx],
            grow_mat[3][best_idx]
        )
        new_pos = _bind_literal(pos, literal.name, literal.op,
                                literal.num_val, literal.nom_val)
        new_neg = _bind_literal(neg, literal.name, literal.op,
                                literal.num_val, literal.nom_val)

        # a literal that filters neither set only restricts unseen data
        if len(new_pos) == len(pos) and len(new_neg) == len(neg):
            break
        append_literal(learned_rule, literal)
        pos, neg = new_pos, new_neg

    return learned_rule


def _prune_keeping_forced(pos_prune, neg_prune, rule, forced_features):
    """
    Prune only the non-forced literals. Forced literals are bound first
    to restrict the data, then pruning applies to the free suffix.
    """
    forced = [lit for lit in rule if lit.name in forced_features]
    free = [lit for lit in rule if lit.name not in forced_features]

    bound_pos = pos_prune
    bound_neg = neg_prune
    for lit in forced:
        bound_pos = _bind_literal(bound_pos, lit.name, lit.op,
                                  lit.num_val, lit.nom_val)
        bound_neg = _bind_literal(bound_neg, lit.name, lit.op,
                                  lit.num_val, lit.nom_val)

    if len(free) == 0:
        return forced

    pruned_free = _pruning_irep(bound_pos, bound_neg, free)
    return forced + pruned_free


def _rule_to_trxf_conjunction(rule, column_name_list, nominal_encoders):
    """
    Convert a list of Literal objects to a TRXF Conjunction.
    """
    conjunction = Conjunction([])
    for lit in rule:
        name = column_name_list[lit.name]
        feature = Feature(name)
        if lit.op == EQ:
            relation = Relation.EQ
            value = nominal_encoders[name].classes_[lit.nom_val]
        elif lit.op == LE:
            relation = Relation.LE
            value = lit.num_val
        else:
            relation = Relation.GE
            value = lit.num_val
        conjunction.add_predicate(Predicate(feature, relation, value))
    return conjunction
