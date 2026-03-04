"""Operator map from invariant comparison keys to vectorized predicates."""

OP_MAP = {
    "eq": lambda s, v: s == v,
    "neq": lambda s, v: s != v,
    "in": lambda s, v: s.isin(v),
    "not_in": lambda s, v: ~s.isin(v),
    "gt": lambda s, v: s > v,
    "gte": lambda s, v: s >= v,
    "lt": lambda s, v: s < v,
    "lte": lambda s, v: s <= v,
}
