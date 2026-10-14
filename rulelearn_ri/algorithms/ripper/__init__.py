"""
The :mod:`ripper.RipperExplainer` module includes rule based classifier RIPPER.
"""


from .ripper import RipperExplainer
from .single_rule import optimize_single_rule


__all__ = ['RipperExplainer', 'optimize_single_rule']
