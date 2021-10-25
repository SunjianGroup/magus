import logging
from .naive import NaiveComparator
from .bruteforce import ZurekComparator
from .base import OrGate, AndGate


calc_dict = {
    'naive': NaiveComparator,
    'bruteforce': ZurekComparator,
    }
connect_dict = {
    'and': AndGate,
    'or': OrGate,
}

log = logging.getLogger(__name__)


def get_one_comparator(comparator_type, p_dict):
    if comparator_type not in comparator_dict:
        raise Exception('Unknown comparator: {}'.format(comparator_type))
    return comparator_dict[comparator_type](**p_dict)


def get_comparator(p_dict=None):
    if p_dict is None:
        return OrGate([NaiveComparator(), ZurekComparator()])
    if type(p_dict['type']) is list:
        comparators = [get_one_comparator(t, p_dict) for t in p_dict['type']]
        if 'connect' not in p_dict:
            p_dict['connect'] = 'and'
        return connect_dict[p_dict['connect']](comparators)
    return get_one_calculator(p_dict['type'], p_dict)
