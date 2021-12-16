import logging
from magus.utils import load_plugins, COMPARATOR_PLUGIN, COMPARATOR_CONNECT_PLUGIN


load_plugins(__file__, 'magus.comparators')


log = logging.getLogger(__name__)


def get_comparator(p_dict):
    comparators = {
        'connect': 'or', 
        'naive': {},
        'ase-zurek': {},
        }
    if 'Comparator' in p_dict:
        comparators.update(p_dict['Comparator'])
    comparator_list = []
    for comparator_name, para in comparators.items():
        if comparator_name == 'connect':
            continue
        comparator_list.append(COMPARATOR_PLUGIN[comparator_name](**{**p_dict, **para}))
    return COMPARATOR_CONNECT_PLUGIN[comparators['connect']](comparator_list)
