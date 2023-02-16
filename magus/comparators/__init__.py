import logging
from magus.utils import load_plugins, COMPARATOR_PLUGIN, COMPARATOR_CONNECT_PLUGIN


log = logging.getLogger(__name__)


def get_comparator(p_dict):
    load_plugins(__file__, 'magus.comparators')
    comparators = {
        'connect': 'or', 
        'comparator_list': ['naive', 'zurek'],
        }
    if 'Comparator' in p_dict:
        comparators.update(p_dict['Comparator'])
    comparator_list = []
    for comparator_name in comparators['comparator_list']:
        if comparator_name in comparators:
            p_dict_ = {**p_dict, **comparators[comparator_name]}
        else:
            p_dict_ = {**p_dict}
        comparator_list.append(COMPARATOR_PLUGIN[comparator_name](**p_dict_))
    return COMPARATOR_CONNECT_PLUGIN[comparators['connect']](comparator_list)
