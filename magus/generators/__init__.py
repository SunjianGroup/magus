from .random import SPGGenerator, MoleculeSPGGenerator, LayerSPGGenerator
from .ga import GAGenerator, AutoOPRatio
from ..operations import op_dict, get_default_op
import logging
import numpy as np


log = logging.getLogger(__name__)


def get_random_generator(p_dict):
    if p_dict['molMode']:
        return MoleculeSPGGenerator(**p_dict)
    elif p_dict['structureType'] == 'layer':
        return LayerSPGGenerator(**p_dict)
    else:
        return SPGGenerator(**p_dict)


def get_ga_generator(p_dict):
    operators = get_default_op(p_dict)
    if 'OffspringCreator' in p_dict:
        operators.update(p_dict['OffspringCreator'])

    op_list, op_prob = _cal_op_prob_(operators, op_dict)

    if p_dict['autoOpRatio']:
        return AutoOPRatio(op_list, op_prob, **p_dict)
    else:
        return GAGenerator(op_list, op_prob, **p_dict)

def _cal_op_prob_(operators, op_dict):
    op_list, op_prob = [], []
    for op_name, para in operators.items():
        assert op_name in op_dict, '{} not in op_dict'.format(op_name)
        op_list.append(op_dict[op_name](**para))
        if 'prob' not in para:
            para['prob'] = -1.
        op_prob.append(para['prob'])
    op_prob = np.array(op_prob)
    sum_prob = np.sum(op_prob[op_prob > 0])
    assert sum_prob <= 1, "Please cheak probability settings"
    if len(op_prob[op_prob < 0]) > 0:
        op_prob[op_prob < 0] = (1 - sum_prob) / len(op_prob[op_prob < 0])
    return op_list, op_prob
