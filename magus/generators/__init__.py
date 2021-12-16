from .random import SPGGenerator, MoleculeSPGGenerator
from .ga import GAGenerator
from ..operations import op_dict, get_default_op
import logging


log = logging.getLogger(__name__)


def get_random_generator(p_dict):
    if p_dict['molMode']:
        return MoleculeSPGGenerator(**p_dict)
    else:
        return SPGGenerator(**p_dict)


def get_ga_generator(p_dict):
    operators = get_default_op(p_dict)
    if 'OffspringCreator' in p_dict:
        operators.update(p_dict['OffspringCreator'])
    num = 3 * int((1 - p_dict['randFrac']) * p_dict['popSize'] / len(operators)) + 1
    op_nums, op_list = [], []
    for op_name, para in operators.items():
        assert op_name in op_dict, '{} not in op_dict'.format(op_name)
        op_list.append(op_dict[op_name](**para))
        if 'num' not in para:
            para['num'] = num
        op_nums.append(para['num'])
    return GAGenerator(op_nums, op_list, **p_dict)
