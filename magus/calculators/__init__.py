# calculators in localopt should be moved here
from .ase import EMTCalculator, LJCalculator, XTBCalculator, QUIPCalculator
from .vasp import VaspCalculator
from .gulp import GulpCalculator
from .mtp import MTPCalculator, MTPLammpsCalculator
from .lammps import LammpsCalculator
from .base import AdjointCalculator
from string import digits
import logging
from copy import deepcopy


__all__ = ['EMTCalculator', 'LJCalculator', 'XTBCalculator', 
           'VaspCalculator', 'GulpCalculator', 'QUIPCalculator'
           'MTPCalculator', 'MTPLammpsCalculator',]
calc_dict = {
    'vasp': VaspCalculator,
    'emt': EMTCalculator,
    'lj': LJCalculator,
    'gulp': GulpCalculator,
    'xtb': XTBCalculator,
    'lammps': LammpsCalculator,
    'quip': QUIPCalculator,
    'mtp': MTPCalculator,
    'mtp-lammps': MTPLammpsCalculator,
    }
need_convert = ['jobPrefix', 'eps', 'maxStep', 'optimizer', 'maxMove', 
                'relaxLattice', 'exeCmd', 'calculator']
log = logging.getLogger(__name__)


def get_one_calculator(p_dict):
    if 'calculator' not in p_dict:
        log.warning('calculator not given, auto guess by jobprefix')
        calculator = p_dict['jobPrefix'].lower().translate(str.maketrans('', '', digits))
    else:
        calculator = p_dict['calculator']
    print(calculator)
    if calculator not in calc_dict:
        raise Exception('Unknown calculator: {}'.format(calculator))
    return calc_dict[calculator](**p_dict)


def get_calculator(p_dict):
    if type(p_dict['jobPrefix']) is list:
        if 'calculator' not in p_dict:
            log.warning('calculator not given, auto guess by jobprefix')
            p_dict['calculator'] = [job.lower().translate(str.maketrans('', '', digits)) for job in p_dict['jobPrefix']]
        calcs = []
        for i, job in enumerate(p_dict['jobPrefix']):
            p_dict_ = deepcopy(p_dict)
            for key in need_convert:
                if key in p_dict_:
                    if type(p_dict_[key]) is list:
                        assert len(p_dict_[key]) == len(p_dict['jobPrefix']), '{} and jobPrefix length do not match'.format(key)
                        p_dict_[key] = p_dict_[key][i]
            calcs.append(get_one_calculator(p_dict_))
        return AdjointCalculator(calcs)
    return get_one_calculator(p_dict)
