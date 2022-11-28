# calculators in localopt should be moved here
from .individuals import *
from .populations import *
import logging


log = logging.getLogger(__name__)


def get_population(p_dict):
    pop_dict = {'fix': FixPopulation, 'var': VarPopulation}
    Pop = pop_dict[p_dict['formulaType']]
    Pop.set_parameters(**p_dict)
    return Pop
