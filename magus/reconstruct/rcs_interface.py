"""****************************************************************
THIS IS INTERFACE FILE TO MAGUS. 
Supported structure types: 
    (i) Surface reconstructions
    (ii) Clusters *stand-alone and **on surface
    (iii) Interfaces between two bulk structures
****************************************************************""" 
import logging

log = logging.getLogger(__name__)

rcs_type_list = ['surface', 'cluster', 'adclus', 'interface']

def rcs_interface(rcs_magus_parameters):

    setattr(rcs_magus_parameters, "RandomGenerator_", rcs_random_generator(rcs_magus_parameters.p_dict))
    setattr(rcs_magus_parameters, "NextPopGenerator_", rcs_ga_generator(rcs_magus_parameters.p_dict))
    set_rcs_population(rcs_magus_parameters)
    


"""**********************************************
#1. Change init random population generator.
**********************************************"""
from .generator import SurfaceGenerator, ClusterSPGGenerator
def rcs_random_generator(p_dict): 
    if p_dict['structureType'] == 'surface':
        return SurfaceGenerator(**p_dict)
    elif p_dict['structureType'] == 'cluster' or 'adclus':
        return ClusterSPGGenerator(**p_dict)


"""**********************************************
#2. Change GA population generator. Operators are changed.
**********************************************"""
from ..generators import  _cal_op_prob_,op_dict, GAGenerator, AutoOPRatio
from .ga import rcs_op_dict, rcs_op_list


def rcs_ga_generator(p_dict):

    operators = get_rcs_op(p_dict)
    if 'OffspringCreator' in p_dict:
        operators.update(p_dict['OffspringCreator'])

    op_dict.update(rcs_op_dict)
    
    op_list, op_prob = _cal_op_prob_(operators, op_dict)

    if p_dict['autoOpRatio']:
        generator =  AutoOPRatio(op_list, op_prob, **p_dict)
    else:
        generator = GAGenerator(op_list, op_prob, **p_dict)

    for i, op in enumerate(generator.op_list):
        if op.__class__ not in rcs_op_list:
            if hasattr(op, 'mutate'):
                log.info("set method 'mutate_bulk' to 'mutate' of {}".format(op.__class__.__name__))
                setattr(generator.op_list[i], 'mutate', generator.op_list[i].mutate_bulk)
            elif hasattr(op, 'cross'):
                log.info("set method 'cross_bulk' to 'cross' of {}".format(op.__class__.__name__))
                setattr(generator.op_list[i], 'cross', generator.op_list[i].cross_bulk)

    return generator


from ..operations import get_default_op
def get_rcs_op(p_dict):
    #DONE 'cutandsplice', 'slip', 'lattice', 'ripple', 'rattle'
    operators = get_default_op(p_dict)
    
    if p_dict['structureType'] == 'surface':
        del operators['lattice']
        if len(p_dict['symbols']) > 1:
            operators['formula'] = {}
        
        #operators['lyrslip'] = {}
        #operators['lyrsym'] = {}
        #operators['shell'] = {}
        
    if p_dict['structureType'] == 'cluster' or 'adclus':
        del operators['slip']
        #operators['soft'] = {}
        
        #operators['shell'], operators['clusym'] = {}, {}
        
    return operators


"""**********************************************
#3. Change Population type, including individual type and fitness_calculator.
**********************************************"""

from .individuals import RcsPopulation
import copy
def set_rcs_population(parameters):
    p_dict = copy.deepcopy(parameters.p_dict)
    p_dict['atoms_generator'] = parameters.RandomGenerator
    p_dict['units'] = parameters.RandomGenerator.units
    parameters.Population_ = RcsPopulation
    parameters.Population_.set_parameters(**p_dict)

"""**********************************************
#4. Change entrypoints functions. 
**********************************************"""