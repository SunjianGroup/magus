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
from .generator import SurfaceGenerator, ClusterSPGGenerator, InterfaceGenerator
def rcs_random_generator(p_dict): 
    if p_dict['structureType'] == 'surface':
        return SurfaceGenerator(**p_dict)
    if p_dict['structureType'] == 'interface':
        return InterfaceGenerator(**p_dict)
    elif p_dict['structureType'] == 'cluster' or p_dict['structureType'] == 'adclus':
        return ClusterSPGGenerator(**p_dict)


"""**********************************************
#2. Change GA population generator. Operators are changed.
**********************************************"""
from ..generators import  _cal_op_prob_,op_dict, GAGenerator, AutoOPRatio
from .ga import rcs_op_dict, rcs_op_list, GA_interface


def rcs_cross(cls, ind1, ind2):
    newind = cls.func(ind1, ind2)
    if 'size' in ind1.info and not (newind is None):
        newind.info['size'] = ind1.info['size']
    return newind

def rcs_mutate(cls, ind):
    newind = cls.func(ind)
    if 'size' in ind.info and not (newind is None):
        newind.info['size'] = ind.info['size']
    return newind

def rcs_get_new_ind(cls, ind):
    newind = cls.ori_get_new_ind(ind)
    if newind is None:
        return newind
    if hasattr(ind, 'info'):    #mutate
        if 'size' in ind.info:
            newind.info['size'] = ind.info['size']
    else:                               #cross
        if 'size' in ind[0].info:
            newind.info['size'] = ind[0].info['size']
    return newind


def rcs_ga_generator(p_dict):

    operators = get_rcs_op(p_dict)
    if 'OffspringCreator' in p_dict:
        operators.update(p_dict['OffspringCreator'])
    
    op_dict.update(rcs_op_dict)
    
    for name in op_dict.keys():
        op = op_dict[name]

        if hasattr(op, 'ver_rcs'):
            continue
        
        GA_interface()
        setattr(op, 'ver_rcs', True)

        setattr(op, 'ori_get_new_ind', op.get_new_individual)
        setattr(op, "get_new_individual", rcs_get_new_ind)

        s_t = p_dict['structureType']

        if hasattr(op, 'mutate'):
            if hasattr(op, "mutate_{}".format(s_t)):
                func = getattr(op, "mutate_{}".format(s_t))
            else:
                func = getattr(op, "mutate_bulk")
            setattr(op, "func", func)
            setattr(op, 'mutate', rcs_mutate)
        
        elif hasattr(op, 'cross'):
            if hasattr(op, "cross_{}".format(s_t)):
                func = getattr(op, "cross_{}".format(s_t))
            else:
                func = getattr(op, "cross_bulk")
            setattr(op, "func", func)                
            setattr(op, 'cross', rcs_cross)
        
        op_dict[name] = op

    
    op_list, op_prob = _cal_op_prob_(operators, op_dict)

    if p_dict['autoOpRatio']:
        generator =  AutoOPRatio(op_list, op_prob, **p_dict)
    else:
        generator = GAGenerator(op_list, op_prob, **p_dict)

    return generator


from ..operations import get_default_op
def get_rcs_op(p_dict):
    #DONE 'cutandsplice', 'slip', 'lattice', 'ripple', 'rattle'
    operators = get_default_op(p_dict)
    
    if p_dict['structureType'] == 'surface':
        del operators['lattice']
        
    if p_dict['formulaType'] == 'var':
        operators['formula'] = {}

        #operators['slip'] = {}
        #operators['sym'] = {}
        #operators['shell'] = {}
        
    if p_dict['structureType'] == 'cluster' or p_dict['structureType'] == 'adclus':
        del operators['slip']
        #operators['soft'] = {}
        
        #operators['shell'], operators['sym'] = {}, {}
        
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

"""**********************************************
#5. Prepare function: calculate ref slab energy. 
**********************************************"""