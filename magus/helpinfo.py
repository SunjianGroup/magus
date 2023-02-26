"""****************************************************************
PRINT PARAMETER INFORMATION IN CLASSES.
****************************************************************""" 
from magus.entrypoints.checkpack import check_list
from magus.utils import load_plugins
from pathlib import Path
import importlib, numpy

from magus.operations import op_dict
from magus.reconstruct.ga import rcs_op_dict
from magus.parameters import magusParameters

options = ['parameters','generators', 'individuals', 'populations', 'operations'] + ['calculators']#list(check_list.keys())

def output_helpinfo(instance, is_instance = False):
    if is_instance:
        static_info =instance.__class__.export_all_help_info()
    else:
        static_info = instance.export_all_help_info()
    
    """e.g. static_info = list([
        (<class 'magus.reconstruct.individuals.Surface'>, {'default': {...}, 'requirement': {...}}), 
        (<class 'magus.populations.individuals.Individual'>, {'default': {...}, 'requirement': {...}})
    ]
    """
    print("parameter information for {}".format(instance))
    info = {}
    static_info = [si[1] for si in static_info]         #delete class name
    static_info.reverse()       #from child class to base class
    for si in static_info:
        for key in si:              #'default', 'requirement'
            if key in info:
                info[key].update(si[key])
            else:
                info[key] = si[key]            

    static_info = {}
    for key in info:        #'default', 'requirement'
        parmset = info[key]
        if len(parmset) > 0:       
            static_info[key] = {parm:parmset[parm] for parm in sorted(parmset.keys())}

    for item in static_info:        #now static_info= {'default':{...}, 'requirement': {...}}
        print("+++++\t{} parameters\t+++++".format(item))
        for key in static_info[item]:
            note = static_info[item][key]
            #(!) skip 'inner' parms to avoid misleading
            if note[0].find("inner") == 0:
                continue
            help = note[0]
            #add default value (if there is)
            if len(note) > 1:
                help += "\n                  default value: {}".format(note[1])
            #add real time value (if we have instance)
            if is_instance:
                #try:
                real_time_value = getattr(instance, key)
                help += "\n                  real-time value: {}".format(real_time_value)
                #except:
                #    pass
            
            print("{}: {}".format(key.ljust(15, ' '), help))
    print("----------------------------------------------------------------")

class basehelp:
    candidates = {}

    def show_avail(self):
        print("+ Available types for module **{}** include: ".format(self.__class__.__name__))
        for key in self.candidates.keys():
            print("\t++ {} {}".format(key, self.__class__.__name__))
    
    def get_instance(self, input_file = None):
        raise NotImplementedError
        
    def __init__(self, input_file=None, all = False, type = 'non-set', show_avail = False,**kwargs):
        if show_avail:
            self.show_avail()
            return
        
        for key in self.candidates.keys():
            if key == type or all or type == 'non-set':
                if input_file is None:
                    ins = getattr(importlib.import_module(self.paths[key]), self.candidates[key])
                else:
                    ins = self.get_instance(input_file)
                
                output_helpinfo(ins, is_instance=(not input_file is None))
                       
class generators(basehelp):
    candidates = {key:"{}SPGGenerator".format(key) for key in ["Molecule", "Layer", "Cluster"]}
    candidates.update({"Bulk": "SPGGenerator", "Surface": "SurfaceGenerator"})
    paths = {key: "magus.reconstruct.generator" for key in ["Cluster", "Surface"]}
    paths.update({key: "magus.generators.random" for key in ["Molecule", "Layer", "Bulk"]})

    def get_instance(self, input_file=None):
        m = magusParameters(input_file)
        return m.RandomGenerator

class individuals(basehelp):
    candidates = {key:key for key in ["Bulk", "Layer", "ConfinedBulk", "Surface", "Cluster", "AdClus"]}
    paths = {key: "magus.populations.individuals" for key in ["Bulk", "Layer", "ConfinedBulk"]}
    paths.update({key: "magus.reconstruct.individuals" for key in ["Surface", "Cluster", "AdClus"]})
    
    def get_instance(self, input_file=None):
        m = magusParameters(input_file)
        return m.Population.Ind(**m.p_dict)


class populations(basehelp):
    candidates = {key:"{}Population".format(key) for key in ["Fix", "Var"]}
    candidates.update({"Surface": "RcsPopulation"})
    paths = {key: "magus.populations.populations" for key in ["Fix", "Var"]}
    paths.update({"Surface": "magus.reconstruct.individuals"})

    def get_instance(self, input_file=None):
        m = magusParameters(input_file)
        return m.Population

class operations(basehelp):
    candidates = op_dict.copy()
    candidates.update(rcs_op_dict)

    def __init__(self, input_file=None, all = False, type = 'non-set', show_avail = False,**kwargs):
        if show_avail:
            self.show_avail()
            return
        for key in self.candidates.keys():
            if key == type or all or type == 'non-set':
                ins = self.candidates[key]
                if not input_file is None:
                    m = magusParameters(input_file)
                    ins = ins(**m.p_dict)
                
                output_helpinfo(ins, is_instance=(not input_file is None))


class parameters(basehelp):
    def __init__(self, input_file=None, all = False, type = 'non-set', show_avail = False,**kwargs):
        if show_avail:
            print("+ Available types for module **parameters** include: \n\t++ parameters")
            return
        
        ins = getattr(importlib.import_module("magus.parameters"), "magusParameters")     
        if not input_file is None:
            m = magusParameters(input_file)
            ins = ins(**m.p_dict)
                
        output_helpinfo(ins, is_instance=(not input_file is None))


class helpPLUGIN(basehelp): 
    def __init__(self, pack, input_file=None, all = False, type = 'non-set', show_avail = False,**kwargs):
        
        path = Path(__file__).parent.joinpath(pack, '__init__.py')
        load_plugins(path, 'magus.' + pack)
        self.candidates = {}
        for name in check_list[pack]:
            self.candidates.update(name.plugins)
        if show_avail:
            self.show_avail()
            return

        for key in self.candidates.keys():
            if key == type or all or type == 'non-set':
                ins = self.candidates[key]
                output_helpinfo(ins, is_instance=False)
                if not input_file is None:
                    print("+++++	real time values	+++++\n{}".format(self.get_instance(input_file)))
                    print("----------------------------------------------------------------")

                            
    def get_instance(self, input_file=None):
        m = magusParameters(input_file)
        return m.MainCalculator
        
class calculators(helpPLUGIN):
    def __init__(self, **kwargs):
        super().__init__("calculators", **kwargs)
"""
class comparators(helpPLUGIN):
    def __init__(self, **kwargs):
        super().__init__("comparators", **kwargs)
class fingerprints(helpPLUGIN):
    def __init__(self, **kwargs):
        super().__init__("fingerprints", **kwargs)
"""    