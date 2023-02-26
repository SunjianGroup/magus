import importlib, numpy
from magus.parameters import magusParameters
from magus.helpinfo import options as _help_options_

def parmhelp(*args, input_file=None, all = False, type = "non-set", show_avail = False, **kwargs):
    
    for name in _help_options_:
        if (kwargs[name] is True) or (all is True) or type == 'non-set' or (show_avail is True):
            f = getattr(importlib.import_module("magus.helpinfo"), name)
            f(input_file=input_file, all = all, type = type, show_avail = show_avail)