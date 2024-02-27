import logging
from magus.parameters import magusParameters
from magus.search.search import Magus
from magus.search.search_ml import MLMagus
from magus.search.search_pa import PaMagus

try:
    from magus.reconstruct.fly_interface import interface_smfr
    GLOBAL_use_smfr = True
except:
    import traceback, warnings
    warnings.warn("Failed to load module for on-the-fly spacegroup miner and fragment reorganizer:\n {}".format(traceback.format_exc()) +
                  "\nThis warning above can be ignored if the mentioned functions are not needed, elsewise should be fixed.\n" )
    GLOBAL_use_smfr = False


log = logging.getLogger(__name__)


def search(*args, input_file='input.yaml',
           use_ml=False, use_parallel = False, restart=False, use_smfr = False, **kwargs):
    log.info(" Initialize ".center(40, "="))
    parameters = magusParameters(input_file)
    if use_ml:
        m = MLMagus(parameters, restart=restart)
    elif use_parallel:
        m = PaMagus(parameters, restart = restart)
    else:
        m = Magus(parameters, restart=restart)
    
    if use_smfr and GLOBAL_use_smfr:
        interface_smfr(m, restart = restart)

    m.run()
