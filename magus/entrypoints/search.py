import logging
from magus.parameters import magusParameters
from magus.search.search import Magus
from magus.search.search_ml import MLMagus

log = logging.getLogger(__name__)


def search(*args, input_file='input.yaml',
           use_ml=False, use_parallel = False, restart=False, use_smfr = False, **kwargs):
    log.info(" Initialize ".center(40, "="))
    parameters = magusParameters(input_file)
    if use_ml:
        m = MLMagus(parameters, restart=restart)
    elif use_parallel:
        from magus.search.search_pa import PaMagus
        m = PaMagus(parameters, restart = restart)
    else:
        m = Magus(parameters, restart=restart)
    
    if use_smfr:
        from magus.reconstruct.fly_interface import interface_smfr
        interface_smfr(m, restart = restart)

    m.run()
