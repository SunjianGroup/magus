import logging
from magus.parameters import magusParameters
from magus.search.search import Magus
from magus.search.search_ml import MLMagus


log = logging.getLogger(__name__)


def search(*args, input_file='input.yaml',
           use_ml=False, restart=False, **kwargs):
    log.info(" Initialize ".center(40, "="))
    parameters = magusParameters(input_file)
    if use_ml:
        m = MLMagus(parameters, restart=restart)
    else:
        m = Magus(parameters, restart=restart)
    m.run()
