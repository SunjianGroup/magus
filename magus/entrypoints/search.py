import logging
from magus.logger import set_logger
from magus.parameters import magusParameters
from magus.search.search import Magus
from magus.search.search_mtp import MLMagus


log = logging.getLogger(__name__)


def search(*args, input_file='input.yaml', log_level='INFO', 
           use_ml=False, restart=False, **kwargs):
    set_logger(level=log_level, log_path='log.txt')
    log.info(" Initialize ".center(40, "="))
    parameters = magusParameters('input.yaml')
    atoms_generator = parameters.get_AtomsGenerator()
    pop_generator = parameters.get_PopGenerator()
    main_calculator = parameters.get_MainCalculator()
    Population = parameters.get_Population()
    if use_ml:
        ml_calculator = parameters.get_MLCalculator()
        m = MLMagus(
            parameters=parameters.parameters, 
            atoms_generator=atoms_generator, 
            pop_generator=pop_generator,
            main_calculator=main_calculator, 
            ml_calculator=ml_calculator,
            Population=Population,
            restart=restart)
    else:
        m = Magus(
            parameters=parameters.parameters, 
            atoms_generator=atoms_generator, 
            pop_generator=pop_generator,
            main_calculator=main_calculator, 
            Population=Population,
            restart=restart)
    m.run()
