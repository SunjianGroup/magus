from magus.parameters import magusParameters
import ase.io
import logging

_applied_operations_ = ['cutandsplice', 'replaceball',
                                  'soft', 'perm', 'lattice', 'ripple', 'slip', 'rotate', 'rattle', 'formula',
                                  'lyrslip', 'lyrsym', 'shell', 'clusym']

def mutate(*args, input_file='input.yaml', seed_file = 'seed.traj', output_file='result', **kwargs):
    
    log = logging.getLogger(__name__)

    op_nums = {}
    for key in _applied_operations_:
        op_nums[key+'Num'] = kwargs[key] if key in kwargs else 0

    m = magusParameters(input_file)
    if not hasattr(m.parameters, 'OffspringCreator'):
        setattr(m.parameters, 'OffspringCreator', {})
    inputparm = getattr(m.parameters, 'OffspringCreator') 
    for key in list(op_nums.keys()):
        if key not in inputparm:
            m.parameters.OffspringCreator[key] =  op_nums[key]
    print(m.parameters.OffspringCreator)
    PopGenerator = m.get_PopGenerator()
    Population = m.get_Population()
    seed_pop = ase.io.read(seed_file, index = ':')
    for i, _ in enumerate(seed_pop):
        seed_pop[i].info['energy'] = 0
        seed_pop[i].info['enthalpy'] = 0
    seed_pop = Population(seed_pop, 'seedPop')
    seed_pop.gen = 0
    next_pop = PopGenerator.generate(seed_pop, m.parameters.saveGood)
    log.debug("generated {} individuals.".format(len(next_pop)))
    next_pop.save(filename = output_file[:-1], gen = output_file[-1], savedir = '.')