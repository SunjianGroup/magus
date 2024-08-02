import logging


#_applied_operations_ = list(op_dict.keys()) + list(rcs_op_dict.keys()) 

_applied_operations_ = [ 'cutandsplice', 'replaceball',
                                                  'soft', 'perm', 'lattice', 'ripple', 'slip', 'rotate', 'rattle', 'formula',
                                                  'sym', 'shell']

#   ``Magus mutate -s seed.traj --rattle

log = logging.getLogger(__name__)

def mutate(*args, input_file='input.yaml', seed_file = 'seed.traj', output_file='result.traj', number = 10, **kwargs):
    from magus.parameters import magusParameters
    import ase.io
    from magus.operations import op_dict
    
    m = magusParameters(input_file)
    ga = m.NextPopGenerator


    operators = {}
    set_parm = m.p_dict['OffspringCreator'] if 'OffspringCreator' in m.p_dict else {}

    for key in _applied_operations_:
        if kwargs[key]:
            parm = {}
            if key in set_parm:
                parm.update(set_parm[key])
            op = op_dict[key](**parm)
            operators[op] = {}

    ga.op_list = list(operators.keys())
    ga.op_prob = [1.0/len(ga.op_list)]*len(ga.op_list)
    pop = m.Population

    log.info("use ga generater: {}".format(ga))
    
    seed_pop = ase.io.read(seed_file, index = ':')
    for i, _ in enumerate(seed_pop):
        if not 'energy' in seed_pop[i].info:
            seed_pop[i].info['energy'] = 0
        if not 'enthalpy' in seed_pop[i].info:
            seed_pop[i].info['enthalpy'] = 0
        if not 'identity' in seed_pop[i].info:
            seed_pop[i].info['identity'] = 'init1-1'
            seed_pop[i].info['gen'] = 1

    
    seed_pop = pop(seed_pop, 'seedPop')
    seed_pop.gen = 0
    next_pop =ga.get_next_pop(seed_pop, int(number))
    log.info("generated {} individuals.".format(len(next_pop)))
    new_frames = []
    for atoms in next_pop:
        new_frames.append(atoms.for_calculate() if hasattr(atoms, "for_calculate") else atoms)

    ase.io.write(output_file, new_frames, format = 'traj')
