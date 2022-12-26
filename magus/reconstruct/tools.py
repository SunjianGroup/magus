import ase.io
import logging
from ..parameters import magusParameters


def getslab(filename = 'Ref/layerslices.traj', slabfile = 'slab.vasp', *args, **kwargs):
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s   %(message)s",datefmt='%H:%M:%S')

    m = magusParameters('input.yaml')
    
    pop = ase.io.read(filename, index = ':', format = 'traj')
    ind, rcs = (m.Population).Ind, None
    #rcs-magus
    if len(pop) == 3:
        rcs = ind(pop[2])
        rcs.buffer = True
        rcs.bulk_layer, rcs.buffer_layer = pop[0], pop[1]
        
    #ads-magus
    elif len(pop) == 2:
        rcs = ind(pop[1])
        rcs.buffer = False
        rcs.bulk_layer = pop[0]

    if not slabfile is None:
        ase.io.write(slabfile, rcs.for_calculate(), format = 'vasp',vasp5=True,direct = True)
    else:
        return rcs.for_calculate()

def analyze(*args, **kwargs):
    return