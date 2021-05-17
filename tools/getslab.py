#!/usr/bin/env python
import sys
import re
import os
import pprint
import yaml
import ase.io
from ase import Atoms
import numpy as np
import logging

from magus.population import RcsInd
import ase.io
from magus.parameters import magusParameters
from magus.initstruct import ReconstructGenerator, Generator
import spglib
import ase.io
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s   %(message)s",datefmt='%H:%M:%S')
m = magusParameters('input.yaml')
g = ReconstructGenerator(m.parameters)

filename = sys.argv[1] if len(sys.argv)>1 else 'Ref/layerslices.traj'
pop = ase.io.read(filename, index = ':', format = 'traj')
ind = RcsInd(m.parameters)
pop[2].info['size']=[1,1]
rcs = ind(pop[2])
rcs.layerslices = pop
#rcs = rcs.addextralayer('relaxable')
#rcs = rcs.addextralayer('bulk')
#rcs = rcs.addvacuum(add = 1)
rcs = rcs.addbulk_relaxable_vacuum()
ase.io.write('slab.vasp',rcs, format = 'vasp',vasp5=True,direct = True)
