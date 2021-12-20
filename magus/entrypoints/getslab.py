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
from magus.utils import sort_elements

def getslab(filename = 'Ref/layerslices.traj', slabfile = 'slab.vasp', *args, **kwargs):
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s   %(message)s",datefmt='%H:%M:%S')
    m = magusParameters('input.yaml')
    g = ReconstructGenerator(m.parameters)

    pop = ase.io.read(filename, index = ':', format = 'traj')
    ind, rcs = RcsInd(m.parameters), None
    #rcs-magus
    if len(pop) == 3:
        rcs = ind(pop[2])
        rcs.layerslices = pop
        #rcs = rcs.addextralayer('relaxable')
        #rcs = rcs.addextralayer('bulk')
        #rcs = rcs.addvacuum(add = 1)
        rcs = rcs.addbulk_relaxable_vacuum()
    #ads-magus
    elif len(pop) == 2:
        rcs = ind(pop[1])
        rcs.layerslices = pop
        rcs = rcs.addextralayer('bulk', add = 1)

    if not slabfile is None:
        ase.io.write(slabfile, rcs, format = 'vasp',vasp5=True,direct = True)
    else:
        return rcs