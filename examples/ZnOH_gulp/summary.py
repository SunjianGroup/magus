#!/usr/bin/env python
import sys
import spglib as spg
import re
import os
import pprint
import yaml
import pandas as pd
import ase.io
from ase import Atoms
import numpy as np


savePri = 1
saveStd = 0

pd.options.display.max_rows = 200

filename = sys.argv[1]
symprec = 0.1
# images = read_yaml(filename)
images = ase.io.read(filename, format='traj', index=':')

names = locals()
showList = [
'symmetry',
#'enthalpy',
'ehull',
#'predictE',
'parentE',
#'symbols',
'formula',
#'dominators',
#'gap',
#'volume',
#'Origin',
#'utilVal',
#'sigma',
#'relaxD',
#'fullSym',
#'lengths',
#'angles',
]
allRes = []

for i, at in enumerate(images):
    posname = "POSCAR_%s.vasp" %(i)
    ase.io.write(posname, at, direct = True, vasp5 = True)
    symmetry = spg.get_spacegroup(at, symprec)
    cellpar = np.round(at.get_cell_lengths_and_angles(), 2)
    cellpar = cellpar.tolist()
    lengths = cellpar[:3]
    angles = cellpar[3:]

    if savePri:
        priInfo = spg.find_primitive(at, symprec)
        if priInfo:
            lattice, scaled_positions, numbers = priInfo
            priAt = Atoms(cell=lattice, scaled_positions=scaled_positions, numbers=numbers)
            ase.io.write("pri_{}.vasp".format(i), priAt, direct=1, vasp5=1)
        else:
            ase.io.write("pri_{}.vasp".format(i), at, direct=1, vasp5=1)

    if saveStd:
        stdInfo = spg.standardize_cell(at, symprec=symprec)
        if stdInfo:
            lattice, scaled_positions, numbers = stdInfo
            stdInfo = Atoms(cell=lattice, scaled_positions=scaled_positions, numbers=numbers)
            ase.io.write("std_{}.vasp".format(i), stdInfo, direct=1, vasp5=1)
        else:
            ase.io.write("std_{}.vasp".format(i), at, direct=1, vasp5=1)



    volume = at.get_volume()
    volume = round(volume, 3)

    fullSym = at.get_chemical_formula()

    oneRes = list()

    outFeatures = ['volume', 'symmetry', 'angles', 'lengths', 'fullSym']

    for feature in showList:
        if feature in at.info.keys():
            oneRes.append(at.info[feature])
        elif feature in outFeatures:
            oneRes.append(names[feature])
        else:
            oneRes.append(None)



    allRes.append(oneRes)

table = pd.DataFrame(allRes, columns=showList)
print(table.sort_values('ehull', axis=0))
#print(table.sort_values('enthalpy', axis=0))

    # outD = dict()
    # for feature in showList:
    #     outD[feature] = names[feature]

    # pprint.pprint(outD)
    # pprint.pprint((symmetry, gap, enthalpy, volume, symbols, formula, parentE))

    # output = "%s\tgap:%s\tenthalpy:%s\tvolume:%s\tsymbols:%s\tformula:%s\tparentE:%s" %(symmetry, gap, enthalpy, volume, symbols, formula, parentE)
    # print(output)

