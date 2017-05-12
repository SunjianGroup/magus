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

def read_yaml(filename):
 
    readList = yaml.load(open(filename))
 
    images = list()
    for struct, info in readList:
        image = Atoms(**struct)
        image.info = info
        images.append(image)
 
    return images

filename = sys.argv[1]
symprec = 0.5
images = read_yaml(filename)

names = locals()
showList = [
'symmetry', 
'enthalpy', 
'predictE', 
'parentE', 
#'symbols', 
#'formula', 
#'gap', 
#'volume',
#'Origin',
'utilVal',
'sigma',
'relaxD',
]
allRes = []

for i, at in enumerate(images):
    posname = "POSCAR_%s" %(i)
    ase.io.write(posname, at, direct = True, vasp5 = True)
    symmetry = spg.get_spacegroup(at, symprec)
    # symmetry = os.popen(
    #     "phonopy --symmetry --tolerance %s -c POSCAR_%s | grep space_group_type"
    #     % (symprec, i)).readlines()[0].split()[1]
    volume = at.get_volume()
    volume = round(volume, 3)

    oneRes = list()

    for feature in showList:
        if feature in at.info.keys():
            oneRes.append(at.info[feature])
        elif feature is 'volume' or feature is 'symmetry':
            oneRes.append(names[feature])
        else:
            oneRes.append(None)     

    allRes.append(oneRes)

table = pd.DataFrame(allRes, columns=showList)
print(table.sort_values('enthalpy', axis=0))

    # outD = dict()
    # for feature in showList:
    #     outD[feature] = names[feature]

    # pprint.pprint(outD)
    # pprint.pprint((symmetry, gap, enthalpy, volume, symbols, formula, parentE))

    # output = "%s\tgap:%s\tenthalpy:%s\tvolume:%s\tsymbols:%s\tformula:%s\tparentE:%s" %(symmetry, gap, enthalpy, volume, symbols, formula, parentE)
    # print(output)

