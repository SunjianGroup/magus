import numpy as np
import os, subprocess, shutil, logging, copy, sys, yaml, traceback
from ase.atoms import Atoms
import re
from ase.units import GPa, eV, Ang, Ry, Bohr
from ase.io import read

def dump_vaspc(atoms, vaspSetup,  mode='w'):
    kmesh=0.03
    a,b,c=atoms.cell[0],atoms.cell[1],atoms.cell[2]
    a0, b0, c0, alpha, beta, gamma = atoms.cell.cellpar()
    symbols = list(set(atoms.get_chemical_symbols()))
    masses = list(set(atoms.get_masses()))
    dimc=vaspSetup['structure_type']

    with open('KPOINTS', mode) as f:
        f.write("manual"+'\n'+'0'+'\n'+"Monkhorst-Pack"+'\n')
        if dimc == 'confined_2d':
            f.write("%d %d 1 0 0 0\n" %(1/(a0*kmesh),1/(b0*kmesh)))
        elif dimc == 'confined_1d':
            f.write("1 1 %d 0 0 0\n" %(1/(c0*kmesh)))
        f.write("0 0 0")

    with open('OPTCELL','w') as f:
        if dimc == 'confined_2d':
            f.write("111")
        elif dimc == 'confined_1d':
            f.write("001")
