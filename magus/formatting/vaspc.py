import numpy as np
import os, subprocess, shutil, logging, copy, sys, yaml, traceback
from ase.atoms import Atoms
import re
from ase.units import GPa, eV, Ang, Ry, Bohr
from ase.io import read,write

def load_incar():
    with open('INCAR') as fin:
        incar=fin.readlines()
    new_incar = []
    for i in incar:
        if ('LJ_' not in  i) and ('PSTRESS' not in i):
            new_incar.append(i)
    return new_incar


def dump_vaspc(atoms, vaspSetup, vaspIncar, mode='w'):
    kmesh=0.03
    a,b,c=atoms.cell[0],atoms.cell[1],atoms.cell[2]
    a0, b0, c0, alpha, beta, gamma = atoms.cell.cellpar()
 #   symbols = list(set(atoms.get_chemical_symbols()))
    masses = list(set(atoms.get_masses()))
    pseudopotentials = vaspSetup['pp_setup']
    LJD = vaspSetup['LJD']
    LJA = vaspSetup['LJA']
    LJW = vaspSetup['LJW']
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
            f.write("110")
        elif dimc == 'confined_1d':
            f.write("001")
    write('POSCAR',atoms)

    with open('POSCAR','r') as f:
        symbols = f.readlines()[0].split()

    if os.path.exists('POTCAR'):
        os.remove('POTCAR')

    d,a,w = ' LJ_D = ',' LJ_A = ',' LJ_W = '
    for i,symbol in enumerate(symbols):
        os.system('cat %s%s%s/POTCAR >>POTCAR' %(vaspSetup['pp_dir'],symbol,pseudopotentials[symbol]))
        d = d+str(LJD[symbol])+' '
        a = a+str(LJA[symbol])+' '
        w = w+str(LJW[symbol])+' '

    with open('INCAR', 'w') as f:
        for fin in vaspIncar:
            f.write(fin)
        f.write(' PSTRESS = ' + str(10*float(vaspSetup['pressure']))+'\n')
        f.write(d+'\n')
        f.write(a+'\n')
        f.write(w+'\n')

