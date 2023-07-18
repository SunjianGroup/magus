import numpy as np
import os, subprocess, shutil, logging, copy, sys, yaml, traceback
from ase.atoms import Atoms
import re
from ase.units import GPa, eV, Ang, Ry, Bohr
from ase.io import read
# TODO: 0d, 1d, 2d...

def dump_espresso(atoms, espressoSetup, filename, mode='w'):
    pressure = espressoSetup['pressure']
    kmesh = espressoSetup['kmesh']
    pseudopotentials = espressoSetup['pp_setup']
    a,b,c=atoms.cell[0],atoms.cell[1],atoms.cell[2]
    a0, b0, c0, alpha, beta, gamma = atoms.cell.cellpar()
    symbols = list(set(atoms.get_chemical_symbols()))
    masses = list(set(atoms.get_masses()))
    with open(filename, mode) as f:
        f.write(" ntyp = %d \n" %(len(symbols)))
        f.write(" nat = %d \n/ \n" %(len(atoms)))
        f.write("&ELECTRONS \n conv_thr = 1.0E-10, \n mixing_beta = 0.4,\n/\n&IONS\n/\n&CELL\n")
        f.write(" cell_dynamics = 'bfgs' \n press = %f \n/\n" %(pressure))
        f.write('ATOMIC_SPECIES\n') 
        for i,symbol in enumerate(symbols):
            f.write("%s %.2f %s \n" %(symbol,masses[i],pseudopotentials[symbol]))
        f.write('\nK_POINTS automatic\n')
        f.write("%d %d %d 0 0 0\n\n" %(1/(a0*kmesh),1/(b0*kmesh),1/(c0*kmesh)))
        f.write('CELL_PARAMETERS (angstrom)\n')
        f.write("%.6f %.6f %.6f\n%.6f %.6f %.6f\n%.6f %.6f %.6f\n" %(a[0],a[1],a[2],b[0],b[1],b[2],c[0],c[1],c[2]))
        f.write('\nATOMIC_POSITIONS (crystal)\n')
        for atom in atoms:
            f.write("%s %.6f %.6f %.6f\n" %(atom.symbol, atom.a, atom.b, atom.c))
        f.write('End final coordinates\n')
#dump_espresso(a,espresso_setup,'struct')

def load_espresso(filename):
    pv = 0
    forces, stress = [], []
    i = 0
    with open(filename) as f:
        lines = f.readlines()
    while i < len(lines):
        if 'Begin final coordinates' in lines[i]:
            # if set conv, first lattice will be read
            cell,symbols,positions = [], [], []
            i += 5
            for _ in range(3):
                cell.append([float(c) for c in lines[i].split()])
                i += 1
            i+=1
            scaled = 'crystal' in lines[i]
            i+=1
            while 'End final coordinates' not in lines[i]:
                symbols.append(lines[i].split()[0])
                positions.append([float(p) * Ang for p in lines[i].split()[1:4]]) 
                i+=1
            positions = np.array(positions)
            while "!    total energy" not in lines[i]:
                i+=1
            energy=float(lines[i].split()[4]) * Ry
            while 'force =' not in lines[i]:
                i+=1
            while 'force =' in lines[i]:
                forces.append([float(f) * Ry / Bohr for f in lines[i].split()[6:9]])
                i+=1
            i+=7
            for si in range(3):
                stress.append(float(lines[i].split()[si]) *Ry/ (Bohr**3))
                i += 1
        i += 1
    atoms = Atoms(cell=cell, positions=positions, pbc=True)
    if scaled:
        atoms.set_scaled_positions(positions)
    atoms.set_chemical_symbols(symbols)
    atoms.info['energy'] = energy
    atoms.info['forces'] = forces
    atoms.info['stress'] = stress
    return atoms
