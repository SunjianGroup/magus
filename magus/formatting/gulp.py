import numpy as np
from ase.atoms import Atoms
import re
from ase.units import GPa, eV, Ang
# TODO: 0d, 1d, 2d...


def dump_gulp(atoms, filename, shell=None, mode='w'):
    if shell is not None:
        assert isinstance(shell, list), "shell should be a list!"
    s = "cell\n"
    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    s += "%g %g %g %g %g %g\n" %(a, b, c, alpha, beta, gamma)
    s += "fractional\n"
    # core
    for atom in atoms:
        s += "%s core %.6f %.6f %.6f \n" %(atom.symbol, atom.a, atom.b, atom.c)
    # shell
    for atom in atoms:
        if shell is not None and atom.symbol in shell:
            s += "%s shel %.6f %.6f %.6f \n" %(atom.symbol, atom.a, atom.b, atom.c)
    with open(filename, mode) as f:
        f.write(s)


def load_gulp(filename):
    pv = 0
    forces, stress = [], []
    i = 0
    with open(filename) as f:
        lines = f.readlines()
    while i < len(lines):
        if 'Pressure*volume' in lines[i]:
            pv = float(lines[i].split()[2])
        if 'Total lattice' in lines[i] and 'eV' in lines[i]:
            energy = float(lines[i].split()[4]) - pv
        if 'Final internal derivatives' in lines[i]:
            i += 6
            while '------' not in lines[i]:
                forces.append([-float(f) * eV / Ang for f in lines[i].split()[3:6]])
                i += 1
            forces = np.array(forces)

        # coordinate format example:
        #   Final fractional coordinates of atoms :    
        #                                                                                            
        # --------------------------------------------------------------------------------
        #    No.  Atomic        x           y          z          Radius
        #         Label       (Frac)      (Frac)     (Frac)       (Angs) 
        # --------------------------------------------------------------------------------
        #      1  O     c     0.509638    0.713272    0.160793    0.000000
        #      2  O     s     0.509638    0.713272    0.160793    0.000000
        # --------------------------------------------------------------------------------
        if 'coordinates of atoms' in lines[i]:
            positions, symbols = [], []
            scaled = 'fractional' in lines[i]
            i += 6
            while '------' not in lines[i]:
                line = lines[i].split()
                i += 1
                if line[2] == 's':
                    continue
                positions.append([float(p) * Ang for p in line[3:6]])
                symbols.append(line[1])
            positions = np.array(positions)
        if 'Final cell parameters and derivatives' in lines[i]:
            i += 3
            for _ in range(6):
                stress.append(lines[i].split()[4])
                i += 1
        if 'Cartesian lattice vectors' in lines[i]:
            # if set conv, first lattice will be read
            cell = []
            i += 2
            for _ in range(3):
                cell.append([float(c) for c in lines[i].split()])
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
