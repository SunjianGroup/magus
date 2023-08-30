import numpy as np
from ase.atoms import Atoms
from ase.units import GPa
from collections import defaultdict


def dump_cfg(frames, filename, symbol_to_type, mode='w'):
    with open(filename, mode) as f:
        for atoms in frames:
            ret = ''
            ret += 'BEGIN_CFG\n'
            ret += 'Size\n{}\n'.format(len(atoms))
            try:
                cell = atoms.get_cell()[:]
                ret += 'Supercell\n{} {} {}\n{} {} {}\n{} {} {}\n'.format(*cell[0], *cell[1], *cell[2])
            except:
                pass
            cartes = atoms.positions
            has_forces = False
            has_constraints = False
            try:
                atoms.info['forces'] = atoms.get_forces()
            except:
                pass
            fields = ['id', 'type', 'cartes_x', 'cartes_y', 'cartes_z']
            if 'forces' in atoms.info:
                fields.extend(['fx', 'fy', 'fz'])
                forces = atoms.info['forces']
                has_forces = True

            flags = np.array(['T']*len(atoms))
            if atoms.constraints:
                info.append('mvable')
                for constr in atoms.constraints:
                    flags[constr.index] = 'F'
                has_constraints = True

            ret += 'AtomData: ' + ' '.join(fields) + '\n'
            for i, atom in enumerate(atoms):
                atom_info = '{} {} {} {} {} '.format(i + 1, symbol_to_type[atom.symbol], *cartes[i])
                if has_forces:
                    atom_info += '{} {} {} '.format(*forces[i])
                if has_constraints:
                    atom_info += flags[i]
                ret += atom_info + '\n'
            try:
                atoms.info['energy'] = atoms.get_potential_energy()
            except:
                pass
            if 'energy' in atoms.info:
                ret += 'Energy\n{}\n'.format(atoms.info['energy'])
            try:
                atoms.info['stress'] = atoms.get_stress()
            except:
                pass
            if 'stress' in atoms.info:
                stress = atoms.info['stress'] * atoms.get_volume() * -1.
                ret += 'PlusStress: xx yy zz yz xz xy\n{} {} {} {} {} {}\n'.format(*stress)
            if 'identification' in atoms.info:
                ret += 'Feature identification {}\n'.format(atoms.info['identification'])
            ret += 'END_CFG\n'
            f.write(ret)


#TODO 
# different cell
# pbc
def load_cfg(filename, type_to_symbol):
    frames = []
    with open(filename) as f:
        line = 'chongchongchong!'
        while line:
            line = f.readline()
            if 'BEGIN_CFG' in line:
                cell = np.zeros((3, 3))

            if 'Size' in line:
                line = f.readline()
                natoms = int(line.split()[0])
                positions = np.zeros((natoms, 3))
                forces = np.zeros((natoms, 3))
                energies = np.zeros(natoms)
                symbols = ['X'] * natoms
                constraints = ['T'] * natoms

            if 'Supercell' in line: 
                for i in range(3):
                    line = f.readline()
                    for j in range(3):
                        cell[i, j] = float(line.split()[j])

            if 'AtomData' in line:
                d = defaultdict(int)
                for (i, x) in enumerate(line.split()[1:]):
                    d[x] = i

                for _ in range(natoms):
                    line = f.readline()
                    fields = line.split()
                    i = int(fields[d['id']]) - 1
                    symbols[i] = type_to_symbol[int(fields[d['type']])]
                    positions[i] = [float(fields[d[attr]]) for attr in ['cartes_x', 'cartes_y' ,'cartes_z']]
                    forces[i] = [float(fields[d[attr]]) for attr in ['fx', 'fy' ,'fz']]
                    energies[i] = float(fields[d['site_en']])

                atoms = Atoms(symbols=symbols, cell=cell, positions=positions, pbc=True)
                if d['fx'] != 0:
                    atoms.info['forces'] = forces
                if d['site_en'] != 0:
                    atoms.info['energies'] = energies

            if 'Energy' in line and 'Weight' not in line:
                line = f.readline()
                atoms.info['energy'] = float(line.split()[0])

            if 'PlusStress' in line:
                line = f.readline()
                plus_stress = np.array(list(map(float, line.split())))
                atoms.info['stress'] = -plus_stress / atoms.get_volume()
                atoms.info['pstress'] = atoms.info['stress'] / GPa

            if 'END_CFG' in line:
                frames.append(atoms)

            if 'identification' in line:
                atoms.info['identification'] = int(line.split()[2])

    return frames
