import numpy as np
from ase.atoms import Atoms
from magus.reconstruct import fixatoms
from ase.units import GPa

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
            try:
                atoms.info['forces'] = atoms.get_forces()
            except:
                pass
            
            info = ['AtomData: id type cartes_x cartes_y cartes_z']

            if 'forces' in atoms.info:
                info.append('fx fy fz')

            flags = np.array(['T']*len(atoms))
            if atoms.constraints:
                info.append('mvable')
                for constr in atoms.constraints:
                    flags[constr.index] = 'F'

            ret += ' '.join(info) +'\n'
            for i, atom in enumerate(atoms):
                s = ''
                for key in info:
                    if key == 'AtomData: id type cartes_x cartes_y cartes_z':
                        s += '{} {} {} {} {}'.format(i + 1, symbol_to_type[atom.symbol], *cartes[i])
                    elif key == 'fx fy fz':
                        s += ' {} {} {}'.format(*atoms.info['forces'][i])
                    elif key == 'mvable':
                        s += ' {}'.format(flags[i])
                ret += s+'\n'

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
                positions = []
                symbols = []

            if 'Size' in line:
                line = f.readline()
                natoms = int(line.split()[0])
            
            if 'Supercell' in line: 
                for i in range(3):
                    line = f.readline()
                    for j in range(3):
                        cell[i, j] = float(line.split()[j])

            if 'AtomData' in line:
                has_force = False
                has_constraints = False

                if 'fx' in line:
                    has_force = True
                    forces = []
                if 'mvable' in line:
                    has_constraints = True
                    c = []

                for _ in range(natoms):
                    line = f.readline()
                    fields = line.split()
                    symbols.append(type_to_symbol[int(fields[1])])
                    startindex = 2
                    if has_constraints:
                        startindex +=1
                        if fields[2] == 'F':
                            c.append(int(fields[0]) -1)

                    positions.append(list(map(float, fields[startindex: startindex+3])))
                    if has_force:
                        forces.append(list(map(float, fields[startindex+3: startindex+6])))

                atoms = Atoms(symbols=symbols, cell=cell, positions=positions, pbc=True)
                if has_force:
                    atoms.info['forces'] = np.array(forces)
                if has_constraints:
                    atoms.set_constraint(fixatoms(c))

            if 'Energy' in line:
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
