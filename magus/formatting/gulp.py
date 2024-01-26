#NOTE: Although more than 80% cases turning on 'use_spg_init' while opti nosymm can work,
# (The left 20% cases fails mainly because it cannot get a correct formula)
#This setting probably doesn't make special sense (?) since it is said that in this case "space" is only used for initialization.
#All in all, pls 'use_spg_init' with and only along with opti symm

import numpy as np
from ase.spacegroup import crystal, Spacegroup
from ase.build import make_supercell
from ase.constraints import FixAtoms  
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.geometry.cell import cellpar_to_cell, cell_to_cellpar
from ase.units import GPa, eV, Ang
import spglib
from ase import Atoms
import math
# TODO: 0d, 1d, 2d...
from ..utils import multiply_cell

def dump_gulp(atoms0, filename, shell=None, mode='w', use_spg_init=False):
    atoms = atoms0.copy()
    if use_spg_init:
        symmetry_dataset = spglib.get_symmetry_dataset(atoms,0.1)
        spg = symmetry_dataset['number']
        # sometimes structures is like cell *(2,1,1), and cannot get it back just by crystal(Spacegroup, basis),
        # in which case the number of atoms is scaled by 2.
        # Method I: just put it in with 'space 1'.
        # Method II: relax it with the small cell and *(2,1,1) after relaxation. *
            
        std_para = spglib.standardize_cell(atoms, symprec=0.1, to_primitive=False)
        std_atoms = Atoms(cell=std_para[0], scaled_positions=std_para[1], numbers=std_para[2])

        # idk how to map indexes from atoms to std_atoms. In our program <1.7.0>, I use a temp way to load lower (for surface) and upper (for interfaces)
        # delineations of constraints. It won't fit other systems, and may induce mistakes when dealing with cells which have different length c between atoms and std_atoms. 
        if atoms.constraints:
            cst_atoms = atoms.copy()
            length_c = cst_atoms.cell.cellpar()[2]
            length_c1 =std_atoms.cell.cellpar()[2]
            while len(cst_atoms.constraints)>0:
                del cst_atoms[cst_atoms.constraints[0].index[0]]
            sp = cst_atoms.get_scaled_positions()[:,2]
            upper_limit = (np.max(sp) + 0.05 / length_c) /length_c * length_c1
            lower_limit = (np.min(sp) - 0.05 / length_c) /length_c * length_c1

            sp1 = std_atoms.get_scaled_positions()[:,2]
            index = [i for i,p in enumerate(sp1) if  (p<=lower_limit or p>=upper_limit)]
            std_atoms.set_constraint(FixAtoms(indices=index))
        
        else:
            upper_limit = 1.0
            lower_limit = 0.0

        std_symmetry_dataset = spglib.get_symmetry_dataset(std_atoms,0.1)
    
        unique = np.unique(std_symmetry_dataset['equivalent_atoms'])

        atoms_number = len(atoms)
        atoms = std_atoms[unique]

    else:
        spg = 1
        atoms_number = len(atoms)

        if atoms.constraints:
            cst_atoms = atoms.copy()
            length_c = cst_atoms.cell.cellpar()[2]

            while len(cst_atoms.constraints)>0:
                del cst_atoms[cst_atoms.constraints[0].index[0]]
            sp = cst_atoms.get_scaled_positions()[:,2]
            upper_limit = (np.max(sp) + 0.05 / length_c) 
            lower_limit = (np.min(sp) - 0.05 / length_c) 
        else: 
            upper_limit = 1.0
            lower_limit = 0.0

    fixids = []
    if atoms.constraints:
        for c in atoms.constraints:
            if isinstance(c, FixAtoms):
                fixids.extend(c.index)

    if shell is not None:
        assert isinstance(shell, list), "shell should be a list!"

    s = "cell\n"
    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    s += "%g %g %g %g %g %g\n" %(a, b, c, alpha, beta, gamma)
    
    if atoms.constraints:
        # GULP manual: "Here a flag of 1 implies that a degree of freedom should be allowed to vary 
        # and 0 will imply that it should be kept fixed"
        s = s[:-1]      # remove the '\n' of last line
        s += "  0 0 0 0 0 0\n"
        transformation_matrix = atoms0.get_cell()
    else:
        transformation_matrix = np.eye(3)

    s += "fractional\n"
    # core
    # GULP manual: "These are, in order, the charge, the site occupancy (which defaults to 1.0), 
    # the ion radius for a breathing shell model (which defaults to 0.0) 
    # and 3 flags to identify geometric variables (1 ⇒ vary, 0 ⇒ fix)"
    for i,atom in enumerate(atoms):
        if not atoms.constraints:
            s += "%s core %.6f %.6f %.6f \n" %(atom.symbol, atom.a, atom.b, atom.c)
        else:
            s += "%s core %.6f %.6f %.6f 0.0 1.0 0.0 %.0f\n" %(atom.symbol, atom.a, atom.b, atom.c, 0 if i in fixids else 1)
    # shell
    for atom in atoms:
        if shell is not None and atom.symbol in shell:
            s += "%s shel %.6f %.6f %.6f \n" %(atom.symbol, atom.a, atom.b, atom.c)
    s += "space\n{}\n".format(spg)

    # Write the transformation_matrix and lower/upper limit in input ?
    s += "# Transform M {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*np.ceil(transformation_matrix.flatten() * 1000)/1000)
    s += "# Constraints delineations %.6f %.6f \n"%(lower_limit, upper_limit)
    s += "# Total number of atoms {}\n".format(atoms_number)

    with open(filename, mode) as f:
        f.write(s)

def load_spg(input):
    with open(input) as inputf:
        _lines = inputf.readlines()
        i = 0
        nosym = False
        while i <len(_lines):
            if 'space' in _lines[i]:
                spg = int(_lines[i+1])
            if 'nosym' in _lines[i]:
                nosym = True
            if 'Transform M' in _lines[i]:
                transformation_matrix = _lines[i].split()[-9:]
                transformation_matrix = np.array([float(num) for num in transformation_matrix]).reshape(3,3)
            if 'Constraints delineations' in _lines[i]:
                limits = _lines[i].split()[-2:]
                limits = [float(l) for l in limits]
            if 'Total number of atoms' in _lines[i]:
                atoms_number = int(_lines[i].split()[-1])
            i +=1
    if nosym:
        spg = -1 * spg
    return spg, transformation_matrix, limits, atoms_number

def load_constraints(input):
    with open(input) as inputf:
        _lines = inputf.readlines()
        i = 0
        read_pos = False
        fixindex = []
        while i <len(_lines):
            if 'fractional' in _lines[i]:
                read_pos = True
                Num = 0
                i+=1
                continue
            if read_pos:
                origin_pos = _lines[i]
                if len(origin_pos.split()) > 8:
                    if int(origin_pos.split()[-1]) == 0:
                        fixindex.append(Num)
                Num += 1
            if 'space' in _lines[i]:
                read_pos = False
                break
            i +=1

    return fixindex


def load_gulp(filename):
    pv = 0
    forces, stress = [], []
    i = 0

    spg, transformation_matrix, limits, atoms_number = load_spg('input')
    fixindex = load_constraints('input')

    with open(filename) as f:
        lines = f.readlines()
        primitive_cell_volume = -1.0
        Primitive_energy = -1.0
        Non_primitive_energy = -1.0

    while i < len(lines): 
        
        if 'Pressure*volume' in lines[i]:
            pv = float(lines[i].split()[2])
        if 'Total lattice' in lines[i] and 'eV' in lines[i]:
            energy = float(lines[i].split()[4]) - pv
        if 'Total lattice' in lines[i] and 'eV' in lines[i+2]:
            Primitive_energy      =       float(lines[i+1].split()[-2]) - pv
            Non_primitive_energy  =       float(lines[i+2].split()[-2]) - pv

        if 'Primitive cell volume' in lines[i]:
            primitive_cell_volume = float(lines[i].split()[-2])
        if 'Non-primitive cell volume' in lines[i]:
            Non_primitive_cell_volume = float(lines[i].split()[-2])
        #if 'Final enthalpy' in lines[i] and 'eV' in lines[i]:
        #    energy = float(lines[i].split()[-2]) - pv
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
        if 'coordinates of atoms' in lines[i] or 'Final asymmetric unit coordinates' in lines[i]:
            positions, symbols = [], []
            scaled = ('fractional' in lines[i]) or ('Frac' in lines[i+4])
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
            cellpar = []
            i += 3
            for _ in range(6):
                splitline = lines[i].split()
                stress.append(float(splitline[4]))
                cellpar.append(float(splitline[1]))
                i += 1
        if 'Cartesian lattice vectors' in lines[i]:
            # if set conv, first lattice will be read
            cell = []
            i += 2
            for _ in range(3):
                cell.append([float(c) for c in lines[i].split()])
                i += 1
        i += 1

    if not scaled:
        positions = np.dot(positions, np.linalg.inv(cell))
    if spg > 1:
        issuccess = False
        if Primitive_energy == -1.0:
            Primitive_energy = energy
        if Non_primitive_energy == -1.0:
            Non_primitive_energy = energy
        if primitive_cell_volume == -1.0:
            primitive_cell_volume = np.linalg.det(cellpar_to_cell(cellpar))

        if not math.fabs(np.linalg.det(cellpar_to_cell(cellpar)) - primitive_cell_volume)/primitive_cell_volume <0.03:
            #Not primitive
            atoms = crystal(symbols = symbols, basis=positions,spacegroup = spg, cellpar = cellpar, primitive_cell = False)
            energy = Non_primitive_energy
            if spglib.get_symmetry_dataset(atoms, 0.1)['number'] > 1:
                issuccess = True
        if not issuccess:
            new_cell = np.dot(Spacegroup(spg).reciprocal_cell, cellpar_to_cell(cellpar))
            #positions = np.dot(np.dot(positions, cellpar_to_cell(cellpar)), np.linalg.inv(new_cell))
            cellpar = cell_to_cellpar(new_cell)
            energy = Primitive_energy * np.linalg.det(Spacegroup(spg).reciprocal_cell)
            atoms = crystal(symbols = symbols, basis=positions,spacegroup = spg, cellpar = cellpar, primitive_cell = False)
        assert spglib.get_symmetry_dataset(atoms, 0.1)['number'] > 1, "cannot get cell to target spacegroup '{}'".format(spg) 
    else:
        atoms = Atoms(symbols = symbols, scaled_positions = positions, cell = cell)
        

    if not len(atoms) == atoms_number:
        _Len_atoms = len(atoms)
        pri_para = spglib.find_primitive(atoms, 0.2)
        pri_cell = Atoms(cell=pri_para[0], scaled_positions=pri_para[1], numbers=pri_para[2], pbc = 1)
        atoms = multiply_cell(pri_cell, atoms_number // len(pri_cell))
        energy = energy / _Len_atoms * len(atoms)

    assert len(atoms) == atoms_number, 'cannot get {} atoms, current formula {}'.format(atoms_number, atoms.symbols.formula.count())

    if len(fixindex):
        atoms.set_cell(transformation_matrix, scale_atoms = True)        
        transformation_matrix = np.eye(3)
        sp = atoms.get_scaled_positions()[:,2]
        
        index = [i for i,p in enumerate(sp) if  (p<=limits[0] or p>=limits[1])]
        atoms.set_constraint(FixAtoms(indices=index))

    else:
        # standardize cellpar in case that fail in check_cell
        # Why in 'else' !! : constraints disappears when make_supercell
        # atoms = make_supercell(atoms, minkowski_reduce(atoms.get_cell()[:])[1])

        def add_symmetry(atoms0, keep_n_atoms=True, to_primitive=False):
            std_para = spglib.standardize_cell(atoms0, symprec=0.1, to_primitive=to_primitive)
            if std_para is None:
                return False
            std_atoms = Atoms(cell=std_para[0], scaled_positions=std_para[1], numbers=std_para[2])
            if keep_n_atoms:
                if len(atoms0) % len(std_atoms) == 0:
                    std_atoms = multiply_cell(std_atoms, len(atoms0) // len(std_atoms))
                elif not to_primitive:
                    return add_symmetry(keep_n_atoms, to_primitive=True)
            atoms0.set_cell(std_atoms.cell)
            atoms0.set_scaled_positions(std_atoms.get_scaled_positions())
            atoms0.set_atomic_numbers(std_atoms.numbers)
            return True
        
        try:
            atoms0 = atoms.copy()
            if add_symmetry(atoms0):
                atoms = atoms0
        except:
            pass

    atoms.pbc = True
    # idk why i set pbc here... If you meet bugs, try delete this pbc setting

    atoms.info['energy'] = energy
    atoms.info['forces'] = forces
    atoms.info['stress'] = stress
    return atoms
