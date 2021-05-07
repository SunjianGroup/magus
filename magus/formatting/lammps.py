import numpy as np
import ase
from ase.atoms import Atoms
from ase.quaternions import Quaternions
from ase.calculators.singlepoint import SinglePointCalculator
from ase.parallel import paropen
from ase.utils import basestring
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write


class Atomic:
    def __init__(self, atoms, symbol_to_type):
        atoms.set_cell(atoms.cell.cellpar(), True)
        self.atoms = atoms
        self.n = len(atoms)
        self.symbol_to_type = symbol_to_type
        self.cell=self.atoms.get_cell()
        self.output = ['\n']

    def print_natoms(self):
        self.output.append('{} atoms\n'.format(self.n))
    
    def print_ntypes(self):
        self.output.append('{} atom types\n'.format(len(self.symbol_to_type)))
        
    def print_cell(self):
        self.output.append('0.000000    {}   xlo xhi'.format(self.cell[0,0]))
        self.output.append('0.000000    {}   ylo yhi'.format(self.cell[1,1]))
        self.output.append('0.000000    {}   zlo zhi\n'.format(self.cell[2,2]))
        self.output.append('{}   {}   {}   xy xz yz'.format(self.cell[1,0],self.cell[2,0],self.cell[2,1]))
        
    def print_mass(self):
        self.output.append('\nMasses\n')
        for symbol in self.symbol_to_type.keys():
            self.output.append('{} {}'.format(self.symbol_to_type[symbol], atomic_masses[atomic_numbers[symbol]]))
            
    def print_atoms(self):
        self.output.append('\nAtoms\n')
        for i,atom in enumerate(self.atoms):
            self.output.append('{} {} {} {} {}'.format(i + 1, self.symbol_to_type[atom.symbol],
                atom.position[0], atom.position[1], atom.position[2]))
    
    def dump(self, filename, mode='w'):
        self.print_natoms()
        self.print_ntypes()
        self.print_cell()
        self.print_mass()
        self.print_atoms()
        with open(filename, mode) as f:
            for line in self.output:
                f.write(line+'\n')        

class Charge(Atomic):
    def __init__(self, atoms, charges):
        super().__init__(atoms)
        self.charge = charges
    
    def print_atoms(self):
        self.output.append('\nAtoms\n')
        for i, atom in enumerate(self.atoms):
            self.output.append('{} {} {} {} {} {}'.format(i + 1, self.symbol_to_type[atom.symbol],
                self.charge[atom.symbol], atom.position[0], atom.position[1], atom.position[2]))


def dump_lmps(atoms, filename, symbol_to_type, mode='w', atom_type='atomic'):
    if atom_type == 'atomic':
        dumper = Atomic(atoms, symbol_to_type)
    dumper.dump(filename, mode=mode)

def load_lmps(filename, type_to_symbol, timerange=None, order=True):
    specorder = []
    for i in range(len(type_to_symbol)):
        specorder.append(type_to_symbol[i])
    frames = read(filename, index=':', format='lammps-dump-text', specorder=specorder)
    return frames
