from ase.atoms import Atoms
from ase.calculators.orca import ORCA
import logging
#from ase.io.orca import write_orca
from ase.calculators.calculator import FileIOCalculator
from ase.units import Hartree, Bohr
import numpy as np
import os, sys
'''
modified ASE's ORCA Calculator to a FileIO class, for borrowing some FileIo codes from ase.
See notes of .localopt.ASEORCACalculator for more info about ORCA-ASE interface.
'''

class RelaxOrca(ORCA):
    """
    Slightly modify ORCA's read_forces and read_energy, in case orca fails.
    """
    def read_forces(self):
        try:
            ORCA.read_forces(self)
        except FileNotFoundError:
            #if orca fails, no file named orca.engrad can be found, just return a wrong result and remove this structure.
            errmessage = os.popen("grep \"ORCA finished\" orca.out").readline()
            logging.warning("{}, used wrong energy and forces to remove this structure".format(errmessage))
            self.results['energy'] = 10000
            self.results['forces'] = np.array([10000])
        return 

class OrcaIo(RelaxOrca, FileIOCalculator):
    def __init__(self, label='orca'):
        RelaxOrca.__init__(self, label=label)

    def write_input(self, atoms, properties=None, system_changes=None, orcasimpleinput = 'TightSCF Opt', orcablocks = [''], charge = 0, mult = 1):
        FileIOCalculator.write_input(self, atoms, properties=None, system_changes=None)
        f = open(self.label + '.inp', 'w')
        f.write("! %s \n" % orcasimpleinput)
        for s in orcablocks:
            if s != '\n':
                f.write("%s \n" % s)

        #borrowed from ase.io.orca.write_orca here
        f.write('*xyz')
        f.write(" %d" % charge)
        f.write(" %d \n" % mult)
        for atom in atoms:
            if atom.tag == 71:  # 71 is ascii G (Ghost)
                symbol = atom.symbol + ' : '
            else:
                symbol = atom.symbol + '   '
            f.write(symbol +
                    str(atom.position[0]) + ' ' +
                    str(atom.position[1]) + ' ' +
                    str(atom.position[2]) + '\n')
        f.write('*\n')
        f.close()

    def read_forces(self):
        RelaxOrca.read_forces(self)
        return self.results['forces']
    
    def read_energy(self):
        try:
            file = open(self.label + '.engrad', 'r')
        except FileNotFoundError:
            self.results['energy'] = 10000
            return self.results['energy']
            
        lines = file.readlines()
        file.close()
        for i, line in enumerate(lines):
            if line.find('# The current total energy') >= 0:
                energy = float(lines[i+2])
                break
        self.results['energy'] = energy * Hartree 
        return self.results['energy']

    def read_positions(self):
        try:
            f = open(self.label + '.engrad', 'r')
        except FileNotFoundError:
            self.results['symbols'] = np.array([1])
            self.results['positions'] = np.array([[0,0,0]])
            return self.results['symbols'], self.results['positions']

        for line in f:
            if line.startswith('# The atomic numbers and current coordinates'):
                break

        line = f.readline()     #skip next '#' line
        symbols = []
        positions = []
        
        for line in f:
            if line.startswith('#'):
                break
            words = line.split()
            symbols.append(words[0])
            positions.append([float(word) for word in words[1:]])
        self.results['symbols'] = symbols
        self.results['positions'] = np.array(positions) * Bohr
        return self.results['symbols'], self.results['positions']

    def read_relaxsteps(self):
        command = "grep \"GEOMETRY OPTIMIZATION CYCLE\" "+ self.label + ".out | tail -1 | awk '{print $5}'"
        relaxsteps = os.popen(command).readlines()[0]
        return int(relaxsteps)

