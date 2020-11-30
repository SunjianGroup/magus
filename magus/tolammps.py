import ase.io
import ase
#TODO move to formatting
class Atomic:
    def __init__(self,atoms):
        atoms.set_cell(atoms.get_cell_lengths_and_angles(),True)
        self.atoms=atoms
        self.n=len(atoms)
        self.type=dict()
        for n in atoms.get_atomic_numbers():
            if n not in self.type:
                self.type[n]=len(self.type)+1
        self.cell=self.atoms.get_cell()
        self.output = ['\n']

    def print_natoms(self):
        self.output.append('{} atoms\n'.format(self.n))
    
    def print_ntypes(self):
        self.output.append('{} atom types\n'.format(len(self.type)))
        
    def print_cell(self):
        self.output.append('0.000000    {}   xlo xhi'.format(self.cell[0,0]))
        self.output.append('0.000000    {}   ylo yhi'.format(self.cell[1,1]))
        self.output.append('0.000000    {}   zlo zhi\n'.format(self.cell[2,2]))
        self.output.append('{}   {}   {}   xy xz yz'.format(self.cell[1,0],self.cell[2,0],self.cell[2,1]))
        
    def print_mass(self):
        self.output.append('\nMasses\n')
        for n in self.type.keys():
            self.output.append('{} {}'.format(self.type[n],ase.data.atomic_masses[n]))
            
    def print_atoms(self):
        self.output.append('\nAtoms\n')
        for i,atom in enumerate(self.atoms):
            self.output.append('{} {} {} {} {}'.format(i+1,self.type[atom.number],
                atom.position[0],atom.position[1],atom.position[2]))
    
    def dump(self,filename):
        self.print_natoms()
        self.print_ntypes()
        self.print_cell()
        self.print_mass()
        self.print_atoms()
        with open(filename,'w') as f:
            for line in self.output:
                f.write(line+'\n')        

class Charge(Atomic):
    def __init__(self,atoms,charges):
        super().__init__(atoms)
        self.charge=charges
    
    def print_atoms(self):
        self.output.append('\nAtoms\n')
        for i,atom in enumerate(self.atoms):
            self.output.append('{} {} {} {} {} {}'.format(i+1,self.type[atom.number],
                self.charge[atom.number],atom.position[0],atom.position[1],atom.position[2]))

if __name__ == '__main__':
    from lammps import lammps
    from ase import Atom, Atoms
    from ase.build import bulk
    from ase.calculators.lammpslib import LAMMPSlib

    cmds = ["pair_style eam/alloy","pair_coeff * * NiAlH_jea.eam.alloy Ni H"]                                                            

    Ni = bulk('Ni', cubic=True)
    H = Atom('H', position=Ni.cell.diagonal()/2)
    NiH = Ni + H 

    lammps = LAMMPSlib(lmpcmds=cmds, log_file='test.log')

    NiH.set_calculator(lammps)
    print("Energy ", NiH.get_potential_energy())