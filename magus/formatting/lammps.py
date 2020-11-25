import ase.io
import ase

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

from ase.atoms import Atoms
from ase.quaternions import Quaternions
from ase.calculators.singlepoint import SinglePointCalculator
from ase.parallel import paropen
from ase.utils import basestring
import numpy as np

def read_lammps_dump(f,timerange=None, order=True, atomsobj=Atoms,typelist=None,numlist=None):
    if typelist is None and numlist is None:
        raise Exception("sha dou mei you wo za du?")
    natoms = 0
    images = []
    line = 'go'
    while line:
        line = f.readline()
        if 'ITEM: TIMESTEP' in line:
            lo = []
            hi = []
            tilt = []
            id = []
            symbols = []
            positions = []
            scaled_positions = []
            velocities = []
            forces = []
            quaternions = []
            line = f.readline()
            time = int(line.split()[0])
            if timerange and time > np.max(timerange):
                break

        if 'ITEM: NUMBER OF ATOMS' in line:
            line = f.readline()
            natoms = int(line.split()[0])
            
        if 'ITEM: BOX BOUNDS' in line:
            # save labels behind "ITEM: BOX BOUNDS" in
            # triclinic case (>=lammps-7Jul09)
            tilt_items = line.split()[3:]
            for i in range(3):
                line = f.readline()
                fields = line.split()
                lo.append(float(fields[0]))
                hi.append(float(fields[1]))
                if (len(fields) >= 3):
                    tilt.append(float(fields[2]))

            # determine cell tilt (triclinic case!)
            if (len(tilt) >= 3):
                # for >=lammps-7Jul09 use labels behind
                # "ITEM: BOX BOUNDS" to assign tilt (vector) elements ...
                if (len(tilt_items) >= 3):
                    xy = tilt[tilt_items.index('xy')]
                    xz = tilt[tilt_items.index('xz')]
                    yz = tilt[tilt_items.index('yz')]
                # ... otherwise assume default order in 3rd column
                # (if the latter was present)
                else:
                    xy = tilt[0]
                    xz = tilt[1]
                    yz = tilt[2]
            else:
                xy = xz = yz = 0
            xhilo = (hi[0] - lo[0]) - (xy**2)**0.5 - (xz**2)**0.5
            yhilo = (hi[1] - lo[1]) - (yz**2)**0.5
            zhilo = (hi[2] - lo[2])
            if xy < 0:
                if xz < 0:
                    celldispx = lo[0] - xy - xz
                else:
                    celldispx = lo[0] - xy
            else:
                celldispx = lo[0]
            celldispy = lo[1]
            celldispz = lo[2]

            cell = [[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]]
            celldisp = [[celldispx, celldispy, celldispz]]

        def add_quantity(fields, var, labels):
            for label in labels:
                if label not in atom_attributes:
                    return
            var.append([float(fields[atom_attributes[label]])
                        for label in labels])
                
        if 'ITEM: ATOMS' in line:
            if timerange and time not in timerange:
                for n in range(natoms):
                    line = f.readline()
                continue
            # (reliably) identify values by labels behind
            # "ITEM: ATOMS" - requires >=lammps-7Jul09
            # create corresponding index dictionary before
            # iterating over atoms to (hopefully) speed up lookups...
            atom_attributes = {}
            for (i, x) in enumerate(line.split()[2:]):
                atom_attributes[x] = i
            for n in range(natoms):
                line = f.readline()
                fields = line.split()
                id.append(int(fields[atom_attributes['id']]))
                if numlist:
                    symbols.append(numlist[id[-1]])
                if typelist:
                    symbols.append(typelist[int(fields[atom_attributes['type']])])
                add_quantity(fields, positions, ['x', 'y', 'z'])
                add_quantity(fields, scaled_positions, ['xs', 'ys', 'zs'])
                add_quantity(fields, velocities, ['vx', 'vy', 'vz'])
                add_quantity(fields, forces, ['fx', 'fy', 'fz'])
                add_quantity(fields, quaternions, ['c_q[1]', 'c_q[2]',
                                                   'c_q[3]', 'c_q[4]'])

            if order:
                def reorder(inlist):
                    if not len(inlist):
                        return inlist
                    outlist = [None] * len(id)
                    for i, v in zip(id, inlist):
                        outlist[i - 1] = v
                    return outlist
                symbols = reorder(symbols)
                positions = reorder(positions)
                scaled_positions = reorder(scaled_positions)
                velocities = reorder(velocities)
                forces = reorder(forces)
                quaternions = reorder(quaternions)

            if len(quaternions):
                images.append(Quaternions(symbols=types,
                                          positions=positions,
                                          cell=cell, celldisp=celldisp,
                                          quaternions=quaternions))
            elif len(positions):
                images.append(atomsobj(
                    symbols=symbols, positions=positions,
                    celldisp=celldisp, cell=cell))
            elif len(scaled_positions):
                images.append(atomsobj(
                    symbols=symbols, scaled_positions=scaled_positions,
                    celldisp=celldisp, cell=cell))

            if len(velocities):
                images[-1].set_velocities(velocities)
            if len(forces):
                images[-1].info['forces'] = forces
    return images



if __name__=="__main__":
    import time
    filename='Li.dump'
    typelist=np.array([3]*1025)

    with open(filename) as f:
        t=time.time()
        a=read_lammps_dump(f,typelist,timerange=np.arange(0,60000,100))
        print(time.time()-t)
    

    

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