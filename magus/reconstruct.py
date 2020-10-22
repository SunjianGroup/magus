from . import moveatom
import numpy as np
import ase.io
#import spglib 
from ase.data import atomic_numbers, covalent_radii
from .utils import sort_elements


def str_(l):
    l=str(l)
    return l[l.find('[')+1:l.find(']')]

class move:
    def __init__(self, originpositions, shift, atomnumber, atomname, lattice, spacegroup = 1, eq_atoms=None):
        self.info=moveatom.Info(np.random.randint(1000))
        for i in range(len(atomnumber)):
            pos=np.array(originpositions[i]).flatten()
            self.info.AppendAtoms(int(atomnumber[i]),atomname[i],covalent_radii[atomic_numbers[atomname[i]]],pos)   #(atomsnumber, atomname, atomradius, atomspos)
            self.info.AppendShift(np.array(shift[i]))  #shift for the atoms above
            if(eq_atoms):
                self.info.EquivalentAtoms(np.array(1.0*eq_atoms[i]))
        self.lattice=lattice.flatten()
        self.info.SetLattice(np.array(self.lattice))
        self.info.spacegroup = spacegroup
        self.info.resultsize=1
        
    def Rotate(self, centeratomtype, centeratompos, clustersize, targetatomtype, threshold, maxattempts):
        self.info.Rotate(centeratomtype, centeratompos, clustersize, targetatomtype, threshold, maxattempts)
        self.info.WritePoscar()
        '''
        for i in range(self.info.resultsize):
            print(self.info.GetPosition(i))
        '''
        return

    def Shift(self, threshold, maxattempts):
        label = self.info.Shift(threshold, maxattempts)
        self.info.WritePoscar()
        '''
        for i in range(self.info.resultsize):
            print(self.info.GetPosition(i))
        '''
        return label

    def WritePoscar(self):
        self.info.WritePoscar()
        return

    def GetPos(self):
        return self.info.GetPosition(0)

#m=move()
#m.Shift(0.5, 100)
#m.Rotate(1, 1, 6, 0, 0.5, 100)

class reconstruct:
    def __init__(self, moverange, originlayer, threshold, maxattempts=100):
        self.originlayer=originlayer
        self.range=moverange
        self.threshold = threshold
        self.maxattempts = maxattempts

        self.atoms = sort_elements(self.originlayer)
        self.atoms.set_cell(self.atoms.get_cell_lengths_and_angles().copy(), scale_atoms=True)
        
        self.pos=[]
        self.atomname=[]
        self.atomnum=[]
        self.shifts=[]
        self.eq_atoms=[]

    def reconstr(self):
        self.lattice=np.array(self.atoms.get_cell())
        atomicnum=self.atoms.numbers
        atomname=self.atoms.get_chemical_symbols()
        #dataset = spglib.get_symmetry_dataset(self.atoms, symprec=1e-2)
        #self.spacegroup = dataset['number']
        #transmatrix = dataset['transformation_matrix']
        #shiftarray = dataset['origin_shift']
        #eq_atoms=dataset['equivalent_atoms']

        unique, t=np.unique(atomicnum, return_index=True)
        for i in range(len(unique)):
            atomnum=np.sum(atomicnum==unique[i])
            self.atomname.append(atomname[t[i]])

            index = range(t[i], t[i]+atomnum)
            atom_part = self.atoms[index].copy()
            #eqatom = eq_atoms[index].copy()
            #eqatom -= t[i]

            #index = sorted(range(0,atomnum), key=lambda x: eqatom[x] )
            #atom_part = atom_part[index]
            #eqatom = eqatom[index]

            atompos=atom_part.get_scaled_positions().copy()
            #for i in range(len(atompos)):
                #atompos[i] = np.dot(transmatrix, atompos[i]) + shiftarray

            shift1=atompos[:,2].copy()
            shift1=np.array(shift1).flatten()
            shift1*=self.range

            self.shifts.append(shift1)
            self.atomnum.append(atomnum)

            self.pos.append(atompos)
            #self.eq_atoms.append(eqatom)

        m=move(self.pos, self.shifts, self.atomnum, self.atomname, self.lattice)  #, self.spacegroup, self.eq_atoms)
            
        label = m.Shift(self.threshold, self.maxattempts)
        if label:
            self.positions=m.GetPos()
            self.positions=np.array(self.positions).reshape(sum(self.atomnum),3)

            #for i in range(len(self.positions)):
                #self.positions[i] = np.dot(np.linalg.inv(transmatrix), (self.positions[i]-shiftarray))
        
            return label, self.positions
        else:
            return label, None

    def WritePoscar(self, filename):
        f=open(filename,'w')
        f.write("filename\n")
        f.write('1 \n')
        for i in range(3):
            f.write(str_(self.lattice[i])+'\n')

        for name in self.atomname:
            f.write(name+'  ')

        f.write('\n')
        f.write(str_(np.array(self.atomnum)))
        f.write('\nDirect\n') 
        po=self.positions
        for i in range(len(po)):
            f.write(str_(po[i])+'\n')
        f.close()
        return


class cutcell:
    def __init__(self,originstruct,rcs_supercell, slicepos, direction=None):
        
        originatoms = originstruct.copy()
        atoms=originatoms.copy()


        #Add direction here
        if direction:
            rx, ry, rz = direction[0], direction[1], direction[2]            
            supercell = atoms * (4, 4, 4)
            supercell.translate( [ -1*np.sum(atoms.get_cell(), axis=0)] *len(supercell))
            
            points = [[rx, 0, 0], [0, ry, 0], [0, 0, rz]]
            newcell_c = np.array([rx, ry, rz])

            if [0,0,0] in points:
                points.remove([0,0,0])
                newcell_a = np.array(points[1]) - np.array(points[0])
                
                newcell_a = norm(newcell_a)
                newcell_b = np.cross(newcell_a, newcell_c)
                newcell_b = np.array([1 if newcell_b[i]!=0 else 0 for i in range(3)])
            
            else:
                newcell_a = np.array(points[1]) - np.array(points[0])
                
                newcell_b = np.array(points[2]) - np.array(points[0])

            newcell = np.array([newcell_a, newcell_b, newcell_c])                
            newcell = np.dot(newcell, atoms.get_cell())
            supercell.translate([-0.5*np.sum(newcell, axis=0)]*len(supercell))
            supercell.set_cell(newcell)
            pos =supercell.get_scaled_positions(wrap=False).copy()

            for i in range(len(pos)):
                for j in range(3):
                    if abs(pos[i][j]-1)< 1e-4:
                        pos[i][j] = 1        
                    if abs(pos[i][j])<1e-4:
                        pos[i][j]=0
        
            index = [i for i in range(len(pos)) if InCell(pos[i])==True]
            
            atoms = supercell[index].copy()
            
            #rotate cellparm a to axis x
            atoms.rotate(atoms.get_cell()[0], [1,0,0],rotate_cell=True)
            #atoms.rotate(atoms.get_cell()[2], [0,0,1],rotate_cell=True)
            '''may cause errs
            stdcell = atoms.get_cell().copy()
            stdcell[2] = norm(stdcell[2])
            atoms.set_cell(stdcell, scale_atoms=True)
            '''
        
            originatoms = atoms.copy()

        rcs_x, rcs_y, rcs_z = rcs_supercell[0], rcs_supercell[1], rcs_supercell[2]
        cell=originatoms.get_cell().copy()
        cell[0]*= rcs_x
        cell[1]*= rcs_y

        supercell = [rcs_x, rcs_y , rcs_z]
        
        for i in range(3):
            for x in range(1, supercell[i]):
                atoms_tmp=originatoms.copy()
                trans=[originatoms.get_cell()[i]*x]*len(atoms_tmp)
                atoms_tmp.translate(trans)
                atoms+=atoms_tmp
            originatoms = atoms.copy()

        
        atoms.set_cell(cell)
        originatoms = atoms.copy()
        
        pop= []
        
        
        cell = originatoms.get_cell().copy()
        cell[2] = cell[2]*slicepos[0]
        trans=[ cell[2]*(-1) ]*len(atoms)
        atoms.translate(trans)

        for i in range(1, len(slicepos)):

            cell = originatoms.get_cell().copy()
            cell[2] = cell[2]*(slicepos[i]-slicepos[i-1])
            atoms.set_cell(cell)

            pos = atoms.get_scaled_positions(wrap=False).copy()
            index=[]
            for atom in atoms:
                if pos[atom.index][2]>=0 and pos[atom.index][2]<1 :
                    index.append(atom.index)

            layerslice = atoms[index] .copy()
            layerslice=sort(layerslice)
            pop.append(layerslice)

            trans=[ cell[2]*(-1) ]*len(atoms)
            atoms.translate(trans)

        #add extravacuum to rcs_layer  
        
        cell = pop[2].get_cell()
        cell[2]*=1.5
        pop[2].set_cell(cell)
        
        logging.info("save cutslices into file layerslices.traj")
        ase.io.write("layerslices.traj",pop,format='traj')






















if __name__ == '__main__':
    t=reconstruct(0.8, ase.io.read("POSCAR_3.vasp",format='vasp'), 0.8,2 )
    t.reconstr()
    t.WritePoscar("result.vasp")
