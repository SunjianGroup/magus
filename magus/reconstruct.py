from . import moveatom
import numpy as np
import ase.io
from ase.data import atomic_numbers, covalent_radii


def str_(l):
    l=str(l)
    return l[l.find('[')+1:l.find(']')]

class move:
    def __init__(self, originpositions, shift, atomnumber, atomname, lattice):
        self.info=moveatom.Info(np.random.randint(1000))
        for i in range(len(atomnumber)):
            pos=np.array(originpositions[i]).flatten()
            self.info.AppendAtoms(int(atomnumber[i]),atomname[i],covalent_radii[atomic_numbers[atomname[i]]],pos)   #(atomsnumber, atomname, atomradius, atomspos)
            self.info.AppendShift(np.array(shift[i]))  #shift for the atoms above

        self.lattice=lattice.flatten()
        self.info.SetLattice(np.array(self.lattice))

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
    def __init__(self, range, num_layer, originlayer, extratom=0.5, extratomrange=None):
        self.originlayer=originlayer
        self.range=range
        self.layernum=num_layer+1
        self.atoms = ase.io.read(self.originlayer)
        self.pos=[]
        self.atomname=[]
        self.atomnum=[]
        self.shifts=[]
        self.extratom=extratom
        if not extratomrange:
            self.extratomrange=range
        else:
            self.extratomrange=extratomrange

    def reconstr(self):
        self.lattice=np.array(self.atoms.get_cell())
        self.lattice[2]*=self.layernum

        atomicnum=self.atoms.numbers
        atomname=self.atoms.get_chemical_symbols()
        unique, t=np.unique(atomicnum, return_index=True)
        for i in range(len(unique)):
            atomnum=np.sum(atomicnum==unique[i])
            self.atomname.append(atomname[t[i]])
            self.atomnum.append(atomnum*(self.layernum-1))
            pos=np.array(self.atoms.get_scaled_positions())
            atompos=[]
            shift1=[]
            for layer in range(self.layernum-1):
                p=pos[t[i]:t[i]+atomnum].copy()
                for a in p:
                    a[2]+=layer
                    a[2]/=self.layernum
                atompos=np.append(atompos, p)
               
                shift1.append(p[:,2].copy())
        
                p=pos[t[i]:t[i]+atomnum].copy()

            shift1=np.array(shift1).flatten()
            shift1*=self.range
       
            shift2=[]
            #add extra atoms below
            p=pos[t[i]:t[i]+atomnum].copy()
            for a in p:
                if(a[2]<=self.extratom):
                    a[2]+=(self.layernum-1)
                    a[2]/=self.layernum
                    atompos=np.append(atompos, a)
                    self.atomnum[-1]+=1
                    shift2.append(a[2]*self.extratomrange)

            shift2=np.array(shift2).flatten()
            shift1=np.append(shift1, shift2)

            self.shifts.append(shift1)

            atompos=np.array(atompos).reshape(self.atomnum[-1], 3)

            self.pos.append(atompos)

            
        m=move(self.pos, self.shifts, self.atomnum, self.atomname, self.lattice)
            
        label = m.Shift(0.5, 100)
        self.positions=m.GetPos()
        
        self.positions=np.array(self.positions).reshape(sum(self.atomnum),3)

        return label, self.positions

    def WritePoscar(self, filename):
        f=open(filename,'w')
       
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

if __name__ == '__main__':
    t=reconstruct(0.8, 0, "teststructure.vasp")
    t.reconstr()
    t.WritePoscar("result.vasp")
