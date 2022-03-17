import numpy as np
from ase.io import read
from .base import Individual
from ..utils import check_parameters
from .molecule import Molfilter


def split_modifier(modifier):
    res = []
    for i in range(np.min( [ len(symbols), len(modifier) ])):
        expand = []
        for item in modifier[i]:
            if isinstance(item, int):
                expand.append(item)
            elif isinstance(item, str):
                if '~' not in item:
                    raise Exception ("wrong format")
                s1, s2 = item.split('~')
                s1, s2 = int(s1), int(s2)
                expand.extend(list(range(s1, s2+1)))
        res.append(expand)
    return res


def get_newcell(atoms, thickness):
    # we multiply c by a ratio so the thickness of vacuum is fixed for non orthorhombic cell
    ratio = thickness * np.linalg.norm(atoms.cell.reciprocal()[2]) + 1
    newcell = atoms.get_cell()
    newcell[2] *= ratio
    return newcell


def add_extra_layer(atoms, layer):
    """
    add extra layer to atoms
    atoms is always at the end
    """
    thickness = 1 / np.linalg.norm(atoms.cell.reciprocal()[2])
    n_top = len(atoms)
    newatoms = layer.copy()  # use layer as basic atoms, so the constrains will ramain

    newatoms.set_cell(get_newcell(newatoms, thickness))
    newatoms.extend(atoms)
    newatoms.positions[-n_top:, 2] += thickness
    return newatoms


def add_vacuum_layer(atoms, thickness):
    newatoms = atoms.copy()
    newatoms.set_cell(get_newcell(atoms, thickness))
    return newatoms


class Surface(Individual):
    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)
        Default = {
            'vacuum': 7, 
            'SymbolsToAdd': None, 
            'AtomsToAdd': None, 
            'DefectToAdd': None, 
            'refE': 0, 
            'refFrml': None,
            'vacuum_thickness': 10,
            }
        check_parameters(cls, parameters, [], Default)
        cls.bulk_layer = read('bulk.vasp')
        cls.buffer_layer = read('buffer.vasp')

    def __init__(self, *args, **kwargs):
        if 'symbols' in kwargs:
            if isinstance(kwargs['symbols'], Molfilter):
                kwargs['symbols'] = kwargs['symbols'].to_atoms()
        if len(args) > 0:
            if isinstance(args[0], Molfilter):
                args = list(args)
                args[0] = args[0].to_atoms()
        super().__init__(*args, **kwargs)
        if 'n_top' in self.info:
            n_top = self.info['n_top']
            del self.info['n_top']
            self.__init__(self[-n_top:])

    def substrate_sym(self, symprec = 1e-4):
        
        if not hasattr(self, "substrate_symmetry"):
            rlxatoms = self.layerslices[1] * (*self.info['size'], 1)
            rlxatoms.info['size'] = self.info['size']
            allatoms = self.addextralayer('bulk', atoms = rlxatoms, add = 1)
            
            symdataset = spglib.get_symmetry_dataset(allatoms, symprec= symprec)
            self.substrate_symmetry = list(zip(symdataset['rotations'], symdataset['translations']))
            self.substrate_symmetry = [s for s in self.substrate_symmetry if ( (s[0]==s[0]*np.reshape([1,1,0]*2+[0]*3, (3,3))+ np.reshape([0]*8+[1], (3,3))).all()  and not (s[0]==np.eye(3)).all())]
            cell = allatoms.get_cell()[:]

            #some simple tricks for '2', 'm', '4', '3', '6' symmetry. No sym_matrix containing 'z'_axis transformation are included.
            for i, s in enumerate(self.substrate_symmetry):
                r = s[0]
                sample = np.array([*np.random.uniform(1,2,2), 0])
                sample_prime = np.dot(r, sample)
                cart, cart_prime = np.dot(sample, cell), np.dot(sample_prime, cell)
                length = np.sum([i**2 for i in cart])
                angle = math.acos(np.dot(cart, cart_prime)/ length) / math.pi * 180
                if not (np.round(angle) == np.array([180, 90, 120, 60])).any():
                    self.substrate_symmetry[i] = self.substrate_symmetry[i]+('m',)
                else:
                    self.substrate_symmetry[i] = self.substrate_symmetry[i]+(int(np.round(360.0/angle)),)

            #self.substrate_symmetry is a list of tuple (r, t, multiplity), in which multiplity = 2 for '2', 'm', others = self.rotaterank.

        return self.substrate_symmetry

    def for_calculate(self):
        atoms = add_extra_layer(self, self.buffer_layer)
        atoms = add_extra_layer(atoms, self.bulk_layer)  
        atoms = add_vacuum_layer(atoms, self.vacuum_thickness)
        atoms.info['n_top'] = len(self)   # record the n_top so extra layers can easily removed by atoms[-n_top:]
        return atoms
