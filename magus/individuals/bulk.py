import numpy as np
from ase.data import covalent_radii, atomic_numbers
from .base import Individual
from .molecule import Molfilter
from ..utils import check_parameters


class Bulk(Individual):
    @classmethod
    def set_parameters(cls, **parameters):
        super().set_parameters(**parameters)
        Default = {
            'mol_detector': 0, 
            'bond_ratio': 1.1,
            'radius': [covalent_radii[atomic_numbers[atom]] for atom in cls.symbol_list]}
        check_parameters(cls, parameters, [], Default)
        cls.volume = np.array([4 * np.pi * r ** 3 / 3 for r in cls.radius])

    def __init__(self, *args, **kwargs):
        if 'symbols' in kwargs:
            if isinstance(kwargs['symbols'], Molfilter):
                kwargs['symbols'] = kwargs['symbols'].to_atoms()
        if len(args) > 0:
            if isinstance(args[0], Molfilter):
                args = list(args)
                args[0] = args[0].to_atoms()
        super().__init__(*args, **kwargs)

    def for_heredity(self):
        atoms = self.copy()
        if self.mol_detector > 0:
            atoms = Molfilter(atoms, detector=self.mol_detector, coef=self.bond_ratio)
        return atoms
