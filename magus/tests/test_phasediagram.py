import unittest, os
from ase.io import read
from ase import Atoms
from ase.phasediagram import PhaseDiagram as ASEPhaseDiagram
from magus.phasediagram import PhaseDiagram, get_units
from magus.utils import get_units_numlist


class TestPhaseDiagram:

    def test_get_units(self):
        self.assertIsNotNone(get_units(self.frames))

    def test_decompose(self):
        pd = PhaseDiagram(self.frames)
        refs = [(a.get_chemical_formula(), a.info['enthalpy'] * len(a)) for a in self.frames]
        for s in set([s for a in self.frames for s in a.symbols]):
            refs.append((s, 1000))
        ase_pd = ASEPhaseDiagram(refs, verbose=False)
        for a in self.frames:
            energy = pd.decompose(a) * len(a)
            ase_energy = ase_pd.decompose(a.get_chemical_formula())[0]
            self.assertAlmostEqual(energy, ase_energy)


class TestBinary(TestPhaseDiagram, unittest.TestCase):
    def setUp(self):
        path = os.path.dirname(__file__)
        self.frames = read(os.path.join(path, 'POSCARS/Al-O.traj'), ':')


class TestPseudoBinary(TestPhaseDiagram, unittest.TestCase):
    def setUp(self):
        path = os.path.dirname(__file__)
        self.frames = read(os.path.join(path, 'POSCARS/Al2O3-AlH3.traj'), ':')

class TestTrinary(TestPhaseDiagram, unittest.TestCase):
    def setUp(self):
        path = os.path.dirname(__file__)
        self.frames = read(os.path.join(path, 'POSCARS/Al-O-H.traj'), ':')


if __name__ == '__main__':
    unittest.main()