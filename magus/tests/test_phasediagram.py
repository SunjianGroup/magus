import unittest, os
from ase.io import read
from ase.phasediagram import PhaseDiagram as ASEPhaseDiagram
from magus.phasediagram import PhaseDiagram, get_units
from magus.utils import get_units_numlist


class TestPhaseDiagram:

    def test_get_units(self):
        units = get_units(self.frames)
        for atoms in self.frames:
            self.assertIsNotNone(get_units_numlist(atoms, units))

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
        self.frames = read(os.path.join(path, 'POSCARS/AlxOy.traj'), ':')


class TestPseudoBinary(TestPhaseDiagram, unittest.TestCase):
    def setUp(self):
        path = os.path.dirname(__file__)
        self.frames = read(os.path.join(path, 'POSCARS/AlOHxH2Oy.traj'), ':')

class TestTrinary(TestPhaseDiagram, unittest.TestCase):
    def setUp(self):
        path = os.path.dirname(__file__)
        self.frames = read(os.path.join(path, 'POSCARS/AlxOyHz.traj'), ':')


if __name__ == '__main__':
    unittest.main()