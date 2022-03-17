import unittest, re, spglib, os
import numpy as np
from ase import Atoms
from magus.parameters import magusParameters
from magus.generators import SPGGenerator, MoleculeSPGGenerator


class TestSPGGenerator(unittest.TestCase):
    symprec = 0.1

    def setUp(self):
        if os.path.exists('formula_pool'):
            os.remove('formula_pool')

    def tearDown(self):
        if os.path.exists('formula_pool'):
            os.remove('formula_pool')

    def get_spg(self, atoms):
        spg = spglib.get_spacegroup(atoms, self.symprec)
        pattern = re.compile(r'\(.*\)')
        spg = pattern.search(spg).group()
        spg = int(spg[1:-1])
        return spg

    def test_fix_bulk(self):
        para = {
            'minNAtoms': 24,
            'maxNAtoms': 48,
            'spacegroup': 136,
            'symbols': ['Ti','O'],
            'formula': [1, 2],
            'dRatio': 0.7,
            'volumeRatio': 2.0,
        }
        g = magusParameters(para).RandomGenerator
        self.assertIsInstance(g, SPGGenerator)
        frames = g.generate_pop(5)
        for atoms in frames:
            self.assertEqual(self.get_spg(atoms), 136)

    def test_var_bulk(self):
        para = {
            'formulaType': 'var',
            'minNAtoms': 24,
            'maxNAtoms': 48,
            'spacegroup': '20-30',
            'symbols': ['Zn','O','H'],
            'formula': [[1,0,0],[0,1,1]],
            'dRatio': 0.7,
            'volumeRatio': 3.0,
        }
        g = magusParameters(para).RandomGenerator
        self.assertIsInstance(g, SPGGenerator)
        frames = g.generate_pop(5)
        for atoms in frames:
            self.assertIn(self.get_spg(atoms), np.arange(20, 31))

    def test_var_mol(self):
        NO3 = Atoms('NO3', positions=np.array([
            [2.012707, 2.014563, 4.870574],
            [1.714319, 0.953807, 5.478185],
            [2.311095, 3.075319, 5.478185],
            [2.012707, 2.014563, 3.582428]]))
        NH4 = Atoms('H4N', positions=np.array([
            [4.511281, 4.375470, 3.210227],
            [3.584655, 4.486488, 1.796710],
            [4.670180, 3.191076, 2.019142],
            [3.246077, 3.272899, 2.937012],
            [4.000271, 3.837356, 2.488938]]))
        para = {
            'formulaType': 'var',
            'minNAtoms': 60,
            'maxNAtoms': 80,
            'spacegroup': 56,
            'symbols': ['H','N','O'],
            'molMode': True,
            'inputMols': [NO3, NH4],
            'formula': [[1,0],[0,1]],
            'dRatio': 0.7,
            'volumeRatio': 10.,
        }
        g = magusParameters(para).RandomGenerator
        self.assertIsInstance(g, MoleculeSPGGenerator)
        frames = g.generate_pop(5)
        for atoms in frames:
            self.assertEqual(self.get_spg(atoms), 56)


if __name__ == '__main__':
    unittest.main()