import unittest, os
from ase.io import read
from magus.populations.individuals import *
from magus.operations.crossovers import CutAndSplicePairing, ReplaceBallPairing


class BulkCrossover:
    def setUp(self):
        path = os.path.dirname(__file__)
        atoms1 = read(os.path.join(path, 'POSCARS/bulk_1.vasp'))
        atoms2 = read(os.path.join(path, 'POSCARS/bulk_2.vasp'))
        Bulk.set_parameters(
            symbols=['Ti', 'O'],
            symbol_numlist_pool=[[16, 32]],
            symprec=0.1,
            fp_calc='zernike',
            comparator='naive')
        self.ind1, self.ind2 = Bulk(atoms1), Bulk(atoms2)

    def test_mol_0(self):
        self.ind1.mol_detector = self.ind2.mol_detector = 0
        ind = self.op.cross(self.ind1, self.ind2)
        self.assertIsNotNone(ind)
    
    def test_mol_1(self):
        self.ind1.mol_detector = self.ind2.mol_detector = 1
        ind = self.op.cross(self.ind1, self.ind2)
        self.assertIsNotNone(ind)

    def test_mol_2(self):
        self.ind1.mol_detector = self.ind2.mol_detector = 2
        ind = self.op.cross(self.ind1, self.ind2)
        self.assertIsNotNone(ind)


class TestBulkCutAndSplicePairing(BulkCrossover, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.op = CutAndSplicePairing()


class TestBulkReplaceBallPairing(BulkCrossover, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.op = ReplaceBallPairing()


class LayerCrossover:
    def setUp(self):
        path = os.path.dirname(__file__)
        atoms1 = read(os.path.join(path, 'POSCARS/layer_1.vasp'))
        atoms2 = read(os.path.join(path, 'POSCARS/layer_2.vasp'))
        Layer.set_parameters(
            symbols=['C'],
            symbol_numlist_pool=[[12]],
            symprec=0.1,
            fp_calc='soap',
            comparator='naive')
        self.ind1, self.ind2 = Layer(atoms1), Layer(atoms2)

    def test_layer(self):
        ind = self.op.cross(self.ind1, self.ind2)
        self.assertIsNotNone(ind)


class TestBulkCutAndSplicePairing(BulkCrossover, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.op = CutAndSplicePairing()


if __name__ == '__main__':
    unittest.main()