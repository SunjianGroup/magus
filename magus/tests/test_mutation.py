import unittest, os
from ase.io import read
from magus.populations.individuals import Bulk
from magus.operations.mutations import *


class BulkMutation:
    def setUp(self):
        path = os.path.dirname(__file__)
        atoms = read(os.path.join(path, 'POSCARS/bulk_1.vasp'))
        Bulk.set_parameters(
            symbols=['Ti', 'O'],
            symbol_numlist_pool=[[16, 32]],
            symprec=0.1,
            fp_calc='zernike',
            comparator='naive')
        self.ind = Bulk(atoms)

    def test_mol_0(self):
        self.ind.mol_detector = 0
        ind = self.op.mutate(self.ind)
        self.assertIsNotNone(ind)
    
    def test_mol_1(self):
        self.ind.mol_detector = 1
        ind = self.op.mutate(self.ind)
        self.assertIsNotNone(ind)

    def test_mol_2(self):
        self.ind.mol_detector = 2
        ind = self.op.mutate(self.ind)
        self.assertIsNotNone(ind)


class TestBulkPermMutation(BulkMutation, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.op = PermMutation()


class TestBulkLatticeMutation(BulkMutation, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.op = LatticeMutation()


class TestBulkRippleMutation(BulkMutation, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.op = RippleMutation()


class TestBulkSlipMutation(BulkMutation, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.op = SlipMutation()


class TestBulkRattleMutation(BulkMutation, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.op = RattleMutation()


if __name__ == '__main__':
    unittest.main()