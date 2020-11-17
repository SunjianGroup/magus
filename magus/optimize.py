from ase.optimize import BFGS, LBFGS, FIRE
from ase.atoms import Atoms


class MLFIRE(FIRE):
    def __init__(self, maxstd=0.5, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.maxstd = maxstd

    def converged(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()
        forces_converged = super().converged(forces)
        try:
            if isinstance(self.atoms, Atoms):
                atoms = self.atoms
            else:
                atoms = self.atoms.atoms
            Estd = atoms.calc.model.predict_energy(atoms, eval_std=True)[1]
            std_toobig = (Estd / len(atoms) > self.maxstd)
        except:
            raise Exception('MLFIRE calculator must have uncertainty')
        return forces_converged or std_toobig

