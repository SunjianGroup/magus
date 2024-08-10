import logging, yaml
from magus.calculators.base import ASECalculator
from magus.utils import CALCULATOR_PLUGIN
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes, PropertyNotImplementedError
from ase.neighborlist import neighbor_list


class MiaonetCalculator(Calculator):

    implemented_properties = [
        "energy",
        "energies",
        "forces",
        "stress",
        "dipole",
        "polarizability",
    ]

    def __init__(self,
                 model_file : str="model.pt",
                 device     : str="cpu",
                 **kwargs,
                 ) -> None:
        Calculator.__init__(self, **kwargs)
        self.device = device
        self.model = torch.jit.load(model_file, map_location=device).double()
        self.cutoff = float(self.model.cutoff.detach().cpu().numpy())

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        idx_i, idx_j, offsets = neighbor_list("ijS", atoms, self.cutoff, self_interaction=False)
        offset = np.array(offsets) @ atoms.get_cell()

        data = {
            "atomic_number": torch.tensor(atoms.numbers, dtype=torch.long, device=self.device),
            "idx_i"        : torch.tensor(idx_i, dtype=torch.long, device=self.device),
            "idx_j"        : torch.tensor(idx_j, dtype=torch.long, device=self.device),
            "coordinate"   : torch.tensor(atoms.positions, dtype=torch.double, device=self.device),
            "n_atoms"      : torch.tensor([len(atoms)], dtype=torch.long, device=self.device),
            "offset"       : torch.tensor(offset, dtype=torch.double, device=self.device),
            "scaling"      : torch.eye(3, dtype=torch.double, device=self.device).view(1, 3, 3),
            "batch"        : torch.zeros(len(atoms), dtype=torch.long, device=self.device),
        }

        self.model(data, properties, create_graph=False)
        print(data.keys())
        if "energy" in properties:
            self.results["energy"] = data["energy_p"].detach().cpu().numpy()[0]
        if "energies" in properties:
            self.results["energies"] = data["site_energy_p"].detach().cpu().numpy()
        if "forces" in properties:
            self.results["forces"] = data["forces_p"].detach().cpu().numpy()
        if "stress" in properties:
            virial = data["virial_p"].detach().cpu().numpy().reshape(-1)
            if sum(atoms.get_pbc()) > 0:
                stress = 0.5 * (virial.copy() + virial.copy().T) / atoms.get_volume()
                self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
            else:
                raise PropertyNotImplementedError
        if "dipole" in properties:
            self.results["dipole"] = data["dipole_p"].detach().cpu().numpy()
        if "polarizability" in properties:
            self.results["polarizability"] = data["polarizability_p"].detach().cpu().numpy()


@CALCULATOR_PLUGIN.register('miao')
class MiaoCalculator(ASECalculator):
    def __init__(self, **parameters):
        super().__init__(**parameters)
        self.relax_calc = MiaonetCalculator('model.pt')
        self.scf_calc = MiaonetCalculator('model.pt')
