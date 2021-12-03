from ase.atoms import Atoms
from ase.data import atomic_numbers,covalent_radii,atomic_masses
import numpy as np
from math import cos, sin
from ..crystgraph import quotient_graph, cycle_sums, graph_dim, find_communities, find_communities2, find_communities4, remove_selfloops, nodes_and_offsets
import networkx as nx


def primitive_atoms2molcryst(atoms, coef=1.1):
    """
    Convert crystal to molecular crystal
    atoms: (ASE.Atoms) the input crystal structure
    coef: (float) the criterion for connecting two atoms
    Return: tags and offsets
    """
    QG = quotient_graph(atoms, coef)
    try:
        graphs = nx.connected_component_subgraphs(QG)
    except:
        graphs = [QG.subgraph(c) for c in nx.connected_components(QG)]
    partition = []
    offSets = np.zeros([len(atoms), 3])
    tags = np.zeros(len(atoms))
    for G in graphs:
        if graph_dim(G) == 0 and G.number_of_nodes() > 1:
            nodes, offs = nodes_and_offsets(G)
            partition.append(nodes)
            for i, offSet in zip(nodes, offs):
                offSets[i] = offSet
        else:
            for i in G.nodes():
                partition.append([i])
                offSets[i] = [0,0,0]

    for tag, p in enumerate(partition):
        for j in p:
            tags[j] = tag

    return tags, offSets


def atoms2communities(atoms, coef=1.1):
    """
    Split crystal to communities
    atoms: (ASE.Atoms) the input crystal structure
    coef: (float) the criterion for connecting two atoms
    Return: MolCryst
    """
    QG = quotient_graph(atoms, coef)
    #graphs = nx.connected_component_subgraphs(QG)
    try:
        graphs = nx.connected_component_subgraphs(QG)
    except:
        graphs = [QG.subgraph(c) for c in nx.connected_components(QG)]
    partition = []
    offSets = np.zeros([len(atoms), 3])
    for SG in graphs:
        G = remove_selfloops(SG)
        if graph_dim(G) == 0:
            nodes, offs = nodes_and_offsets(G)
            partition.append(nodes)
            for i, offSet in zip(nodes, offs):
                offSets[i] = offSet
        else:
            # comps = find_communities(G)
            comps = find_communities4(G)
            for indices in comps:
                tmpG = G.subgraph(indices)
                nodes, offs = nodes_and_offsets(tmpG)
                partition.append(nodes)
                for i, offSet in zip(nodes, offs):
                    offSets[i] = offSet

    # logging.debug("atoms2communities partition: {}".format(partition))

    molC = MolCryst(numbers=atoms.get_atomic_numbers(), cell=atoms.get_cell(),
    sclPos=atoms.get_scaled_positions(), partition=partition, offSets=offSets, info=atoms.info.copy())

    return molC

def primitive_atoms2communities(atoms, coef=1.1):
    """
    Split crystal to communities
    atoms: (ASE.Atoms) the input crystal structure
    coef: (float) the criterion for connecting two atoms
    Return: tags and offsets
    """
    QG = quotient_graph(atoms, coef)
    #graphs = nx.connected_component_subgraphs(QG)
    try:
        graphs = nx.connected_component_subgraphs(QG)
    except:
        graphs = [QG.subgraph(c) for c in nx.connected_components(QG)]
    partition = []
    offSets = np.zeros([len(atoms), 3])
    tags = np.zeros(len(atoms))
    for SG in graphs:
        G = remove_selfloops(SG)
        if graph_dim(G) == 0:
            nodes, offs = nodes_and_offsets(G)
            partition.append(nodes)
            for i, offSet in zip(nodes, offs):
                offSets[i] = offSet
        else:
            # comps = find_communities(G)
            comps = find_communities2(G)
            for indices in comps:
                tmpG = G.subgraph(indices)
                nodes, offs = nodes_and_offsets(tmpG)
                partition.append(nodes)
                for i, offSet in zip(nodes, offs):
                    offSets[i] = offSet

    # logging.debug("atoms2communities partition: {}".format(partition))

    for tag, p in enumerate(partition):
        for j in p:
           tags[j] = tag

    return tags, offSets


class Atomset:
    def __init__(self,positions,symbols):
        self.symbols = symbols
        self.position = np.mean(positions,axis=0)
        self.relative_positions = positions - self.position

    def __len__(self):
        return len(self.symbols)

    def to_atoms(self):
        return Atoms(symbols=self.symbols,positions=self.positions)

    def rotate(self, phi, theta, psi):
        rot1 = np.array([[cos(phi), -sin(phi), 0.], [sin(phi), cos(phi), 0.], [0., 0., 1.]])
        rot2 = np.array([[1., 0., 0.], [0., cos(theta), -sin(theta)], [0., sin(theta), cos(theta)]])
        rot3 = np.array([[cos(psi), -sin(psi), 0.], [sin(psi), cos(psi), 0.], [0., 0., 1.]])
        self.relative_positions = self.relative_positions @ rot1 @ rot2 @ rot3

    @property
    def positions(self):
        return self.position + self.relative_positions

    @property
    def symbol(self):
        s = []
        unique_symbols = sorted(np.unique(self.symbols))
        for symbol in unique_symbols:
            s.append(symbol)
            n = self.symbols.count(symbol)
            if n > 1:
                s.append(str(n))
        s = ''.join(s)
        return s

    @property
    def mass(self):
        return sum([atomic_masses[atomic_numbers[symbol]] for symbol in self.symbols])

    @property
    def number(self):
        numbers = [atomic_numbers[symbol] for symbol in self.symbols]
        radius = [covalent_radii[number] for number in numbers]
        return numbers[np.argmax(radius)]


class Molfilter:
    def __init__(self, atoms, detector=1, coef=1.1):
        self.pbc = atoms.pbc
        self.cell = atoms.cell
        self.mols = []
        if detector == 1:
            tags, offsets = primitive_atoms2molcryst(atoms, coef)
        elif detector == 2:
            tags, offsets = primitive_atoms2communities(atoms, coef)

        # add offsets
        positions = atoms.get_positions()
        positions += np.dot(offsets, self.cell)
        symbols = atoms.get_chemical_symbols()

        for tag in np.unique(tags):
            indices = np.where(tags == tag)[0]
            pos = [positions[i] for i in indices]
            sym = [symbols[i] for i in indices]
            self.mols.append(Atomset(pos, sym, tag))

    def __len__(self):
        return len(self.mols)

    def __iter__(self):
        for mol in self.mols:
            yield mol

    def __getitem__(self,i):
        return self.mols[i]

    def get_positions(self):
        return np.array([mol.position for mol in self.mols])

    def set_positions(self, positions):
        for i, mol in enumerate(self.mols):
            mol.position = positions[i]

    @property
    def positions(self):
        return self.get_positions()

    @positions.setter
    def positions(self, positions):
        self.set_positions(positions)

    def get_scaled_positions(self):
        return self.cell.scaled_positions(self.get_positions())

    def set_scaled_positions(self, scaled_positions):
        positions = np.dot(scaled_positions, self.atoms.cell)
        self.set_positions(positions)

    def append(self, mol):
        self.mols.append(mol)

    def to_atoms(self):
        positions = []
        symbols = []
        for mol in self.mols:
            symbols.extend(mol.symbols)
            positions.extend(mol.positions)
        atoms = Atoms(symbols=symbols, positions=positions, pbc=self.pbc, cell=self.cell)
        return atoms
