import itertools
from functools import reduce
import networkx as nx
import numpy as np
from math import gcd
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii

from packaging.version import parse as parse_version
OLD_NETWORKX = parse_version(nx.__version__) < parse_version("2.0")

def get_cycle_sums(G):
    """
    Return the cycle sums of the crystal quotient graph G.
    G: networkx.MultiGraph
    Return: a (Nx3) matrix
    """
    SG = nx.Graph(G) # Simple graph, maybe with loop.
    cycle_basis = nx.cycle_basis(SG)
    cycle_sums = []

    # add cycles only in the simple graph 
    for cycle in cycle_basis:
        cycle_sum = np.zeros([3])
        for i, j in zip(cycle, cycle[1:] + cycle[:1]):
            if i <= j:
                cycle_sum += SG[i][j]['vector']
            else:
                cycle_sum -= SG[i][j]['vector']
        cycle_sums.append(cycle_sum)
    
    # add cycles because of multi edges between two nodes
    for i, j in SG.edges():
        for e in G[i][j].values():
            cycle_sums.append(G[i][j][0]['vector'] - e['vector'])
    cycle_sums = np.unique(cycle_sums, axis=0)
    return cycle_sums


def get_dimension(G):
    """
    Return the dimensionality of the crystal quotient graph G.
    G: networkx.MultiGraph
    Return: int
    """
    cycle_sums = get_cycle_sums(G)
    return np.linalg.matrix_rank(cycle_sums)


def get_multiplicity(G):
    """
    Return the self-penetration multiplicities of the 3D crystal quotient graph G.
    G: networkx.MultiGraph
    Return: int
    """
    cycle_sums = get_cycle_sums(G)
    dimension = np.linalg.matrix_rank(cycle_sums)
    if dimension == 0:
        return 1
    # determinants
    min_multi = 100
    for comb in itertools.combinations(cycle_sums, dimension):
        comb = np.array(comb)
        det = [int(np.linalg.det(comb[:, j])) 
               for j in itertools.combinations(range(comb.shape[1]), dimension)]
        multi = abs(reduce(gcd, det))
        if 0 < multi < min_multi:
            min_multi = multi
    return min_multi


def remove_selfloops(G):
    newG = G.copy()
    if OLD_NETWORKX:
        loops = list(newG.selfloop_edges())
    else:
        loops = list(nx.selfloop_edges(newG))
    newG.remove_edges_from(loops)
    return newG


def find_communities(G):
    G = remove_selfloops(G)
    partition = []
    comp_queue = [G.subgraph(nodes) for nodes in nx.connected_components(G)]
    while len(comp_queue) > 0:
        c = comp_queue.pop()
        if get_dimension(c) == 0:
            partition.append(list(c.nodes()))
        else:
            comp = nx.algorithms.community.girvan_newman(c)
            for indices in next(comp):
                comp_queue.append(G.subgraph(indices))
    return partition


def get_nodes_and_offsets(G):
    assert nx.number_connected_components(G) == 1, "The graph should be connected!"
    offsets = []
    nodes = []
    paths = nx.single_source_shortest_path(G, list(G)[0])
    for node, path in paths.items():
        offset = np.zeros(3)
        for i, j in zip(path[:-1], path[1:]):
            if i <= j:
                offset += G[i][j][0]['vector']
            else:
                offset -= G[i][j][0]['vector']
        offsets.append(offset)
        nodes.append(node)
    return nodes, offsets


def atoms_to_quotient_graph(atoms, coef=1.1):
    """
    initialize crystal quotient graph of the atoms.
    atoms: (ASE.Atoms) the input crystal structure 
    coef: (float) the criterion for connecting two atoms. 
          If d_{AB} < coef*（r_A + r_B), atoms A and B are regarded as connected. 
          r_A and r_B are covalent radius of A,B.
    """
    cutoffs = [covalent_radii[number] * coef for number in atoms.get_atomic_numbers()]
    G = nx.MultiGraph()
    # add nodes
    G.add_nodes_from(np.arange(len(atoms)))
    # add edges
    for i, j, S in zip(*neighbor_list('ijS', atoms, cutoffs, max_nbins=10)):
        if i <= j:
            G.add_edge(i, j, vector=S)
    return G


# algorithm 1
def atoms_to_mol_1(atoms, coef=1.1):
    G = atoms_to_quotient_graph(atoms, coef)
    offsets = np.zeros([len(atoms), 3])
    tags = np.zeros(len(atoms)) - 1
    for nodes in nx.connected_components(G):
        comp = nx.subgraph(G, nodes)
        if get_dimension(comp) == 0 and comp.number_of_nodes() > 1:
            nodes_, offsets_ = get_nodes_and_offsets(comp)
            tags[nodes_] = np.max(tags) + 1
            offsets[nodes_] = offsets_
        else:
            for i in comp.nodes():
                tags[i] = np.max(tags) + 1
                offsets[i] = [0,0,0]
    return tags, offsets


# algorithm 2
def atoms_to_mol_2(atoms, coef=1.1):
    G = atoms_to_quotient_graph(atoms, coef)
    partition = find_communities(G)
    offsets = np.zeros([len(atoms), 3])
    tags = np.zeros(len(atoms)) - 1
    for nodes in partition:
        if len(nodes) > 1:
            comp = nx.subgraph(G, nodes)
            nodes_, offsets_ = get_nodes_and_offsets(comp)
            tags[nodes_] = np.max(tags) + 1
            offsets[nodes_] = offsets_
        else:
            tags[nodes[0]] = np.max(tags) + 1
            offsets[nodes[0]] = [0,0,0]
    return tags, offsets
