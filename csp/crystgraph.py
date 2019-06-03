## Crystal Quotient Graph
from __future__ import print_function, division
from functools import reduce
from ase.neighborlist import NeighborList
from ase.data import covalent_radii
import ase.io
import networkx as nx
import numpy as np
import sys, itertools


def quotient_graph(atoms, coefficient=1.1, ):
    """Return crystal quotient graph of the atoms."""
    cutoffs =  [covalent_radii[number]*coefficient for number in atoms.get_atomic_numbers()]
    # print("cutoffs: %s" %(cutoffs))

    nl = NeighborList(cutoffs, skin=0, self_interaction=True, bothways=True)
    nl.update(atoms)

    G = nx.MultiGraph()

    for i, atom in enumerate(atoms):
        G.add_node(i,)
        # G[i]['symbol'] = atom.symbol
        indices, offsets = nl.get_neighbors(i)
        newIndices = []
        newOffsets = []
        for index, vector in zip(list(indices), list(offsets)):
            if index != i or (vector != np.zeros([1, 3])).any():
                newIndices.append(index)
                newOffsets.append(vector)

        for index, vector in zip(newIndices, newOffsets):
            if i <= index:
                G.add_edge(i, index, vector=vector, direction=(i, index))

    return G

def cycle_sums(G):
    """Return the cycle sums of the crystal quotient graph G."""
    SG = nx.Graph(G) # Simple graph, maybe with loop.
    cycBasis = nx.cycle_basis(SG)
    # print(cycBasis)
    cycleSums = []

    for cyc in cycBasis:
        cycSum = np.zeros([3])
        for i in range(len(cyc)):
            vector = SG[cyc[i-1]][cyc[i]]['vector']
            direction = SG[cyc[i-1]][cyc[i]]['direction']
            cycDi = (cyc[i-1], cyc[i])
            if cycDi == direction:
                cycSum += vector
                # cycSum += SG[cyc[i-1]][cyc[i]]['vector']
                # print("path->: %s, vector: %s" %((cyc[i-1], cyc[i]), SG[cyc[i-1]][cyc[i]]['vector']))

            elif cycDi[::-1] == direction:
                cycSum -= vector
                # cycSum -= SG[cyc[i-1]][cyc[i]]['vector']
                # print("path<-: %s, vector: %s" %((cyc[i-1], cyc[i]), SG[cyc[i-1]][cyc[i]]['vector']))

            else:
                raise RuntimeError("Error in direction!")
                # print("Error in direction!")
        cycleSums.append(cycSum)

    for edge in SG.edges():
        numEdges = list(G.edges()).count(edge)
        if numEdges > 1:
            direction0 = G[edge[0]][edge[1]][0]['direction']
            vector0 = G[edge[0]][edge[1]][0]['vector']
            for j in range(1, numEdges):
                directionJ = G[edge[0]][edge[1]][j]['direction']
                vectorJ = G[edge[0]][edge[1]][j]['vector']
                # cycSum = G[edge[0]][edge[1]][0]['vector'] - G[edge[0]][edge[1]][j]['vector']
                if direction0 == directionJ:
                    cycSum = vector0 - vectorJ
                elif direction0[::-1] == directionJ:
                    cycSum = vector0 + vectorJ
                else:
                    raise RuntimeError("Error in direction!")
                    # print("Error in direction!")
                cycleSums.append(cycSum)

    return np.array(cycleSums)

def graphDim(G):
    return np.linalg.matrix_rank(cycle_sums(G))

def getMut_3D(cycSums):
    assert np.linalg.matrix_rank(cycSums) == 3
    csArr = cycSums.tolist()

    # Basic cycle sums
    noZeroSum = []
    for cs in csArr:
        cst = tuple(cs)
        if cst != (0,0,0):
            if cst[0] < 0:
                cst = tuple([-1*i for i in cst])
            noZeroSum.append(cst)
    noZeroSum = list(set(noZeroSum))
    basicLen = len(noZeroSum)
    noZeroMat = np.array(noZeroSum)

    # determinants
    allDets = []
    for comb in itertools.combinations(range(basicLen), 3):
        mat = noZeroMat[list(comb)]
        det = abs(np.linalg.det(mat))
        if abs(det) > 1e-3:
            allDets.append(det)
    minDet = min(allDets)
    return minDet

def find_communities(QG):
    tmpG = remove_selfloops(QG)
    comp=nx.algorithms.community.girvan_newman(tmpG)
    for c in comp:
        SGs = [tmpG.subgraph(indices) for indices in c]
        dims = [np.linalg.matrix_rank(cycle_sums(SG)) for SG in SGs]
        sumDim = sum(dims)
        if sumDim == 0:
            break

    partition = [list(p) for p in c]
    return partition

def find_communities2(QG, maxStep=1000):
    tmpG = remove_selfloops(QG)
    partition = []
    for i in range(maxStep):
        comp=nx.algorithms.community.girvan_newman(tmpG)
        for c in comp:
            extendList = []
            sumDim = 0
            dims = []
            # print("c: {}".format(sorted(c)))
            for indices in c:
                SG = tmpG.subgraph(indices)
                dim = np.linalg.matrix_rank(cycle_sums(SG))
                sumDim += dim
                dims.append(dim)
                if dim == 0:
                    partition.append(list(indices))
                else:
                    extendList.append(tmpG.subgraph(indices))
            if sumDim == 0:
                return partition
            # print("dims:{}".format(dims))

            if len(extendList) > 0:
                tmpG = reduce(nx.union, extendList)
                break

def remove_selfloops(G):
    newG = G.copy()
    loops = list(newG.selfloop_edges())
    newG.remove_edges_from(loops)
    return newG

def nodes_and_offsets(G):
    offSets = []
    nodes = list(G.nodes())
    paths = nx.single_source_shortest_path(G, nodes[0])
    for index, i in enumerate(nodes):
        if index is 0:
            offSets.append([0,0,0])
        else:
            path = paths[i]
            offSet = np.zeros((3,))
            for j in range(len(path)-1):
                # print(j)
                vector = G[path[j]][path[j+1]][0]['vector']
                direction = G[path[j]][path[j+1]][0]['direction']
                pathDi = (path[j], path[j+1])
                if pathDi == direction:
                    offSet += vector
                elif pathDi[::-1] == direction:
                    offSet -= vector
                else:
                    raise RuntimeError("Error in direction!")
            offSets.append(offSet.tolist())
    return nodes, offSets


