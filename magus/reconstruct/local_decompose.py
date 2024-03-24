import time
import spglib
import ase.io
import numpy as np
from ase.neighborlist import neighbor_list
import networkx as nx
from collections import Counter
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import GeneticOrderMatcher
from packaging.version import parse as parse_version
OLD_NETWORKX = parse_version(nx.__version__) < parse_version("2.0")
import itertools
from magus.utils import get_distance_dict
from ase import Atoms
from ase.data import covalent_radii
import math
from scipy.spatial import ConvexHull

def Molecule_to_Atoms(mol, info = {}):
   '''
   Molecule class by pymatgen to Atoms class by Ase
   '''
   a = Atoms(numbers = mol.atomic_numbers, positions = mol.cart_coords)
   a.info = info
   return a
def Atoms_to_Molecule(atoms):
   '''
   Atoms class by Ase to Molecule class by pymatgen
   '''
   return Molecule(atoms.symbols, atoms.get_positions())


class CrystalGraph(nx.Graph):

   def __init__(self):
      super().__init__()
      self._ubc = None
      self._dof = None

   def __str__(self):
      return "<frag with {} atoms, ".format(len(self)) + \
            "".join(["{}: {}; ".format(key, self.info.get(key, "NA")) for key in self.info]) + ">"

   @classmethod
   def set_standards(cls, **kwargs):
      cls.n_community = kwargs.get("n_community", [3,12])
      cls.prefered_ubc = kwargs.get("prefered_ubc", 2)
      cls.prefered_dof = kwargs.get("prefered_dof", 3)
      cls.prefer_special = kwargs.get("prefer_special", True)
      cls.minimal_density = kwargs.get("minimal_density", 0)

   @property
   def info(self):
      return  {'ubc':self.ubc, 
               'dof':self.dof,
               'origin': self.origin,
               'config_type': self.config_type,
               'dimension': self.dimension,
               'density': self.density}
   
   def __eq__(self, CG2):
      if not len(self) == len(CG2):
         return False
      m1 = self.output_mol()
      m2 = CG2.output_mol()
      f1 = Counter(m1.atomic_numbers)
      f2 = Counter(m2.atomic_numbers)
      for s in f1:
         if not f2.get(s, 0) == f1[s]:
            return False
      m = GeneticOrderMatcher(m1,0.1).fit(m2)
      if len(m) > 0:
         if np.min([x[1] for x in m]) < 0.1:
               return True
      return False
   

   def input_atoms(self, atoms, distance_dict = None):
      """
      initialize crystal quotient graph of the atoms.
      atoms: (ASE.Atoms) the input crystal structure 
      distance_dict: (dictionary) This specifies cutoff values for element pairs. 
                        Specification accepts element numbers of symbols. 
                        Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
      """
      if distance_dict is None:
         distance_dict = [covalent_radii[number] * 1.1 for number in atoms.get_atomic_numbers()]

      # add nodes
      for i in range(0,len(atoms)):
         self.add_node(i, atomic_symbol =atoms[i].symbol, position = atoms[i].position )
      # add edges
      for i, j, D in zip(*neighbor_list('ijD', atoms, distance_dict, max_nbins=10)):
         if i < j:
            self.add_edge(i, j, vector=D, weight = 1.0)  #weight = 1/np.linalg.norm(D))
            #self.add_edge(j, i, vector=-D, weight = 1.0)  #weight = 1/np.linalg.norm(D))

   def output_mol(self):
      return Molecule([self.nodes[i]['atomic_symbol'] for i in self.nodes], 
                        self.positions)

   def output_atoms(self):
      a = Atoms([self.nodes[i]['atomic_symbol'] for i in self.nodes], 
               self.positions)
      a.info['ubc'] = self.ubc
      a.info['dof'] = self.dof
      a.info['origin'] = self.origin
      a.info['config_type'] = self.config_type
      a.info['dimension'] = self.dimension
      a.info['density'] = self.density
      return a

   @property
   def ubc(self):
      if self._ubc is None:
         if len(self) > 1:
            self._ubc = self.rank_ubc()
         else:
            self._ubc = -1
      return self._ubc

   @property
   def dof(self):
      if self._dof is None:
         self._dof = -1
         if len(self) >2:
            try:
               self._dof = self.rank_dof(0.5)
            except:
               pass
            
      return self._dof
   
   @property
   def origin(self):
      if hasattr(self, "_origin"):
         return self._origin
      else:
         return "ukn"
   
   @origin.setter
   def origin(self, value):
      self._origin = value
   
   @property
   def config_type(self):
      if hasattr(self, "_config_type"):
         return self._config_type
      else:
         return "n"
   
   @config_type.setter
   def config_type(self, value):
      self._config_type = value

   def remove_nodes_from(self, *arg):
      super().remove_nodes_from(*arg)
      self._dof = None
      self._ubc = None
   
   def remove_edge(self, *arg):
      super().remove_edge(*arg)
      self._dof = None
      self._ubc = None
   def remove_edges_from(self, *arg):
      super().remove_edges_from(*arg)
      self._dof = None
      self._ubc = None
   
   def subgraph(self, nodes):
      self._dof = None
      self._ubc = None
      return super().subgraph(nodes)

   def rank_ubc(self):
      centrality = nx.betweenness_centrality(self, endpoints=False, normalized=True, weight='weight')
      # 0. rank uniqueness
      uniqueness = len(list(set([np.round(centrality[node],4) for node in centrality])))

      if 0 in list(centrality.values()):
         #*Remove chains
         for i in self.nodes:
            if len(self.edges(i)) >2:
               break
         else:
            if not np.all(np.array(list(centrality.values())) == 0):
               return 100

      return uniqueness

   def rank_mbc(self):
      centrality = nx.betweenness_centrality(self, endpoints=False, normalized=True, weight='weight')
      mean_centrality = np.average([centrality[node] for node in centrality])
      return mean_centrality

   def rank_dof(self, symprec = 0.5):
      mol = self.output_mol()
      eq = PointGroupAnalyzer(mol, symprec).get_equivalent_atoms()['eq_sets']
      n_inequivalent_atoms = len(eq.keys())
      if n_inequivalent_atoms>2:
         return 3*n_inequivalent_atoms-6
      elif n_inequivalent_atoms==2:
         return 1
      elif n_inequivalent_atoms==1:
         return 0

   def exclude_disconnected(self, central, neighbor = 4):
      connected = []
      for node in self.nodes:
         try:
            if nx.shortest_path_length(self, source=node,target=central)<= neighbor:
               connected.append(node)
         except nx.NetworkXNoPath:
            continue

      return self.subgraph(connected)
   
   def cycle_length(self, source, target, output_path = False):
      try:
         L1 = nx.shortest_path(self, source, target)
      except nx.NetworkXNoPath:
         return 3000

      H = self.copy()
      length1 = len(L1) -1
      
      H.remove_edges_from([x for x in itertools.combinations(L1,2)])
      try:
         L2 = nx.shortest_path(H, source, target)
      except nx.NetworkXNoPath:
         return 3000
      length2 = len(L2) -1
      #print(source, target, length1 + length2)
      return length1 + length2
   
   def have_cycle(self):
      try:
         nx.find_cycle(self)
      except nx.NetworkXNoCycle:
         return False
      return True
   
   def have_open_end(self):
      for node in self.nodes:
         if len(self.edges(node)) < 2:
            return True
      return False
   

   def exclude_openend(self, central):
      have_open_end = True
      H_prime = self.copy()
      while have_open_end:
         for node in H_prime.nodes:
            if len(H_prime.edges(node)) < 2:
               if central == node:
                  return H_prime.subgraph([central])
               H_prime = H_prime.subgraph([x for x in H_prime.nodes if not x==node])
               break
         else:  
            have_open_end = False

      return H_prime

   def get_distance(self, i, j):
      '''
      edges = nx.shortest_path(self, source=i, target=j)
      vector = np.array([0,0,0], dtype = 'float64')
      for i in range(0,len(edges)-1):
         vector += np.array(self[edges[i]][edges[i+1]]['vector'])
      return vector
      '''

      return self.nodes[j]['position'] - self.nodes[i]['position']
   
   @property
   def bounding_sphere(self):
      positions = self.positions
      center = np.mean(positions, axis=0)
      return math.sqrt(np.max([np.sum([x**2 for x in (p - center)]) for p in positions]))
   
   @property
   def positions(self):
      return [self.nodes[node]['position'] for node in self.nodes]

   @property
   def dimension(self):
      positions = self.positions
      if len(positions) == 1:
         return 0
      elif np.max(np.array([len(self.edges(i)) for i in self.nodes])) <3 and self.have_open_end():
         return 1
      else:
         positions = positions[1:] - positions[0]
      return np.linalg.matrix_rank(positions)

   @property
   def density(self):
      try:
         edges = len(self.edges)
         volume = ConvexHull(self.positions).volume
         return np.round(edges / volume,4)
      except:
         return 10000
   
   def acceptable(self):
      if len(self) < self.n_community[0] or len(self) > self.n_community[1]:
         return False
      if self.dimension <2:
         return False
      #if self.prefer_special: #and self.config_type == '1st_neighbor':
      #   return True
      if self.ubc > self.prefered_ubc:
         return False
      if self.dof > self.prefered_dof:
         return False
      if self.density < self.minimal_density:
         return False

      return True
   
   def betweenness_rank(self, source, target):
      centrality = nx.betweenness_centrality(self, endpoints=False, normalized=False,weight='weight')
      if centrality[target] == 0:
         return 3000
      return self.cycle_length(source, target)
   

def CGIO_read(file, index = 0):
   frags = ase.io.read(file, index=index)
   for i, f in frags:
      frags[i] = CrystalGraph(f)
      frags[i].info = f.info
   return frags

def CG_ISOLATE_ATOM(symbols):
   frags = []
   for s in symbols:
      f = Atoms(symbols = [s], positions = [[0,0,0]]) 
      
      c = CrystalGraph()
      c.input_atoms(f)
      c.origin= "-1"
      c.config_type="isolate_atom"
      frags.append(c)
   return frags

def CGIO_write(file, cgs):
   frags = list(map(lambda x:x.output_atoms(), cgs))
   ase.io.write(file, frags)

def distance(vector1, vector2):
   return np.sqrt(np.sum([v**2 for v in (vector1 - vector2)]))

def build_neighbor_struct(origin_struct, center_index, furthest_cutoff=5):
   ori_pos = origin_struct[center_index].position

   ori_pos_all = np.dot(origin_struct.get_scaled_positions(wrap = True), origin_struct.get_cell())

   #pos_all = np.array([ori_pos_all + np.dot(list(offset), origin_struct.cell) for offset in itertools.product(*[range(-int(i),int(i)) for i in np.abs(np.ceil(furthest_cutoff/origin_struct.cell.cellpar()[:3]))])])
   pos_all = np.array([ori_pos_all])
   numbers = np.array([origin_struct.numbers for offset in itertools.product(*[range(-int(i),int(i)) for i in np.abs(np.ceil(furthest_cutoff/origin_struct.cell.cellpar()[:3]))])])
   
   pos_all = pos_all.reshape([-1,3])
   numbers = numbers.flatten()

   index = []
   for i,p in enumerate(pos_all):
      ds = distance(ori_pos, p)
      if ds < furthest_cutoff:
         if ds < 1e-2:
            new_central = len(index)
         index.append(i)
         
   center_struct = Atoms(cell = [furthest_cutoff*3]*3, positions = pos_all[index], numbers = numbers[index])
   center_struct.positions += np.sum(center_struct.cell, axis = 0) /2  - np.mean(center_struct.positions,axis = 0)

   return center_struct, new_central


def degree(v1, v2):
   x = np.dot(v1,v2) / np.linalg.norm(v1) /np.linalg.norm(v2)

   if x >1:
      x =1
   elif x<-1:
      x=-1
   
   return np.round(math.acos(x) / math.pi * 180)

def have_sym(vectors1, sym):

   if not len(vectors1) == sym:
      return False
   
   ds = [np.linalg.norm(v) for v in vectors1]
   if np.max(ds) - np.min(ds) > 1e-2:
      return False
   if len(ds) == 2:
      return True

   vectors = np.array([v for v in vectors1])
   vectors -= np.mean(vectors, axis = 0)
   

   ds = [degree(vectors[i], vectors[0]) for i in range(1,len(vectors))]
   ref = np.arange(-180, 181, int(180/sym))
   #print(ds, ref)
   if np.all(np.array([d in ref for d in ds])):
      return True
   return False


def special_frag(G, central, minimal_n_community=3, rank_list = []):
   try:
      H = G.subgraph([node for node in G.nodes if nx.shortest_path_length(G, node, central) <2])
      H.config_type = "1st_neighbor"
      if H.acceptable():
         rank_list.append(H)
   except:
      pass
   #H = G.subgraph([node for node in G.nodes if nx.shortest_path_length(G, node, central) <3])
   #H.config_type = "2st_neighbor"
   #if H.acceptable():
   #   rank_list.append(H)

   try:
      H = G.subgraph(list(set(np.array(nx.find_cycle(G)).flatten())))
      H.config_type = "1st_cycle"
      if H.acceptable():
         rank_list.append(H)
      
   except nx.NetworkXNoCycle:
      pass

def decompose1(G, central, minimal_n_community=3, rank_list = []):
   
   #ase.io.write("step{}.xyz".format(len(rank_list)), G.output_atoms())
   if G.acceptable():
      rank_list.append(G)
   #print([rl[0].nodes for rl in rank_list])

   H = G.exclude_openend(central)
   #H = H.exclude_long_cycle(central)
   if H.acceptable():
      rank_list.append(H)
   if len(H) < minimal_n_community:
      return 

   centrality = {node:len(H.edges(node)) for node in H.nodes}

   sorted_centrality = sorted(list(set([np.round(x,4) for x in centrality.values()])), reverse = False)
   to_rm = [node for node in centrality if np.round(centrality[node],4)==sorted_centrality[0]]
   
   if central in to_rm:
      to_rm.remove(central)
   if len(to_rm) ==0:
      to_rm = [node for node in centrality if np.round(centrality[node],4)==sorted_centrality[1]]
   #print("step 1 ", to_rm)

   neighbor_vectors = {i: H.get_distance(central,i) for i in to_rm}
   neighbor_distance = {i: np.round(np.linalg.norm(neighbor_vectors[i]),4) for i in to_rm}
   snd = sorted(neighbor_distance.items(), key=lambda x:x[1])
   #print("distance", snd)
   maximum_distance = snd[-1][1]

   minimal_centrality = []

   furtherest_neighbor = [i for i in neighbor_distance.keys() if neighbor_distance[i]==maximum_distance]
   #print(maximum_distance, furtherest_neighbor)
   """
   for i in range(len(furtherest_neighbor), 1, -1):
      for L in itertools.combinations(furtherest_neighbor, i):
         if have_sym([neighbor_vectors[node] for node in L],i) or centrality[L[0]]>=1000:
            #print("have sym", L)
            minimal_centrality = list(L)
            break

            
      if len(minimal_centrality) > 0:
         break
   
   else:
      minimal_centrality = [furtherest_neighbor[0]]
   """
   minimal_centrality = furtherest_neighbor

   rm_schemes = [minimal_centrality]
  
   #if (not len(rm_schemes[0]) == 1) :
   #   #rm_schemes.extend([[i] for i in furtherest_neighbor])
   #   rm_schemes.extend([[i] for i in furtherest_neighbor[:1]])
   
   #print(rm_schemes)

   for s in rm_schemes:
      H_prime = H.copy()
      H_prime.remove_nodes_from(s)
      H_prime = H_prime.exclude_disconnected(central)
      #print('got H prime', H_prime.nodes, minimal_n_community)
      if len(H_prime) < minimal_n_community:
         return
      else:
         decompose1(H_prime, central, minimal_n_community = minimal_n_community, rank_list = rank_list)

   

def ranker_central(origin_struct, label):
   return distance(np.sum(origin_struct.cell, axis = 0)/2, origin_struct[label].position)

def is_same_graph(CG1, CG2):
   if not len(CG1) == len(CG2):
      return False
   if not CG1.ubc == CG2.ubc:
      return False
   if not np.all(np.array(sorted(list(CG1.nodes))) == np.array(sorted(list(CG2.nodes)))):
      return False
   return True
    

def decompose(origin_struct, center_index, distance_dict, neighbor_dis=5, path_length_cut = 3, cycle_length_cut = 7,
               n_community=[3, 12], prefered_ubc = 2, prefered_dof = 3, prefered_bs = 4):

   atoms, central = build_neighbor_struct(origin_struct, center_index, neighbor_dis)
   G = CrystalGraph()
   G.input_atoms(atoms, distance_dict)

   G.set_standards(n_community = n_community, prefered_ubc = prefered_ubc, prefered_dof = prefered_dof,
                   prefer_special = True, minimal_density = 0)
   G_connect = G.exclude_disconnected(central, path_length_cut)
   #G_connect = G_connect.exclude_long_cycle(central, cycle_length_cut)
   #ase.io.write('centerstruct{}.xyz'.format(center_index),G_connect.output_atoms())
   G_connect = G.exclude_openend(central)
   
   #ase.io.write('close_cycle{}.xyz'.format(center_index),G_connect.output_atoms())

   #rank_list: graph, uniqueness, mean_centrality 
   rank_list = []
   special_frag(G_connect, central, minimal_n_community=n_community[0], rank_list=rank_list)
   decompose1(G_connect, central, minimal_n_community=n_community[0], rank_list=rank_list)
   #print(time.ctime(), ": remove duplicate graph")

   rank_list = sorted(rank_list, key = lambda x:x.ubc)
   unique_rl = []
   for rl in rank_list:
      for _rl in unique_rl:
         if is_same_graph(rl, _rl):
            break
      else:
         unique_rl.append(rl)
   
   decomposed_pop = []
   #print(time.ctime(), ": calculate rank info duplicate graph")
   for j,rl in enumerate(unique_rl):
      rl.origin = origin_struct.info['identity'] + ": " + str(j)
      decomposed_pop.append(rl)

   return decomposed_pop  


def DECOMPOSE(pop, distance_dict, **kwargs):
   decomposed_pop = []
   for atoms in pop:
      identity = atoms.info.get("identity", 'unknown')
      std_para = spglib.standardize_cell((atoms.cell, atoms.get_scaled_positions(), atoms.numbers), symprec=0.1, to_primitive=False)
      atoms = Atoms(cell=std_para[0], scaled_positions=std_para[1], numbers=std_para[2], pbc = True)
      atoms.info['identity'] =  identity
      #ase.io.write('std.vasp', atoms, vasp5=1)
      
      unique_atoms = list(set(spglib.get_symmetry_dataset((atoms.cell, atoms.get_scaled_positions(), atoms.numbers),0.1)['equivalent_atoms']))
      #print("unique atoms", unique_atoms)
      #print(time.ctime(), ": decompose structure")
      if len(unique_atoms) > len(atoms) / 10:
         kwargs['path_length_cut'] = 2
      if len(unique_atoms) == len(atoms):
         continue
      for i in unique_atoms:
         #try:
         p = decompose(atoms,i, distance_dict, **kwargs)
         for cg1 in p:
            for cg2 in decomposed_pop:
               if cg1 == cg2:
                  break
            else:
               decomposed_pop.append(cg1)
         #except Exception as e:
         #   print("Exception {} happened. return null set frags".format(e))
         #print(time.ctime(), ": decompose base on '{} (no. {})'".format(atoms[i].symbol, i))
         
   #print("decomposed", [(ind.ubc, ind.dof, len(ind)) for ind in decomposed_pop])
   return decomposed_pop


def is_same_frag(a,b):
   if isinstance(a, Atoms):
      acg = CrystalGraph()
      acg.input_atoms(a)
   else:
      acg = a
   if isinstance(b, Atoms):

      bcg = CrystalGraph()
      bcg.input_atoms(b)
   else:
      bcg = b

   if acg==bcg:
      return True
   else:
      return False
    

if __name__ == '__main__':
   distance_dict = {('B', 'B'): 1.848,}

   atoms = [ase.io.read('B_Pnnm_58_88.321077.vasp')] 
   decomposed_pop = DECOMPOSE(atoms, distance_dict, neighbor_dis=5, path_length_cut = 4, minimal_n_community=3)
   decomposed_pop = list(map(lambda x: x.output_atoms(), decomposed_pop))
   for i,mol in enumerate(decomposed_pop):
      ase.io.write('r{}_i{}_n{}_d{}_{}.xyz'.format(mol.info['ubc'], mol.info['dof'], len(mol),mol.info['density'], i), mol)
   
