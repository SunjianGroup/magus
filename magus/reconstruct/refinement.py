import numpy as np
from sklearn import cluster
import logging 


def _matrix_update(matrix, index, line):
    new_matrix = matrix.copy()
    new_matrix[index] = line
    new_matrix[:, index] = line
    return new_matrix

def distance(vector1, vector2):
    return np.sqrt(np.sum([v**2 for v in (vector1 - vector2)]))
    
def distance_matrix(atoms):
    matrix = np.zeros(shape=(len(atoms), len(atoms)))
    for i in range(0,len(atoms)):
        for j in range(0,i):
            matrix[j][i] = distance(atoms[i].position, atoms[j].position)
            matrix[i][j] = matrix[j][i]
    return np.round(matrix, decimals=2)

def update_distance_matrix(matrix, atoms, index):
    line = np.array([distance(atoms[i].position, atoms[index].position) for i in range(0, len(atoms))])
    new_matrix = _matrix_update(matrix, index, line)
    return new_matrix

def connection_map(matrix, nb_range):
    connection_map = np.zeros(shape = (len(matrix), len(matrix)))
    for i in range(0, len(matrix)):
        for j in range(0, i):
            if nb_range[0] <= matrix[i][j] <= nb_range[1]:
                connection_map[i][j] = connection_map[j][i] = 1
    return connection_map

def update_connection_map(connection_map, matrix, nb_range, index):
    line = np.array([nb_range[0]<=m<=nb_range[1] for m in matrix[index]])
    new_map = _matrix_update(connection_map, index, line)
    return new_map

def coordination_number(connection_map):
    return np.sum(connection_map, axis=1)


def sort_atoms_by_distance(atoms):
    center = np.average(atoms.positions, axis = 0)
    distances_to_center = [distance(p, center) for p in atoms.positions]
    newatoms = atoms[np.argsort(distances_to_center)]
    return newatoms    


from sklearn import cluster

class best_cluster:
    def __init__(self, threshold = 0.01):
        self.threshold = threshold

    @staticmethod
    def cluster(matrix, n):
        kmeans = cluster.KMeans(n_clusters=n).fit(matrix)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        inertia = (kmeans.inertia_)/len(matrix)
        return centers, labels, inertia


    def best_cluster(self, matrix, start = 2, logout = False):
        res_array, label_array, ine_array, delta_ine = [], [], [], []
        for n in range(1, 50):
            if n < start:
                res_array.append(None)
                label_array.append(None)
                ine_array.append(1e+5)
                delta_ine.append(1e+5)
                continue

            res, label, inertia = self.cluster(matrix, n)
            res_array.append(res)
            label_array.append(label)
            ine_array.append(inertia)
            delta_ine.append((ine_array[-2] - inertia)/ine_array[-2])
            if logout:
                print("Into {} clusters, inertia = {}, delta_ine = {}".format(n, inertia, delta_ine))
                print("centers = ", res)
            if delta_ine[-1] < self.threshold:
                break

        if logout:
            print("inertia array ", ine_array)
            print("delta inertia array ", delta_ine)
        return res_array[-1], label_array[-1]


def clusterDM(DM, start = 5, delta_bonds = 0.2, logout = False):
    matrix = [[p,0] for p in sorted(set(DM.flatten()))]

    for n in range(1, 50):
        if n < start:
            continue
        centers, _, _ = best_cluster.cluster(matrix, n)
            
        neighbor_orders = centers[:,0]
        neighbor_orders.sort()

        if logout:
            print("Into {} clusters".format(n))
            print("bonds_order", neighbor_orders) 

        if neighbor_orders[0] < 1e-5:
            b1, b2 = neighbor_orders[1:3]
        else:
            b1, b2 = neighbor_orders[0:2]
        #print(b2, b1, b2/b1)    
        if b2 / b1  < 1.0 + delta_bonds:
            break

    return neighbor_orders


def Coordination_info(atoms, threshold = 0.2, logout = False):
    DM = distance_matrix(atoms)
    C_DM = clusterDM(DM, delta_bonds = threshold, logout = logout)
    nb_range = [C_DM[1] - 0.05, C_DM[2]+0.05 ] 
    
    CM = connection_map(DM, nb_range)
    CN = coordination_number(CM)
    nb_number = [np.min(CN), np.max(CN)]
    return nb_range, nb_number

"""
modify individuals with bond lengths and coordination number. 

info LJ 75
nearest neighbor ~ 1.1A                 coordination number 6~12
second nearest neighbor ~ 1.5A     coordination number 0~6
"""

class Refinement:
    def __init__(self, atoms, first_neighbor = [(1.0,1.5),(6,12)],  second_neighbor = [(1.5,1.6),(0,6)]):
        self.nb_range_1, self.nb_num_1 = first_neighbor
        self.nb_range_2, self.nb_num_2 = second_neighbor
        self.atoms = sort_atoms_by_distance(atoms)
        self.fitness()

    @staticmethod
    def atomic_fitness(number, nb_num):
        if nb_num[0] <= number <= nb_num[1]:
            return 0
        elif number < nb_num[0]:
            return number - nb_num[0]
        else:       #number > nb_num[1]
            return  number - nb_num[1]
        

    @staticmethod
    def basic_property(atoms, nb_range, nb_num):
        matrix = distance_matrix(atoms)
        map = connection_map(matrix, nb_range)
        current_nb_num = coordination_number(map)
        fit_array = np.array([Refinement.atomic_fitness(num, nb_num) for num in current_nb_num])
        ind_fitness = np.average(abs(fit_array))

        return matrix, map, current_nb_num, fit_array, ind_fitness


    def fitness(self):
        self.nb_range = self.nb_range_1
        self.nb_num = self.nb_num_1

        self.matrix, self.map, self.current_nb_num, self.fit_array, self.ind_fitness = \
                    Refinement.basic_property(self.atoms, self.nb_range, self.nb_num)

    @staticmethod
    def where_equals(array_list, value):
        index = np.where(array_list == value)[0]
        return np.random.choice(index)
    
    """
    priority: 
    |-------------------|---------------------------|--------------------|
        less          nb_num[min]                nb_num[max]            more

    Remove: 
    A. Have more-neighbored:
        Among his neighbors, delete one with minimum coordination number
    B. Not have more-neighbored:
        Delete one with minimum coordination number

    Add:
    A. Have less-neighbored:
        Among them add to the one with maximum coordination number
    B. Not have less-neighbored:
        Add to whose coordination number in range nb_num[min]~nb_num[max]-1
    """
    def one_step(self): 
        priority_array = np.array(sorted(self.fit_array))

        if priority_array[-1] > 0:
            #Remove - A
            more_neighbored = Refinement.where_equals(self.fit_array, priority_array[-1])
            rm_candidates = np.where(self.map[more_neighbored] == 1)[0]
            to_rm = self.current_nb_num[rm_candidates].argsort()[0]
        else:
            #Remove - B
            to_rm = Refinement.where_equals(self.fit_array, priority_array[0])

        if priority_array[0] < 0 :
            #Add - A
            add_candidates = np.where(priority_array < 0)[0]
            to_add = Refinement.where_equals(self.fit_array, np.max(priority_array[add_candidates]))
        else:
            #Add - B
            add_candidates = np.where(self.current_nb_num < self.nb_num[1])[0]
            to_add = np.random.choice(add_candidates)

        for _ in range(0, 100):

            newatoms = self.atoms.copy()
            r, theta, phi = np.random.uniform(self.nb_range[0], self.nb_range[1]), \
                                    np.random.uniform(0, np.pi), \
                                    np.random.uniform(0, 2*np.pi)
            newatoms[to_rm].position = newatoms[to_add].position + r * np.array([np.sin(theta) * np.cos(phi), 
                                                                            np.sin(theta) * np.sin(phi),
                                                                            np.cos(theta)])
        
            new_dm =  update_distance_matrix(self.matrix, newatoms, to_rm)

            if np.all([d ==0 or d > self.nb_range_1[0] for d in new_dm[to_rm]] ):
                return newatoms
            
        return None
    
    def generator_main(self, max_steps, n_inds, check_converge = 5):
        converged = 0

        for g in range(0, max_steps):
            generation = []
            fitnesses = []
            for _ in range(0, n_inds):
                new_ind = self.one_step()
                if not (new_ind is None):
                    new_ind_fitness = self.basic_property(new_ind, self.nb_range, self.nb_num)[-1]
                    if new_ind_fitness == 0:
                        self.ind_fitness = 0
                        return new_ind
                    generation.append(new_ind)
                    fitnesses.append(new_ind_fitness)
            if len(fitnesses):
                best_fitness = np.argsort(fitnesses)[0]
                best_ind = generation[best_fitness]

                if fitnesses[best_fitness] < self.ind_fitness:
                    converged = 0
                    self.atoms = sort_atoms_by_distance(best_ind)
                    self.fitness()
                else:
                    converged +=1
                    if converged >= check_converge:
                        return self.atoms
            else:
                break
        
        return self.atoms
    

from dscribe.descriptors import LMBTR, SOAP
from sklearn.preprocessing import normalize
class LERefinement:
    def __init__(self, atoms, env_base = None):
        self.atoms = atoms
        self.env_base = env_base
        self.lmbtr = LMBTR(
            species=['H'],
            k2={
                "geometry": {"function": "distance"},
                "grid": {"min": 0, "max": 1.5, "n": 100, "sigma": 0.5},
                #"weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
            },
            
            #k3={
            #    "geometry": {"function": "angle"},
            #    "grid": {"min": 0, "max": 180, "n": 100, "sigma": 0.05},
            #    "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
            #},
            
            periodic=False,
            normalization="none",
        )

        self.soap = SOAP(species=['H'], periodic=False, rcut=1.5, 
                         nmax=8, lmax=6)

        self.dp = self.soap


    def create(self):
        atoms = sort_atoms_by_distance(self.atoms)
        sites =  normalize(self.dp.create(atoms, positions = atoms.positions) )
        #sites =  self.dp.create(atoms, positions = atoms.positions) 
        return sites
    
    @staticmethod
    def rattle(pos):
        r = 2 * np.random.rand()**(1/3)
        theta = np.random.uniform(0, np.pi)
        phi =  np.random.uniform(0, 2*np.pi)

        new_pos = pos + r * np.array([np.sin(theta) * np.cos(phi), 
                                                                        np.sin(theta) * np.sin(phi),
                                                                        np.cos(theta)])

        return new_pos
    
    @staticmethod
    def env_dis(env1, env2):
        return np.sum([x*x for x in (env1 - env2)])
    
    def basic_property(self):
        self.dis_matrix = []
        atoms_env = self.create()
        for env in atoms_env:
            d = [self.env_dis(env, e) for e in self.env_base]
            fit_into = np.argmin(d)
            self.dis_matrix.append([fit_into, d[fit_into]])
        self.dis_matrix = np.array(self.dis_matrix)
        self.ind_fitness = np.sum(self.dis_matrix[:, 1])
        return self.dis_matrix, self.ind_fitness
    
    def one_step(self):
        dis_matrix, fit = self.basic_property()
        to_mv = np.argmax(dis_matrix[:, 1])
        new_atoms = self.atoms.copy()
        new_atoms[to_mv].position = self.rattle(new_atoms[to_mv].position)
        new_env = normalize(self.dp.create(self.atoms, positions = [new_atoms[to_mv].position]) )
        new_fit = np.min([self.env_dis(new_env, e) for e in self.env_base])
        new_fit +=  fit - dis_matrix[to_mv][1]
        print("fit matrix ", self.dis_matrix)
        print("move atom ", to_mv, " from fit ", fit, " to ", new_fit)
        return new_atoms, new_fit


    def generator_main(self, max_steps, n_inds, check_converge = 5):
        converged = 0

        for g in range(0, max_steps):
            generation = []
            fitnesses = []
            for _ in range(0, n_inds):
                new_ind, new_fit = self.one_step()
                if not (new_ind is None):
                    new_ind_fitness = new_fit
                    if new_ind_fitness == 0:
                        self.ind_fitness = 0
                        return new_ind
                    generation.append(new_ind)
                    fitnesses.append(new_ind_fitness)
            if len(fitnesses):
                best_fitness = np.argsort(fitnesses)[0]
                best_ind = generation[best_fitness]

                if fitnesses[best_fitness] < self.ind_fitness:
                    converged = 0
                    self.atoms = sort_atoms_by_distance(best_ind)
                    print("fitness from ", self.ind_fitness, " to ", fitnesses[best_fitness])
                else:
                    converged +=1
                    if converged >= check_converge:
                        return self.atoms
            else:
                break
        
        return self.atoms
        
    
if __name__ == '__main__':
    
    import ase.io
    """
    pop = ase.io.read('result', format='traj', index = ':')
    refined = []
    for atoms in pop:
        r = Refinement(atoms)
        print("basic property", r.current_nb_num,r.ind_fitness)
        r.generator_main(50,50)
        print("now fit", r.ind_fitness, r.fit_array, np.where(r.fit_array !=0))
        refined.append(r.atoms)

    ase.io.write('refined.traj', refined)
    """
    atoms2 = sort_atoms_by_distance( ase.io.read('ref.vasp'))

    fp = LERefinement(atoms2)
    s2 = fp.create()
    c = best_cluster(threshold=0.2)
    centers, labels = c.best_cluster(s2,  logout=True)
    en_types = set(labels)

    np.save('env', s2)
    s2 = np.load('env.npy')

    for t in en_types:
        atom = np.where(labels == t)[0]
        for n in atom:
            atoms2[n].number = t

    #atoms1 = sort_atoms_by_distance(ase.io.read('POSCAR_1.vasp'))
    #refine = LERefinement(atoms1, env_base=s2)
    #refined = refine.generator_main(10,5)


    #ase.io.write('refined1.vasp', refined, vasp5 = 1)
    ase.io.write('typed2.vasp', atoms2, vasp5 = 1)