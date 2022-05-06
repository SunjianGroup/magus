from collections import defaultdict
from ase.io import read
from ase.data import chemical_symbols, covalent_radii, atomic_numbers
import numpy as np
import sys, itertools


min_info = defaultdict(list)
all_type = []
frames = read(sys.argv[1], ':')
for atoms in frames:
    num = atoms.get_atomic_numbers()
    all_type = set(list(all_type) + list(num))
    unique_types = sorted(list(set(num)))
    dis = atoms.get_all_distances(mic=True)
    dis += np.eye(len(dis)) * 100
    iterator = itertools.combinations_with_replacement(unique_types, 2)
    for type1, type2 in iterator:
        x1 = np.where(num == type1)
        x2 = np.where(num == type2)
        min_info[(type1, type2)].append(np.min(dis[x1].T[x2]))
        min_info[(type2, type1)].append(np.min(dis[x1].T[x2]))

print('Min distance:')
ret = "     "
for i in all_type:
    ret += chemical_symbols[i].ljust(6, " ")
print(ret)
for i in all_type:
    ret = chemical_symbols[i].ljust(4, " ")
    for j in all_type:
        ret += "{:.2f}  ".format(min(min_info[(i,j)]))
    print(ret)

# print('\n\nMean distance:')
# ret = "     "
# for i in all_type:
#     ret += chemical_symbols[i].ljust(6, " ")
# print(ret)
# for i in all_type:
#     ret = chemical_symbols[i].ljust(4, " ")
#     for j in all_type:
#         ret += "{:.2f}  ".format(np.mean(min_info[(i,j)]))
#     print(ret)

vr = []
for atoms in frames:
    rs = [covalent_radii[n] for n in atoms.numbers]
    bv = np.sum([4 * np.pi * r ** 3 / 3 for r in rs])
    v = atoms.get_volume()
    vr.append(v / bv)
print('Volume Ratio: \n  min : {}\n  mean: {}\n  max : {}'.format(np.min(vr), np.mean(vr), np.max(vr)))
