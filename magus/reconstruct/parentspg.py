# Spacegroup miner based on supergroup-subgroup relations.
# For example, spacegroup no.2, P-1 could mine into its supergroup no.10, P2/m.
from collections import Counter
import numpy as np
import logging
from magus.reconstruct.supergroupdb import supergroup

_s_1 = {}
for s in range(2,231):
    _s_1[s] = [2]

supergroup[1] = {'supergroups':_s_1}

log = logging.getLogger(__name__)

# International Tables for Crystallography (2016). Vol. A, Figure 3.2.1.3, p. 732.
# supergroups of the three-dimensional crystallographic point groups
_super_group_relation = {
    (1,2, '1'): {'symbol':"C1",
    'order':1,'super':["2",'m','-1','3']},
    (3,6, '2'): {'symbol':"C2",
    'order':2,'super':['4','-4','222','mm2','2/m','6','32']},
    (6,10, 'm'): {'symbol':"Cs",
    'order':2,'super':['mm2','2/m','-6','3m']},
    (2,3, "-1"): {'symbol':"Ci",
    'order':2,'super':['2/m','-3']},
    (143,147, '3'): {'symbol':"C3",
    'order':3,'super':['6','-6','32','3m','-3','23']},
    (75,81, '4'): {'symbol':"C4",
    'order':4,'super':['422','4mm','4/m']},
    (81,83, '-4'): {'symbol':"S4",
    'order':4,'super':['-42m','4/m']},
    (16,25, "222"): {'symbol':"D2",
    'order':4,'super':['422','-42m','mmm','23','622']},
    (25,47, 'mm2'): {'symbol':"C2v",
    'order':4,'super':['-42m','4mm','mmm','-6m2','6mm']},
    (10,16, "2/m"): {'symbol':"C2h",
    'order':4,'super':['4/m','mmm','6/m','-3m']},
    (168,174, '6'): {'symbol':"C6",
    'order':6,'super':['622','6mm','6/m']},
    (174,175, '-6'): {'symbol':"C3h",
    'order':6,'super':['-6m2','6/m']},
    (149,156, "32"): {'symbol':"D3",
    'order':6,'super':['622','-6m2','432','-3m']},
    (156,162, "3m"): {'symbol':"C3v",
    'order':6,'super':['-6m2','6mm','-3m','-43m']},
    (147,149, '-3'): {'symbol':"C3i",
    'order':6,'super':['6/m','m-3','-3m']},
    (89,99, "422"): {'symbol':"D4",
    'order':8,'super':['4/mmm','432']},
    (111,123, "-42m"): {'symbol':"D2d",
    'order':8,'super':['4/mmm','-43m']},
    (99,111, '4mm'): {'symbol':"C4v",
    'order':8,'super':['4/mmm']},
    (83,89, '4/m'): {'symbol':'C4h',
    'order':8,'super':['4/mmm']},
    (47,75, 'mmm'): {'symbol':'D2h',
    'order':8,'super':['4/mmm','m-3','6/mmm']},
    (195,200, '23'): {'symbol':'T',
    'order':12,'super':['432','-43m','m-3']},
    (177,183, '622'): {'symbol':'D6',
    'order':12,'super':['6/mmm']},
    (187,191, '-6m2'): {'symbol':'D3h',
    'order':12,'super':['6/mmm']},
    (183,187, '6mm'): {'symbol':'C6v',
    'order':12,'super':['6/mmm']},
    (175,177, '6/m'): {'symbol':'C6h',
    'order':12,'super':['6/mmm']},
    (162,168, '-3m'): {'symbol':'D3d',
    'order':12,'super':['6/mmm','m-3m']},
    (123,143, '4/mmm'): {'symbol':'D4h',
    'order':16,'super':['m-3m']},
    (207,215, '432'): {'symbol':'O',
    'order':24,'super':['m-3m']},
    (215,221, '-43m'): {'symbol':'Td',
    'order':24,'super':['m-3m']},
    (200,207, 'm-3'): {'symbol':'Th',
    'order':24,'super':['m-3m']},
    (191,195, '6/mmm'): {'symbol':'D6h',
    'order':24,'super':[]},
    (221,231, 'm-3m'): {'symbol':'Oh',
    'order':48,'super':[]},
}
def super_group_relation(spacegroup = -1, pointgroup = ""):
    if spacegroup > 0:
        for key in _super_group_relation:
            if spacegroup in list(range(key[0], key[1])):
                return {**_super_group_relation[key], 
                        "share_pointgroup": list(range(key[0], key[1])),
                        "name": key[2]}
    else:
        for key in _super_group_relation:
            if pointgroup == key[-1]:
                return {**_super_group_relation[key], 
                        "share_pointgroup": list(range(key[0], key[1])),
                        "name": key[2]}


class Miner:
    def __init__(self):
        pass
    
    @staticmethod
    def get_supergroup(supergroup_list, spacegroup = -1, pointgroup = ""):
        x = super_group_relation(supergroup_list, spacegroup=spacegroup, pointgroup=pointgroup)
        supergroup_list.extend([s for s in x['share_pointgroup'] if not s in supergroup_list])
        for name in x['super']:
            Miner.get_supergroup(supergroup_list, spacegroup=-1, pointgroup=name)
    
    '''
    def mine_spg(self, spg, **kwargs):
        minerfunc = {        
                    'boundary_order' : 7, 
                    'exact_spacegroup' : [0.5,0.1], 
                    'same_pointgroup' : [0.4,0.1], 
                    'super_pointgroup' : [0.1,0.8]
                    }
        minerfunc.update(kwargs)
        
        supergroup_with_self = self._mine_spg(spg)
        log.debug("Mine into '{}' (pointgroup: {}; order: {})".format(spg, supergroup_with_self[-1]['ptg-name'], supergroup_with_self[-1]['order']))
        # adjust weight
        if supergroup_with_self[-1]['order']> minerfunc['boundary_order']:
            parm = 0
        else:
            parm = 1

        for k in minerfunc:
            if 'group' in k:
                minerfunc[k] = minerfunc[k][parm]

        miner = Counter({})

        for infodict in supergroup_with_self[:-1]:
            #s:math.log(infodict['order'],2) ???
            miner += Counter({s:infodict['order']**2 for s in infodict['spg-no']})
        _sum_value = sum(miner.values())
        miner = Counter({s: miner[s]/_sum_value * minerfunc['super_pointgroup'] for s in miner})
        log.debug("Supergroup: '{}'".format(list(map(lambda x:(x[0],np.round(x[1]*100,2)), miner.items()))))

        miner += Counter({s:minerfunc['same_pointgroup']/len(supergroup_with_self[-1]['spg-no']) 
                          for s in supergroup_with_self[-1]['spg-no']})
        log.debug("Share pointgroup: '{}'".format(list(map(lambda x:(x[0],np.round(x[1]*100,2)), 
                                                       Counter({s:minerfunc['same_pointgroup']/len(supergroup_with_self[-1]['spg-no']) 
                          for s in supergroup_with_self[-1]['spg-no']}).items()))))

        miner += Counter({spg: minerfunc['exact_spacegroup']})    
        log.debug("Exact spacegroup: '{}'".format(list(map(lambda x:(x[0],np.round(x[1]*100,2)), Counter({spg: minerfunc['exact_spacegroup']}).items()))))  
        #print(sum(miner.values()))
        return miner
    '''
    def mine_spg(self, spg):
        x = super_group_relation(spacegroup = spg)
        print(x)
        log.debug("Mine into '{}' (pointgroup: {}; order: {})".format(spg, x['name'], x['order']))

        miner = {}
        isHighOrder = (x['order'] >7)
        isVeryHighOrder = (x['order'] >40)

        for s in supergroup[spg]['supergroups'].keys():
            if supergroup[spg]['supergroups'][s][0] <=(4 if isHighOrder else 8):
                miner[s] = 2
            else:
                miner[s] = 1

        if isHighOrder:
            miner[spg] = 50

        have_very_high_order = np.max([miner.get(s, 0) for s in range(221,231)])
        for s in range(221,231):
            # add all m-3m, things is weird when dealing with very high order
            miner[s] = have_very_high_order

        
        return miner

from collections import Counter
import yaml
import ase.io
import spglib

# miner_tracker: which spacegroups we mined into
# spg_trakcer:   spacegroups used to generate randoms
# analyzer:      some spacegroup just cannot generate useful structures.
#                the 'score' reflects this ability 

class MinerTracker:
    def __init__(self, trackfile = 'tracker.yaml', max_limit_per_spg = 10000, scale_num_spg=500):
        self.trackfile = trackfile
        self.analyzer = {}
        self.miner_tracker = Counter({})
        self.max_limit_per_spg = max_limit_per_spg
        self.scale_num_spg = scale_num_spg

    def read(self):
        with open(self.trackfile, 'r') as f:
            d = dict(yaml.load(f, Loader=yaml.FullLoader))
            self.analyzer = d['analyzer']
            self.miner_tracker = Counter(d['miner_tracker'])

    def write(self):
        with open(self.trackfile, 'w') as f:
            info = {'analyzer': dict(self.analyzer), 'miner_tracker':dict(self.miner_tracker)}
            yaml.dump(info, f)

    def add_to_analyzer(self, initspg, finspg, dominator):
        if initspg in self.analyzer:
            self.analyzer[initspg].append([finspg, dominator])
        else:
            self.analyzer[initspg] = [[finspg, dominator]]

    def add_generation_to_analyzer(self, gen):
        rawpop = ase.io.read('results/raw{}.traj'.format(gen), ':')
        rawpop = sorted([atom for atom in rawpop if atom.info['origin'] == 'random'], key = lambda x: x.info['enthalpy'])
        initpop = ase.io.read('results/init{}.traj'.format(gen), ':')
        initids = [x.info['identity'] for x in initpop]
        for i,atom in enumerate(rawpop):
            initatom = initpop[initids.index(atom.info['identity'])]
            initspg = spglib.get_symmetry_dataset(initatom, 0.1)['number']
            try:
                finspg = spglib.get_symmetry_dataset(atom, 0.1)['number']
            except:
                finspg = 1
            self.add_to_analyzer(initspg, finspg, i/len(rawpop))
    
    def add_miner_log_to_miner(self,miner):
        self.miner_tracker += miner

    def filter(self, miner):
        for initspg in miner:
            if not initspg in self.analyzer:
                # have no idea about initspg
                continue
            if len(self.analyzer[initspg]) < self.scale_num_spg:
                # have less idea about initspg
                continue
            elif len(self.analyzer[initspg]) > self.max_limit_per_spg:
                # used initspg too many times
                miner[initspg] = 0
            if not initspg in np.array(self.analyzer[initspg])[:,0]:
                # not stable
                miner[initspg] = 0
            if np.mean(np.array(self.analyzer[initspg])[:,1])>0.5:
                # unfavor
                miner[initspg] /= 2
        return miner


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    print(Miner().mine_spg(1))

