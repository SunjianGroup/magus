class PopGenerator:
    def __init__(self,numlist,oplist,parameters):
        self.oplist = oplist
        self.numlist = numlist
        self.p = EmptyClass()
        Requirement = ['popSize','saveGood','molDetector', 'calcType']
        Default = {'chkMol': False,'addSym': False,'randFrac': 0.0}
        checkParameters(self.p,parameters,Requirement,Default)

    def clustering(self, clusterNum):
        Pop = self.Pop
        labels,_ = Pop.clustering(clusterNum)
        uqLabels = list(sorted(np.unique(labels)))
        subpops = []
        for label in uqLabels:
            subpop = [ind for j,ind in enumerate(Pop.pop) if labels[j] == label]
            subpops.append(subpop)

        self.uqLabels = uqLabels
        self.subpops = subpops
    def get_pairs(self, Pop, crossNum ,clusterNum, tryNum=50,k=0.3):
        ##################################
        #temp
        #si ma dang huo ma yi
        k = 2 / len(Pop)
        ##################################
        pairs = []
        labels,_ = Pop.clustering(clusterNum)
        fail = 0
        while len(pairs) < crossNum and fail < tryNum:
            #label = np.random.choice(self.uqLabels)
            #subpop = self.subpops[label]
            label = np.random.choice(np.unique(labels))
            subpop = [ind for j,ind in enumerate(Pop.pop) if labels[j] == label]

            if len(subpop) < 2:
                fail+=1
                continue

            dom = np.array([ind.info['dominators'] for ind in subpop])
            edom = np.exp(-k*dom)
            p = edom/np.sum(edom)
            pair = tuple(np.random.choice(subpop,2,False,p=p))
            if pair in pairs:
                fail+=1
                continue
            pairs.append(pair)
        return pairs

    def get_inds(self,Pop,mutateNum,k=0.3):
        #Pop = self.Pop
        ##################################
        #temp
        #si ma dang huo ma yi
        k = 2 / len(Pop)
        ##################################
        dom = np.array([ind.info['dominators'] for ind in Pop.pop])
        edom = np.exp(-k*dom)
        p = edom/np.sum(edom)
        # mutateNum = min(mutateNum,len(Pop))
        if mutateNum > len(Pop):
            return np.random.choice(Pop.pop,mutateNum,True,p=p)
        else:
            return np.random.choice(Pop.pop,mutateNum,False,p=p)

    def generate(self,Pop,saveGood):
        # calculate dominators before checking formula
        Pop.calc_dominators()

        #remove bulk_layer and relaxable_layer before crossover and mutation
        if self.p.calcType=='rcs':
            Pop = Pop.copy()
            Pop.removebulk_relaxable_vacuum()
        if self.p.calcType=='clus':
            Pop.randrotate()
        if self.p.calcType == 'var':
            Pop.check_full()
        #TODO move addsym to ind
        if self.p.addSym:
            Pop.add_symmetry()
        newPop = Pop([],'initpop',Pop.gen+1)

        operation_keys = list(self.oplist.keys())
        for key in operation_keys:
            op = self.oplist[key]
            num = self.numlist[key]
            if num == 0:
                continue
            log.debug('name:{} num:{}'.format(op.descriptor,num))
            if op.optype == 'Mutation':
                mutate_inds = self.get_inds(Pop,num)
                for i,ind in enumerate(mutate_inds):
                    #if self.p.molDetector != 0 and not hasattr(atoms, 'molCryst'):
                    if self.p.molDetector != 0:
                        if not hasattr(ind, 'molCryst'):
                            ind.to_mol()
                    atoms = op.get_new_individual(ind, chkMol=self.p.chkMol)
                    if atoms:
                        newPop.append(atoms)
            elif op.optype == 'Crossover':
                cross_pairs = self.get_pairs(Pop,num,saveGood)
                #cross_pairs = self.get_pairs(Pop,num)
                for i,parents in enumerate(cross_pairs):
                    if self.p.molDetector != 0:
                        for ind in parents:
                            if not hasattr(ind, 'molCryst'):
                                ind.to_mol()
                    atoms = op.get_new_individual(parents,chkMol=self.p.chkMol)
                    if atoms:
                        newPop.append(atoms)
            log.debug("popsize after {}: {}".format(op.descriptor, len(newPop)))

        if self.p.calcType == 'var':
            newPop.check_full()
        if self.p.calcType=='rcs':
            newPop.addbulk_relaxable_vacuum()
        #newPop.save('testnew')
        newPop.check()
        return newPop

    def select(self,Pop,num,k=0.3):
        ##################################
        #temp
        #si ma dang huo ma yi
        #k = 2 / len(Pop)
        ##################################
        if num < len(Pop):
            # pardom = np.array([ind.info['pardom'] for ind in Pop.pop])
            # edom = np.e**(-k*pardom)
            # p = edom/np.sum(edom)
            # Pop.pop = list(np.random.choice(Pop.pop,num,False,p=p))
            Pop.pop = list(np.random.choice(Pop.pop, num, False))
            return Pop
        else:
            return Pop
        

    def next_Pop(self,Pop):
        saveGood = self.p.saveGood
        popSize = int(self.p.popSize*(1-self.p.randFrac))
        newPop = self.generate(Pop,saveGood)
        return self.select(newPop,popSize)

class MLselect(PopGenerator):
    def __init__(self, numlist, oplist, calc,parameters):
        super().__init__(numlist, oplist, parameters)
        self.calc = calc

    def select(self,Pop,num,k=0.3):
        predictE = []
        if num < len(Pop):
            for ind in Pop:
                ind.atoms.set_calculator(self.calc)
                ind.info['predictE'] = ind.atoms.get_potential_energy()
                predictE.append(ind.info['predictE'])
                ind.atoms.set_calculator(None)

            dom = np.argsort(predictE)
            edom = np.exp(-k*dom)
            p = edom/np.sum(edom)
            Pop.pop = np.random.choice(Pop.pop,num,False,p=p)
            return Pop
        else:
            return Pop
