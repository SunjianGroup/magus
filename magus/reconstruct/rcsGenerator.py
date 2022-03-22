class ReconstructGenerator(SPGGenerator):
    def __init__(self, **parameters):
        Requirement = ['symbols', 'input_layers']
        Default = {
            'cutslices': None, 
            'bulk_layernum': 3, 
            'range': 0.5, 
            'relaxable_layernum': 3, 
            'rcs_layernum': 2, 
            'randratio': 0.5,
            'rcs_x': 1, 
            'rcs_y': 1, 
            'direction': None, 
            'rotate': 0, 
            'matrix': None, 
            'extra_c': 1.0,
            'dimension': 2, 
            'choice': 0,}
        check_parameters(self, parameters, Requirement, Default)

        for i, layer in enumerate(self.input_layers):
            if isinstance(layer, str):
                layer = read(layer)
            assert isinstance(layer, Atoms), "input layers must be Atoms or a file path can be read by ASE"
            for s in layer.get_chemical_symbols():
                assert s in self.symbols, "{} of {} not in given symbols".format(s, layer.get_chemical_formula())
            self.input_layers[i] = layer
        
        assert len(self.input_layers) in [2, 3]

        if len(self.input_layers)==3:
            # mode = 'reconstruct'
            self.ref_layer = self.input_layers[2]
        else:
            # mode = 'add atoms'
            self.ref_layer = self.input_layers[1].copy()
        if 'formula' not in parameters:
            assert 'addFormula' in parameters, "must have 'formula' or 'addFormula'"
            ref_symbols = self.ref_layer.get_chemical_symbols()
            parameters['formula'] = [ref_symbols.count(s) + n_add 
                                     for s, n_add in self.symbols]

        ## ?????????????
        setlattice = np.round(setlattice, 3)
        setlattice[3:] = [i if np.round(i) != 60 else 120.0 for i in setlattice[3:]]
        self.symtype = 'hex' if 120 in np.round(setlattice[3:]) else 'orth'

        super().__init__(**parameters)

    def get_default_formula_pool(self):
        if self.p.AtomsToAdd:
            assert len(self.p.AtomsToAdd)== len(self.p.symbols), 'Please check the length of AddAtoms'
            try:
                self.p.AtomsToAdd = split_modifier(self.p.AtomsToAdd)
            except:
                raise RuntimeError("wrong format of atomstoadd")
        if self.p.DefectToAdd:
            try:
                self.p.DefectToAdd =  split_modifier(self.p.DefectToAdd)
            except:
                raise RuntimeError("wrong format of defectstoadd")

    def __init__(self,parameters):
        para_t = EmptyClass()
        Requirement=['layerfile']
        Default={'cutslices': None, 'bulk_layernum':3, 'range':0.5, 'relaxable_layernum':3, 'rcs_layernum':2, 'randratio':0.5,
        'rcs_x':[1], 'rcs_y':[1], 'direction': None, 'rotate': 0, 'matrix': None, 'extra_c':1.0, 
        'dimension':2, 'choice':0 }

        checkParameters(para_t, parameters, Requirement,Default)
        
        if os.path.exists("Ref") and os.path.exists("Ref/refslab.traj") and os.path.exists("Ref/layerslices.traj"):
            log.info("Used layerslices in Ref.")
        else:
            if not os.path.exists("Ref"):
                os.mkdir('Ref')
            #here starts to get Ref/refslab to calculate refE            
            ase.io.write("Ref/refslab.traj", ase.io.read(para_t.layerfile), format = 'traj')
            #here starts to split layers into [bulk, relaxable, rcs]
            originatoms = ase.io.read(para_t.layerfile)
            layernums = [para_t.bulk_layernum, para_t.relaxable_layernum, para_t.rcs_layernum]
            cutcell(originatoms, layernums, totslices = para_t.cutslices, direction= para_t.direction,rotate = para_t.rotate, vacuum = para_t.extra_c, matrix = para_t.matrix)
            #layer split ends here    

        self.range=para_t.range
        
        self.ind=RcsInd(parameters)

        #here get new parameters for self.generator 
        _parameters = copy.deepcopy(parameters)
        _parameters.attach(para_t)
        self.layerslices = ase.io.read("Ref/layerslices.traj", index=':', format='traj')
        
        setlattice = []
        if len(self.layerslices)==3:
            #mode = 'reconstruct'
            self.ref = self.layerslices[2]
            vertical_dis = self.ref.get_scaled_positions()[:,2].copy()
            mincell = self.ref.get_cell().copy()
            mincell[2] *= (np.max(vertical_dis) - np.min(vertical_dis))*1.2
            setlattice = list(cell_to_cellpar(mincell))
        else:
            #mode = 'add atoms'
            para_t.randratio = 0
            self.ref = self.layerslices[1].copy()
            lattice = self.ref.get_cell().copy()
            lattice [2]/= para_t.relaxable_layernum
            self.ref.set_cell(lattice)
            setlattice = list(cell_to_cellpar(lattice))

        setlattice = np.round(setlattice, 3)
        setlattice[3:] = [i if np.round(i) != 60 else 120.0 for i in setlattice[3:]]
        self.symtype = 'hex' if 120 in np.round(setlattice[3:]) else 'orth'

        self.reflattice = list(setlattice).copy()
        target = self.ind.get_targetFrml()
        _symbol = [s for s in target]
        requirement = {'minLattice': setlattice, 'maxLattice':setlattice, 'symbols':_symbol}

        for key in requirement:
            if not hasattr(_parameters, key):
                setattr(_parameters,key,requirement[key])
            else:
                if getattr(_parameters,key) == requirement[key]:
                    pass
                else:
                    logging.info("warning: change user defined {} to {} to match rcs layer".format(key, requirement[key]))
                    setattr(_parameters,key,requirement[key])

        self.rcs_generator =Generator(_parameters)
        self.rcs_generator.p.choice =_parameters.choice
        #got a generator! next put all parm together except changed ones

        self.p = EmptyClass()
        self.p.attach(para_t)
        self.p.attach(self.rcs_generator.p)

        origindefault={'symbols':parameters.symbols}
        origindefault['minLattice'] = parameters.minLattice if hasattr(parameters, 'minLattice') else None
        origindefault['maxLattice'] = parameters.maxLattice if hasattr(parameters, 'maxLattice') else None

        for key in origindefault:
            if not hasattr(self.p, key):
                pass
            else:
                setattr(self.p,key,origindefault[key])
        
        #some other settings
        minFrml = int(np.ceil(self.p.minAt/sum(self.p.formula)))
        maxFrml = int(self.p.maxAt/sum(self.p.formula))
        self.p.numFrml = list(range(minFrml, maxFrml + 1))
        self.threshold = self.p.dRatio
        self.maxAttempts = 100

    def afterprocessing(self,ind,nfm, origin, size):
        ind.info['symbols'] = self.p.symbols
        ind.info['formula'] = self.p.formula
        ind.info['numOfFormula'] = nfm
        ind.info['parentE'] = 0
        ind.info['origin'] = origin
        ind.info['size'] = size

        return ind
        
    def update_volume_ratio(self, volume_ratio):
        pass
        #return self.rcs_generator.update_volume_ratio(volume_ratio)
    def Generate_ind(self,spg,numlist):
        return self.rcs_generator.Generate_ind(spg,numlist)

    def reconstruct(self, ind):

        c=reconstruct(self.range, ind.copy(), self.threshold, self.maxAttempts)
        label, pos=c.reconstr()
        numbers=[]
        if label:
            for i in range(len(c.atomnum)):
                numbers.extend([atomic_numbers[c.atomname[i]]]*c.atomnum[i])
            cell=c.lattice
            pos=np.dot(pos,cell)
            atoms = ase.Atoms(cell=cell, positions=pos, numbers=numbers, pbc=1)
            
            return label, atoms
        else:
            return label, None

    def rand_displacement(self, extraind, bottomind): 
        rots = []
        trs = []
        for ind in list([bottomind, extraind]):
            sym = spglib.get_symmetry_dataset(ind,symprec=0.2)
            if not sym:
                sym = spglib.get_symmetry_dataset(ind)
            if not sym:
                return False, extraind
            rots.append(sym['rotations'])
            trs.append(sym['translations'])

        m = match_symmetry(*zip(rots, trs), z_axis_only = True)
        if not m.has_shared_sym:
            return False, extraind
        _dis_, rot = m.get()
        #_dis_, rot = match_symmetry(*zip(rots, trs)).get() 
        _dis_[2] = 0
        _dis_ = np.dot(-_dis_, extraind.get_cell())

        extraind.translate([_dis_]*len(extraind))
        return True, extraind

    def get_spg(self, kind, grouptype):
        if grouptype == 'layergroup':
            if kind == 'hex':
                #sym = 'c*', 'p6*', 'p3*', 'p-6*', 'p-3*' 
                return [1, 2, 22, 26, 35, 36, 47, 48] + range(65, 81)  + [10, 13, 18]
            else:
                return list(range(1, 65))
        elif grouptype == 'planegroup':
            if kind == 'hex':
                return [1, 2, 5, 9] + list(range(13, 18))
            else:
                return list(range(1, 13))

    def reset_rind_lattice(self, atoms, _x, _y, botp = 'refbot', type = 'bot'):

        refcell = (self.ref * (_x, _y, 1)).get_cell_lengths_and_angles()
        cell = atoms.get_cell_lengths_and_angles()

        if not np.allclose(cell[:2], refcell[:2], atol=0.1):
            return False, None
        if not np.allclose(cell[3:], refcell[3:], atol=0.5):
            #'hex' lattice
            if np.round(refcell[-1] + cell[-1] )==180:
                atoms = resetLattice(atoms = atoms.copy(), expandsize = (4,1,1)).get(np.dot(np.diag([-1, 1, 1]), atoms.get_cell() ))

            else:
                return False, None
        atoms.set_cell(np.dot(np.diag([1,1, refcell[2]/cell[2]]) ,atoms.get_cell()))
        refcell = (self.ref * (_x, _y, 1)).get_cell()
        atoms.set_cell(refcell, scale_atoms = True)
        pos = atoms.get_scaled_positions(wrap = False)
        refpos = self.ref.get_scaled_positions(wrap = True)
        bot = np.min(pos[:,2]) if type == 'bot' else np.mean(pos[:, 2])
        tobot = np.min(refpos[:,2])*atoms.get_cell()[2] if isinstance(botp, str) else botp
        atoms.translate([ tobot - bot*atoms.get_cell()[2]]* len(atoms))
        return True, atoms
        
        
    def reset_generator_lattice(self, _x, _y, spg):
        symtype = 'default'
        if self.symtype == 'hex':
            if (self.rcs_generator.p.choice == 0 and spg < 13) or (self.rcs_generator.p.choice == 1 and spg < 65):
                #for hex-lattice, 'a' must equal 'b'
                if self.reflattice[0] == self.reflattice[1] and _x == _y:    
                    symtype = 'hex'

        if symtype == 'hex':
            self.rcs_generator.p.GetConventional = False
        elif symtype == 'default': 
            self.rcs_generator.p.GetConventional = True

        self.rcs_generator.p.minLattice = list(self.reflattice *np.array([_x, _y]+[1]*4))
        self.rcs_generator.p.maxLattice = self.rcs_generator.p.minLattice
        return symtype


    def get_lattice(self, numlist):
        _, max_volume = self.get_volume(numlist)
        min_lattice = [2 * np.max(self.radius)] * 3 + [45.] * 3
        max_lattice = [3 * max_volume ** (1/3)] * 3 + [135] * 3
        if self.min_lattice is not None:
            min_lattice = self.min_lattice
        if self.max_lattice is not None:
            max_lattice = self.max_lattice

        self.ref = self.layerslices[1].copy()
        lattice = self.ref.get_cell().copy()
        lattice [2]/= para_t.relaxable_layernum
        self.ref.set_cell(lattice)
        setlattice = list(cell_to_cellpar(lattice))

        setlattice = np.round(setlattice, 3)
        setlattice[3:] = [i if np.round(i) != 60 else 120.0 for i in setlattice[3:]]
        self.symtype = 'hex' if 120 in np.round(setlattice[3:]) else 'orth'

        self.reflattice = list(setlattice).copy()
        target = self.ind.get_targetFrml()
        _symbol = [s for s in target]
        requirement = {'minLattice': setlattice, 'maxLattice':setlattice, 'symbols':_symbol}


    def generate_pop(self, n_pop, *args, **kwargs):
        build_pop = []
        while n_pop > len(build_pop):
            for _ in range(self.max_n_try):
                spg = np.random.choice(self.spacegroup)
                n_split = np.random.choice(self.n_split)
                numlist = self.get_numlist(*args, **kwargs)
                label, atoms = self.generate_ind(spg, numlist, n_split)



                self.rcs_generator.p.choice = self.p.choice
                #logging.debug("formula {} of number {} with chosen spg = {}".format(self.rcs_generator.p.symbols, numlist,spg))
                #logging.debug("with maxlattice = {}".format(self.rcs_generator.p.maxLattice))
                label,ind = self.rcs_generator.Generate_ind(spg,numlist)

                if label:
                    #label, ind = self.reset_rind_lattice(ind, _x, _y, botp = 'refbot', type = 'bot')
                    label, ind = self.reset_rind_lattice(ind, _x, _y, botp = 'refbot')
                if label:
                    _bot_ = (self.layerslices[1] * (_x, _y, 1)).copy()
                    _bot_.info['size'] = [_x, _y]
                    
                    label, ind = self.rand_displacement(ind, self.ind.addvacuum(add = 1, atoms = self.ind.addextralayer('bulk', atoms=_bot_, add = 1)))
                if label:
                    self.afterprocessing(ind,nfm,'rand.symmgen', [_x, _y])
                    ind = self.ind.addbulk_relaxable_vacuum(atoms = ind)
                    buildPop.append(ind)
                if not label:
                    tryNum+=1


                if label:
                    self.afterprocessing(atoms)
                    build_pop.append(atoms)
                    break
            else:
                n_split = np.random.choice(self.n_split)
                numlist = self.get_numlist(*args, **kwargs)
                label, atoms = self.generate_ind(1, numlist, n_split)
                if label:
                    self.afterprocessing(atoms, *args, **kwargs)
                    build_pop.append(atoms)
        return build_pop
