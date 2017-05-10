from __future__ import print_function, division
import os
import ase.io
# from atomdata import *
from ase.data import atomic_numbers, covalent_radii
from ase import Atoms
import math
import random
import re
import logging
import numpy as np
import fractions

def write_spgGenIn(
    composition, #a string like "Ti1O2"
    setRadius, #string
    volume, # list
    spacegroups = [1],# list
    ):
    spgin = open("spgGen.in", 'w')
    spgin.write("""First line is a comment line
    outputDir = spgGenOut
    verbosity = r
    scalingFactor = 1.0
    forceMostGeneralWyckPos = false
    numOfEachSpgToGenerate = 1
    maxAttempts = 100
    """
    )

    spgin.write("composition = %s\n"%(composition))
    spgin.write(setRadius)

    maxLen = volume[1]/9.0
    if maxLen > 3.0:
        spgin.write("minVolume = %s\n"%(volume[0]))
        spgin.write("maxVolume = %s\n"%(volume[1]))
    else:
        maxLen = 4.0
    maxLen = round(maxLen, 3)
    spgin.write("latticeMins = 3.0, 3.0, 3.0, 60.0, 60.0, 60.0\n")
    spgin.write("latticeMaxes = %s, %s, %s, 120.0, 120.0, 120.0\n"%tuple([maxLen]*3))

    spgs = ""
    for num in spacegroups:
        spgs += "%s,"%(num)
    spgin.write("spacegroups = %s"%(spgs))

    spgin.close()


def spgGen(
    symbols, #a list like ['Ti', 'O']
    formula, # a list like [1, 2]
    radius, # a list like [1.0, 0.5]
    volume, # list
    spacegroups = [1],# list
    ):
    composition = ""
    setRadius = ""
    for i, sym in enumerate(symbols):
        if formula[i] > 0:
            composition += "%s%s"% (sym, formula[i])
            setRadius += "setRadius %s=%s\n"%(sym, radius[i])

    write_spgGenIn(composition, setRadius, volume, spacegroups)
    os.system('./spgGen spgGen.in>log 2>&1')
    pop = []
    for num in spacegroups:
        posfile = "spgGenOut/" + composition + '_' + str(num) + '-1'
        if os.path.exists(posfile):
            ind = ase.io.read(posfile, format = 'vasp')
            logging.info('Build random structrue in spacegroup ' + str(num))
            os.system("rm -f " + posfile)
            pop.append(ind)
        else:
            logging.info("Cannot build random structrue in spacegroup " + str(num) + ". Convert to P1")
            write_spgGenIn(composition, setRadius, volume, spacegroups=[1])
            os.system('./spgGen spgGen.in>log 2>&1')
            posfile = "spgGenOut/" + composition + "_1-1"
            if os.path.exists(posfile):
                ind = ase.io.read(posfile, format = 'vasp')
                logging.info("Build random structrue in spacegroup 1")
                os.system("rm -f " + posfile)
                pop.append(ind)
            else:
                logging.info("Cannot build random structrue in spacegroup 1")

    return pop

def build_struct(
    popSize,
    symbols,
    formula,
    numFrml = [1],
    meanVolume = None, #Assume meanVolume is a float, volume of one formula
    radius = None, #Assume radium is a list
    spgs = range(1,231),
    ):

    buildPop = []
    os.chdir('BuildStruct')
    if not radius:
            # radius = [atomRadii[atom] for atom in symbols]
            radius = [covalent_radii[atomic_numbers[atom]] for atom in symbols]

    if not meanVolume:
        meanVolume = 0
        for i in range(len(symbols)):
            meanVolume = meanVolume + 4*math.pi/3*(radius[i]**3)*formula[i]

    for nfm in numFrml:
        if popSize//len(numFrml) == 0:
            break
        inputVolume = nfm * meanVolume
        inputFrml = [nfm*frml for frml in formula]
        chooseSpg = random.sample(spgs, int(popSize/len(numFrml)))

        randomPop = build_struct_per_formula(symbols, inputFrml, radius, inputVolume, chooseSpg)
        for ind in randomPop:
            # sortSybl = re.sub(r'[0-9]', ' ', randomPop[0].get_chemical_formula()).split()
            # sortFrml = re.sub(r'[^0-9]', ' ', randomPop[0].get_chemical_formula()).split()
            # sortFrml = [int(num)/nfm for num in sortFrml]
            ind.info['symbols'] = symbols
            ind.info['formula'] = formula
            ind.info['numOfFormula'] = nfm
            ind.info['parentE'] = 0
            ind.info['Origin'] = 'random'
        buildPop.extend(randomPop)
    if popSize > len(buildPop):
        randomPop = []
        for i in range(popSize - len(buildPop)):
            nfm = random.choice(numFrml)
            inputVolume = nfm * meanVolume
            inputFrml = [nfm*frml for frml in formula]
            chooseSpg = random.sample(spgs, 1)
            randomPop.extend(build_struct_per_formula(symbols, inputFrml, radius, inputVolume, chooseSpg))
            if len(randomPop) > 0:
                randomPop[-1].info['numOfFormula'] = nfm
                randomPop[-1].info['symbols'] = symbols
                randomPop[-1].info['formula'] = formula
                randomPop[-1].info['parentE'] = 0


        # for ind in randomPop:
            # sortSybl = re.sub(r'[0-9]', ' ', randomPop[0].get_chemical_formula()).split()
            # sortFrml = re.sub(r'[^0-9]', ' ', randomPop[0].get_chemical_formula()).split()
            # sortFrml = [int(num)/nfm for num in sortFrml]
        buildPop.extend(randomPop)

        for ind in buildPop:
            ind.info['toCalc'] = True


    os.chdir('..')

    return buildPop

def build_struct_per_formula(
    symbols,
    formula,
    radius,
    meanVolume,
    spgs, #a list of spacegroups, len(spgs) equals number of structures
    ):
    volume=[meanVolume*0.5, meanVolume*1.5]

    radius = [random.uniform(0.6,0.8)*r for r in radius]
    logging.info("%s %s %s %s %s"%(symbols, formula, radius, volume, spgs))
    randomPop = spgGen(symbols, formula, radius, volume, spgs)
    for i, ind in enumerate(randomPop):
        # sortInd = Atoms(sorted(list(ind), key=lambda atom: atomic_numbers[atom.symbol]), pbc=True)
        sortInd = Atoms(sorted(list(ind), key=lambda atom: symbols.index(atom.symbol)))
        sortInd.set_cell(ind.get_cell())
        # ind = Atoms(sorted(list(ind), key=lambda atom: symbols.index(atom.symbol)))
        randomPop[i] = sortInd
    logging.info("Build " + str(len(randomPop)) + ' structures')

    return randomPop

def reduce_formula(inPop, parameters):

    allSym = parameters['symbols']
    reducePop = []
    for ind in inPop:
        selfSym = ind.info['symbols']
        formula = ind.info['formula']
        gcd = reduce(fractions.gcd ,formula)
        minFrml = [nAtoms//gcd for nAtoms in formula]
        symDict = dict(zip(selfSym, minFrml))

        if len(allSym) != len(selfSym):
            for sym in selfSym:
                if sym not in allSym:
                    symDict[sym] = 0

            minFrml = [symDict[sym] for sym in allSym]

        logging.info("minFrml: %s" %(minFrml))
        ind.info['symDict'] = symDict
        ind.info['formula'] = minFrml
        reducePop.append(ind)

    return reducePop

def varcomp_2elements(popSize, symbols, minAt, maxAt):
    """Variable composition"""
    randomPop = list()
    for i in range(popSize):
        numAt = random.randrange(minAt, maxAt+1)
        # print("numAt: %s"%(numAt))
        num1 = random.random()
        # num2 = 1 - ele1
        ele1 = int(math.ceil(num1*numAt))
        ele2 = numAt - ele1
#        inputSym, inputFrml = zip(*filter(lambda x: x[1] > 0, zip(symbols, [ele1, ele2])))
        inputSym = symbols
        inputFrml = [ele1, ele2]
        randomPop.extend(build_struct(1, inputSym, inputFrml))
    return randomPop

def varcomp_build(popSize, symbols, minAt, maxAt):
    """Variable composition"""
    randomPop = list()
    for i in range(popSize):
        numAt = random.randrange(minAt, maxAt+1)
        # print("numAt: %s"%(numAt))
        nums = np.random.rand(len(symbols))
        nums /= nums.sum()
        eles = np.rint(nums * numAt)
        eles[-1] = numAt - eles[:-1].sum()
        # num2 = 1 - ele1
#        inputSym, inputFrml = zip(*filter(lambda x: x[1] > 0, zip(symbols, [ele1, ele2])))
        inputSym = symbols
        inputFrml = eles.tolist()
        randomPop.extend(build_struct(1, inputSym, inputFrml))
    return randomPop

def symbols_and_formula(atoms):

    allSym = atoms.get_chemical_symbols()
    symbols = list(set(allSym))
    numOfSym = lambda sym: len([i for i in allSym if i == sym])
    formula = map(numOfSym, symbols)

    return symbols, formula


def read_seeds(parameters, seedFile='Seeds/POSCARS'):

    seedPop = []
    setSym = parameters['symbols']
    setFrml = parameters['formula']
    minAt = parameters['minAt']
    maxAt = parameters['maxAt']
    calcType = parameters['calcType']

    if calcType == 'fix':
        setGcd = reduce(fractions.gcd, setFrml)
        setRd = [x/setGcd for x in setFrml]

    if os.path.exists(seedFile):
        readPop = ase.io.read(seedFile, index=':', format='vasp-xdatcar')
        if len(readPop) > 0:
            logging.info("Reading Seeds ...")
        for ind in readPop:
            selfSym, selfFrml = symbols_and_formula(ind)
            # logging.debug('selfSym: {!r}'.format(selfSym))
            symDic = dict(zip(selfSym, selfFrml))
            for sym in [s for s in setSym if s not in selfSym]:
                symDic[sym] = 0

            if False not in map(lambda x: x in setSym, selfSym):
                ind.info['symbols'] = setSym
            else:
                logging.debug("ERROR in check symbols")
                continue

            if calcType == 'var':
                if minAt <= len(ind) <= maxAt:
                    formula = [symDic[sym] for sym in setSym]
                    ind.info['formula'] = formula
                    seedPop.append(ind)
                else:
                    logging.debug("ERROR in check formula")

            elif calcType == 'fix':
                if len(selfSym) == len(setSym):
                    formula = [symDic[sym] for sym in setSym]
                    logging.debug('formula: {!r}'.format(formula))
                    selfGcd = reduce(fractions.gcd, formula)
                    selfRd = [x/selfGcd for x in formula]
                    if selfRd == setRd:
                        ind.info['formula'] = formula
                        ind.info['numOfFormula'] = len(ind)/sum(formula)
                        seedPop.append(ind)
                    else:
                        logging.debug("ERROR in check formula")



    logging.info("Read Seeds: %s"%(len(seedPop)))
    return seedPop
