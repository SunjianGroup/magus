from __future__ import print_function, division
import os
import subprocess, shutil
import ase.io
# from atomdata import *
from ase.data import atomic_numbers, covalent_radii
from ase import Atoms
from scipy.spatial.distance import cdist, pdist
from . import GenerateNew
import math
import random
import re
import logging
import numpy as np
import fractions
from .utils import *
try:
    from functools import reduce
except ImportError:
    pass


def generate_centers_cell(formula, spg, radius, minVol, maxVol):
    assert len(formula) == len(radius)
    numType = len(formula)
    generator = GenerateNew.Info()
    generator.spg = spg
    generator.spgnumber = 1
    if minVol:
        generator.minVolume = minVol
    if maxVol:
        generator.maxVolume = maxVol
    generator.maxAttempts = 10
    numbers = []
    for i in range(numType):
        generator.AppendAtoms(formula[i], i, radius[i])
        numbers.extend([i]*formula[i])

    label = generator.PreGenerate()
    if label:
        cell = generator.GetLattice(0)
        cell = np.reshape(cell, (3,3))
        positions = generator.GetPosition(0)
        positions = np.reshape(positions, (-1, 3))
        return label, cell, numbers, positions
    else:
        return label, None, None, None

def mol_radius_and_rltPos(atoms):
    pos = atoms.get_positions()
    center = pos.mean(0)
    dists = cdist([center], pos)
    radius = dists.max()
    rltPos = pos - center
    return radius, rltPos

def generate_one_mol_crystal(molFormula, spg, radius, rltPosList, molNumList, minVol=None, maxVol=None):
    assert len(radius) == len(molFormula) == len(rltPosList) == len(molNumList)
    # numType = len(molFormula)
    label, cell, molIndices, centers = generate_centers_cell(molFormula, spg, radius, minVol, maxVol)
    print(radius, spg)
    # print(pdist(np.dot(centers, cell)))
    if label:
        # tmpAts = Atoms(cell=cell, scaled_positions=centers, numbers=[1]*len(molIndices), pbc=1)
        # print(tmpAts.get_all_distances(mic=True))
        numList = []
        posList = []
        for i, molInd in enumerate(molIndices):
            numList.append(molNumList[molInd])
            molPos = np.dot(centers[i], cell) + np.dot(rltPosList[molInd], rand_rotMat())
            posList.append(molPos)
        pos = np.concatenate(posList, axis=0)
        numbers = np.concatenate(numList, axis=0)
        atoms = Atoms(cell=cell, positions=pos, numbers=numbers, pbc=1)
        return atoms
    else:
        return None

def generate_mol_crystal_list(molList, molFormula, spgList, numStruct, minVol=None, maxVol=None):
    assert len(molList) == len(molFormula)
    radius = []
    molNumList = []
    rltPosList = []
    minVol = 0
    for i, mol in enumerate(molList):
        numbers = mol.get_atomic_numbers()
        rmol, rltPos = mol_radius_and_rltPos(mol)
        rAt = covalent_radii[numbers].max()
        rmol += rAt
        radius.append(rmol)
        molNumList.append(numbers)
        rltPosList.append(rltPos)
        vol = 8*rmol**3
        # minVol += vol * molFormula[i]

    # minVol = 2*minVol
    # maxVol = 2*minVol

    molPop = []
    for _ in range(numStruct):
        spg = random.choice(spgList)
        # atoms = generate_one_mol_crystal(molFormula, spg, radius, rltPosList, molNumList,)
        atoms = generate_one_mol_crystal(molFormula, spg, radius, rltPosList, molNumList,minVol, maxVol)
        if atoms:
            molPop.append(atoms)

    return molPop

def build_mol_struct(
    popSize,
    symbols,
    formula,
    inputMols,
    molFormula,
    numFrml = [1],
    spgs = range(1,231),
    tryNum = 10
):
    buildPop = []
    for nfm in numFrml:
        numStruct = popSize//len(numFrml)
        if numStruct == 0:
            break
        inputFrml = [nfm*frml for frml in formula]
        inputMolFrml = [nfm*frml for frml in molFormula]
        randomPop = generate_mol_crystal_list(inputMols, inputMolFrml, spgs, numStruct)
        for ind in randomPop:
            ind.info['symbols'] = symbols
            ind.info['formula'] = formula
            ind.info['numOfFormula'] = nfm
            ind.info['molFormula'] = molFormula
            ind.info['parentE'] = 0
            ind.info['Origin'] = 'random'
        buildPop.extend(randomPop)

     # Build structure to fill buildPop
    for i in range(tryNum):
        if popSize <= len(buildPop):
            break
        randomPop = []
        for i in range(popSize - len(buildPop)):
            nfm = random.choice(numFrml)
            inputFrml = [nfm*frml for frml in formula]
            inputMolFrml = [nfm*frml for frml in molFormula]
            randomPop.extend(generate_mol_crystal_list(inputMols, inputMolFrml, spgs, 1))
            if len(randomPop) > 0:
                randomPop[-1].info['numOfFormula'] = nfm
                randomPop[-1].info['symbols'] = symbols
                randomPop[-1].info['formula'] = formula
                randomPop[-1].info['molFormula'] = molFormula
                randomPop[-1].info['parentE'] = 0
                randomPop[-1].info['Origin'] = 'random'
        buildPop.extend(randomPop)

    return buildPop

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

    spgin.write("minVolume = %s\n"%(volume[0]))
    spgin.write("maxVolume = %s\n"%(volume[1]))
    #maxLen = 1.414*volume[1]/9.0
    maxLen = (1.414*volume[1])**(1./3)
    if maxLen > 3.0:
        pass
    else:
        maxLen = 4.0
    maxLen = round(maxLen, 3)
    # spgin.write("latticeMins = 3.0, 3.0, 3.0, 45, 45, 45\n")
    # spgin.write("latticeMaxes = %s, %s, %s, 135, 135, 135\n"%tuple([maxLen]*3))
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
    toP1=0,
    ):
    composition = ""
    setRadius = ""
    for i, sym in enumerate(symbols):
        if formula[i] > 0:
            composition += "%s%s"% (sym, formula[i])
            setRadius += "setRadius %s=%s\n"%(sym, radius[i])

    write_spgGenIn(composition, setRadius, volume, spacegroups)
    # os.system('randSpg spgGen.in>log 2>&1')
    subprocess.call('randSpg spgGen.in>log 2>err', shell=True)
    pop = []
    for num in spacegroups:
        posfile = "spgGenOut/" + composition + '_' + str(num) + '-1'
        if os.path.exists(posfile):
            ind = ase.io.read(posfile, format = 'vasp')
            logging.info('Build random structrue in spacegroup {}'.format(num))
           # os.system("rm -f " + posfile)
            os.remove(posfile)
            pop.append(ind)
        elif toP1:
            logging.info("Cannot build random structrue in spacegroup {}. Convert to P1".format(num))
            write_spgGenIn(composition, setRadius, volume, spacegroups=[1])
            subprocess.call('randSpg spgGen.in>log 2>err', shell=True)
            posfile = "spgGenOut/" + composition + "_1-1"
            if os.path.exists(posfile):
                ind = ase.io.read(posfile, format = 'vasp')
                logging.info("Build random structrue in spacegroup 1")
           #     os.system("rm -f " + posfile)
                os.remove(posfile)
                pop.append(ind)
            else:
                logging.info("Cannot build random structrue in spacegroup 1")
        else:
            # never create P1 structure
            logging.info("Cannot build random structrue in spacegroup {}.".format(num))

    return pop

def spgGen_per_formula(
    symbols, #a list like ['Ti', 'O']
    formula, # a list like [1, 2]
    radius, # a list like [1.0, 0.5]
    volume, # list [min volume, max volume]
    spacegroups, # list
    ):
    composition = ""
    setRadius = ""
    for i, sym in enumerate(symbols):
        if formula[i] > 0:
            composition += "%s%s"% (sym, formula[i])
            setRadius += "setRadius %s=%s\n"%(sym, radius[i])
    write_spgGenIn(composition, setRadius, volume, spacegroups)
    subprocess.call('randSpg spgGen.in>log 2>&1', shell=True)
    pop = []
    for num in spacegroups:
        posfile = "spgGenOut/{}_{}-1".format(composition, num)
        if os.path.exists(posfile):
            ind = ase.io.read(posfile, format = 'vasp')
            os.remove(posfile)
            pop.append(ind)

    return pop

def build_struct_per_formula2(
    symbols,
    formula,
    spgs, #a list of spacegroups, len(spgs) equals number of structures
    initRadius=None,
    meanVolume=None,
    volRatio=1.5
    ):

    if not initRadius:
        # radius = [atomRadii[atom] for atom in symbols]
        initRadius = [covalent_radii[atomic_numbers[atom]] for atom in symbols]

    if not meanVolume:
        meanVolume = 0
        for i in range(len(symbols)):
            meanVolume = meanVolume + 4*math.pi/3*(initRadius[i]**3)*formula[i]
    meanVolume = volRatio * meanVolume

    volume=[meanVolume*0.5, meanVolume*1.5]

    radius = [random.uniform(0.6,0.8)*r for r in initRadius]
    # logging.info("%s %s %s %s %s"%(symbols, formula, radius, volume, spgs))
    rndPop= spgGen_per_formula(symbols, formula, radius, volume, spgs)
    return rndPop

def build_struct_per_formula(
    symbols,
    formula,
    radius,
    meanVolume,
    spgs, #a list of spacegroups, len(spgs) equals number of structures
    toP1=0,
    ):
    volume=[meanVolume*0.5, meanVolume*1.5]

    radius = [random.uniform(0.6,0.8)*r for r in radius]
    logging.info("%s %s %s %s %s"%(symbols, formula, radius, volume, spgs))
    randomPop = spgGen(symbols, formula, radius, volume, spgs, toP1)
    for i, ind in enumerate(randomPop):
        # sortInd = Atoms(sorted(list(ind), key=lambda atom: atomic_numbers[atom.symbol]), pbc=True)
        sortInd = Atoms(sorted(list(ind), key=lambda atom: symbols.index(atom.symbol)))
        sortInd.set_cell(ind.get_cell())
        # ind = Atoms(sorted(list(ind), key=lambda atom: symbols.index(atom.symbol)))
        randomPop[i] = sortInd
    logging.info("Build " + str(len(randomPop)) + ' structures')

    return randomPop

def build_struct(
    popSize,
    symbols,
    formula,
    numFrml = [1],
    meanVolume = None, #Assume meanVolume is a float, volume of one formula
    radius = None, #Assume radium is a list
    spgs = range(1,231),
    volRatio = 1.5,
    tryNum = 10
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

    meanVolume = volRatio * meanVolume

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


    # Build structure with high symm
    for i in range(tryNum):
        if popSize <= len(buildPop):
            break
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
        buildPop.extend(randomPop)


    # Allow P1 structure
    if popSize > len(buildPop):
        randomPop = []
        for i in range(popSize - len(buildPop)):
            nfm = random.choice(numFrml)
            inputVolume = nfm * meanVolume
            inputFrml = [nfm*frml for frml in formula]
            chooseSpg = random.sample(spgs, 1)
            randomPop.extend(build_struct_per_formula(symbols, inputFrml, radius, inputVolume, chooseSpg, toP1=1))
            if len(randomPop) > 0:
                randomPop[-1].info['numOfFormula'] = nfm
                randomPop[-1].info['symbols'] = symbols
                randomPop[-1].info['formula'] = formula
                randomPop[-1].info['parentE'] = 0
        buildPop.extend(randomPop)

        for ind in buildPop:
            ind.info['toCalc'] = True


    os.chdir('..')

    return buildPop


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

def varcomp_build(popSize, symbols, minAt, maxAt, formula, invFrml, fullEles=False, spgs=range(2,231), trySpgNum=10, tryFrmlNum=100, volRatio=1.5):
    """Variable composition"""
    os.chdir('BuildStruct')
    randomPop = list()
    for i in range(popSize):
        for j in range(tryFrmlNum):
            numAt = random.randrange(minAt, maxAt+1)
            # print("numAt: %s"%(numAt))
            nums = np.random.rand(len(symbols))
            nums /= nums.sum()
            eles = np.rint(nums * numAt)
            eles[-1] = numAt - eles[:-1].sum()
            # num2 = 1 - ele1
    #        inputSym, inputFrml = zip(*filter(lambda x: x[1] > 0, zip(symbols, [ele1, ele2])))
            # check block
            npFrml = np.array(formula)
            coef = np.rint(np.dot(np.dot(np.expand_dims(eles,axis=0), npFrml.T), invFrml))
            tmpEles = np.dot(coef, npFrml)
            tmpSum = tmpEles.sum()
            if tmpSum < minAt or tmpSum > maxAt:
                continue
            else:
                eles = tmpEles[0]

            # make sure that structures contain all species if fullEles == True
            if fullEles and 0 in tmpEles[0]:
                continue


            inputSym = symbols
            inputFrml = eles.tolist()
            inputFrml = [int(i) for i in inputFrml]
            inputSpgs = random.sample(spgs, trySpgNum)
            frmlPop = build_struct_per_formula2(inputSym, inputFrml, inputSpgs, volRatio=volRatio)
            # print("trying composition: {}".format(inputFrml))
            logging.info("trying composition: {}".format(inputFrml))
            if len(frmlPop) > 0:
                rndIndex = random.randrange(0, len(frmlPop))
                chsInd = frmlPop[rndIndex]
                chsInd.info['symbols'] = symbols
                chsInd.info['formula'] = inputFrml
                chsInd.info['numOfFormula'] = 1
                chsInd.info['parentE'] = 0
                chsInd.info['Origin'] = 'random'
                randomPop.append(chsInd)
                # print("Build random strucuture in spacegroup {} for {}".format(inputSpgs[rndIndex], chsInd.get_chemical_formula()))
                logging.info("Build random strucuture in spacegroup {} for {}".format(inputSpgs[rndIndex], chsInd.get_chemical_formula()))
                break
        # randomPop.extend(build_struct(1, inputSym, inputFrml))
    os.chdir('..')
    return randomPop



def read_seeds(parameters, seedFile='Seeds/POSCARS'):

    seedPop = []
    setSym = parameters['symbols']
    setFrml = parameters['formula']
    minAt = parameters['minAt']
    maxAt = parameters['maxAt']
    calcType = parameters['calcType']

    if os.path.exists(seedFile):
        readPop = ase.io.read(seedFile, index=':', format='vasp-xdatcar')
        if len(readPop) > 0:
            logging.info("Reading Seeds ...")

        seedPop = read_bare_atoms(readPop, setSym, setFrml, minAt, maxAt, calcType)

        # if calcType == 'fix':
        #     setGcd = reduce(fractions.gcd, setFrml)
        #     setRd = [x/setGcd for x in setFrml]

        # for ind in readPop:
        #     selfSym, selfFrml = symbols_and_formula(ind)
        #     # logging.debug('selfSym: {!r}'.format(selfSym))
        #     symDic = dict(zip(selfSym, selfFrml))
        #     for sym in [s for s in setSym if s not in selfSym]:
        #         symDic[sym] = 0

        #     if False not in map(lambda x: x in setSym, selfSym):
        #         ind.info['symbols'] = setSym
        #     else:
        #         logging.info("ERROR in checking symbols")
        #         continue

        #     if calcType == 'var':
        #         if minAt <= len(ind) <= maxAt or len(selfSym) < len(setSym):
        #             formula = [symDic[sym] for sym in setSym]
        #             ind.info['formula'] = formula
        #             seedPop.append(ind)
        #         else:
        #             logging.info("ERROR in checking number of atoms")

        #     elif calcType == 'fix':
        #         if len(selfSym) == len(setSym):
        #             formula = [symDic[sym] for sym in setSym]
        #             logging.info('formula: {!r}'.format(formula))
        #             selfGcd = reduce(fractions.gcd, formula)
        #             selfRd = [x/selfGcd for x in formula]
        #             if selfRd == setRd:
        #                 ind.info['formula'] = formula
        #                 ind.info['numOfFormula'] = len(ind)/sum(formula)
        #                 seedPop.append(ind)
        #             else:
        #                 logging.info("ERROR in checking formula")

    logging.info("Read Seeds: %s"%(len(seedPop)))
    return seedPop
