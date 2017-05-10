from __future__ import print_function, division
import os
import re
import sys
import logging
import yaml
import numpy as np
import ase.io
from ase import Atoms

def write_results(Pop, gen, prefix, resultsDir = "results"):
    '''
    Pop: population to write,
    gen: generation,
    prefix = 'gen', 'pareto', 'good'
    resultsDir: directory to save results
    '''

    if not os.path.exists(resultsDir):
        os.mkdir(resultsDir)
    # os.chdir(resultsDir)

    # write_py("%s/%s%s.py" %(resultsDir, prefix, gen), Pop)
    write_yaml("{}/{}{}.yaml".format(resultsDir, prefix, gen), Pop)

    # formulas = set([ind.get_chemical_formula() for ind in Pop])
    # for frml in formulas:
    #     writePop = [ind for ind in Pop if ind.get_chemical_formula() == frml]

    #     # ase.io.write("%s%s_%s.traj" %(prefix, gen, frml), writePop, 'traj')
    #     write_py("%s%s_%s.py" %(prefix, gen, frml), writePop)
    # ase.io.write('pareto0.traj', paretoPop, 'traj')
    # ase.io.write('Gen0.traj', optPop, 'traj')
    # os.chdir('..')

def write_py(fileobj, images, **kwargs):
    """Write to ASE-compatible python script."""
    if isinstance(fileobj, str):
        fileobj = open(fileobj, 'w')

    fileobj.write('from ase import Atoms\n\n')
    fileobj.write('import numpy as np\n\n')
    # fileobj.write('from numpy import array\n\n')

    if not isinstance(images, (list, tuple)):
        images = [images]
    fileobj.write('images = [\n')

    for image in images:
        fileobj.write("    Atoms(symbols='%s',\n"
                      "          pbc=%s,\n"
                      "          cell=\n      %s,\n"
                      "          positions=\n      %s,\n"
                      "          info=%s\n),\n"% (
            image.get_chemical_formula(mode='reduce'),
            re.sub(r'array', 'np.array', repr(image.pbc)),
            re.sub(r'array', 'np.array', repr(image.cell)),
            re.sub(r'array', 'np.array', repr(image.positions)),
            re.sub(r'array', 'np.array', repr(image.info))))

    fileobj.write(']')

def read_py(filename):
    # modulename = filename[:-3]
    # exec("from %s import images" %(modulename))
    # logging.info(filename)
    os.system("cp %s read_temp.py"% (filename))
#    if 'read_temp' in sys.modules:
#        logging.info("read_temp in sys.modules")
    import read_temp
    reload(read_temp)
    images = read_temp.images
    os.system("rm read_temp.py*")
    return images

def write_xsf(filename, image):
    """Write to xsf file for fingerprint calculation"""
    cell = image.get_cell()
    with open(filename, 'w') as fileobj:
        fileobj.write("# total energy = 0 eV\n\n")
        fileobj.write("CRYSTAL\n")
        fileobj.write("PRIMVEC\n")

        for i in range(3):
            fileobj.write("%f %f %f\n" %(cell[i][0], cell[i][1], cell[i][2],))

        fileobj.write("PRIMCOORD\n")
        fileobj.write("%s 1\n"%(len(image)))

        for atom in image:
            pos = atom.position
            fileobj.write("%s %f %f %f 0 0 0\n" %(atom.symbol, pos[0], pos[1], pos[2]))

def write_dataset(dataPop, filename="dataset.yaml", resultsDir = "results"):

    if os.path.exists("%s/%s"%(resultsDir, filename)):
        dataDic = yaml.load(open("%s/%s"%(resultsDir, filename)))
        saveFps = dataDic['data']
        saveEns = dataDic['value']
    else:
        dataDic = dict()
        saveFps = list()
        saveEns = list()

    inFps = [ind.info['fingerprint'] for ind in dataPop]
    inFps = [fp.flatten().tolist() for fp in inFps]
    inEns = [ind.info['enthalpy'] for ind in dataPop]

    saveFps.extend(inFps)
    saveEns.extend(inEns)
    dataDic['data'] = saveFps
    dataDic['value'] = saveEns

    with open("%s/%s"%(resultsDir, filename), 'w') as f:
        f.write(yaml.dump(dataDic))

def read_dataset(filename="dataset.yaml", resultsDir = "results"):

    dataDic = yaml.load(open("%s/%s"%(resultsDir, filename)))
    data = np.array(dataDic['data'])
    value = np.array(dataDic['value'])

    return data, value

def write_yaml(filename, images):

    if not isinstance(images, (list, tuple)):
        images = [images]

    writeList = list()
    for atoms in images:
        struct = dict()
        struct['cell'] = atoms.get_cell()
        struct['positions'] = atoms.get_positions()
        struct['numbers'] = atoms.get_atomic_numbers()
        struct['pbc'] = atoms.get_pbc()

        for key, val in struct.items():
            struct[key] = val.tolist()
        
        info = atoms.info
        # for key, val in info.items():
        #     if isinstance(val, np.ndarray):
        #         info[key] = val.astype(float)
        
        writeList.append((struct, info))
    
    with open(filename, 'w') as fileObj:
        fileObj.write(yaml.dump(writeList))

def read_yaml(filename):

    readList = yaml.load(open(filename))

    images = list()
    for struct, info in readList:
        image = Atoms(**struct)
        image.info = info
        images.append(image)
    
    return images




        
