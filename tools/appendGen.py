#!/usr/bin/env python
import sys, os, shutil
import ase.io
import logging

#logging.basicConfig(level=logging.DEBUG, format="%(asctime)s   %(message)s",datefmt='%H:%M:%S')

assert len(sys.argv) == 4, 'wrong arguement type: source > target'  
source, target = (sys.argv[1], sys.argv[3]) if (sys.argv[2] == 'to' or sys.argv[2] =='t') else (sys.argv[3], sys.argv[1])

print('append {} to {}'.format(source, target))

#unique = ['best', 'good', 'allparameters.yaml']
print('append bestpop ...')
bestpop1 = ase.io.read('{}/best.traj'.format(target), index = ':')
bestpop2 = ase.io.read('{}/best.traj'.format(source), index = ':')
bestpop2.extend(bestpop1)
ase.io.write('{}/best.traj'.format(target), bestpop2, format = 'traj')

print('append goodpop ...')
from magus.parameters import magusParameters
goodpop1 = ase.io.read('{}/good.traj'.format(target), index = ':')
goodpop2 = ase.io.read('{}/good.traj'.format(source), index = ':')
goodpop2.extend(goodpop1)

parameters = magusParameters('input.yaml')
Population = parameters.get_Population()
goodpop = Population(goodpop2)
goodpop.del_duplicate()
goodpop.save(filename='good',gen='',savedir=target)

print('move parameters.yaml ...')
if os.path.exists('{}/allparameters.yaml'.format(target)):
    shutil.move('{}/allparameters.yaml'.format(target), '{}/allparameters1.yaml'.format(target))

if os.path.exists('{}/allparameters1.yaml'.format(target)):
    i=1
    while os.path.exists("{}/allparameters{}.yaml".format(target, i)):
        i+=1
    shutil.move('{}/allparameters.yaml'.format(source), '{}/allparameters{}.yaml'.format(target, i))


files = ['initpop', 'raw','savegood', 'keep', 'gen'] 

def dirNumGen(dirname):

    for i in range(1,1000):
        for name in files:
            if not os.path.exists('{}/{}{}.traj'.format(dirname,name,i)):
                i = i-1 if name == files[0] else i
                return i
    

targetGen = dirNumGen(target)

print('append generations ... {} gens in {} in total.'.format(targetGen, target))
for i in range(1,1000):
    for name in files:
        if os.path.exists('{}/{}{}.traj'.format(source,name,i)):
            shutil.move('{}/{}{}.traj'.format(source,name,i), '{}/{}{}.traj'.format(target,name,i + targetGen))

print('Finish!')
