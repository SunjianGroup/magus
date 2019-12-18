from __future__ import print_function, division
import linecache
import logging
def read_eigen(eigen = 'EIGENVAL'):

    linecache.clearcache()
    l6 = linecache.getline(eigen, 6).split()
    eN = int(l6[0])
    filledBand = int(eN/2)
    nK = int(l6[1])
    nB = int(l6[2])
    #print eN, filledBand, nK, nB
    for i in range(nK):
        if i == 0:
            line = filledBand + 8
    #        print line
            gapStart = float(linecache.getline(eigen, line).split()[1])
            gapEnd = float(linecache.getline(eigen, line+1).split()[1])
        else:
            line = i*(nB+2) + filledBand + 8
    #        print line
            newStart = float(linecache.getline(eigen, line).split()[1])
            newEnd = float(linecache.getline(eigen, line+1).split()[1])

            gapStart = newStart if gapStart < newStart else gapStart
            gapEnd = newEnd if gapEnd > newEnd else gapEnd

    eGap = gapEnd - gapStart if gapEnd - gapStart > 0.1 else 0
#    logging.info(gapStart, gapEnd, eGap)
    return eGap
