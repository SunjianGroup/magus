# For spacegroup probability update: 
# "What is the according layergroup number of a specific spacegroup"

lyr_spg = {
    1: 'p1',	2: 'p-1',	3: 'p112',	4: 'p11m',	5: 'p11a', 
    6: 'p112/m',	7: 'p112/a',	8: 'p211',	9: 'p2111',	10: 'c211', 
    11: 'pm11',	12: 'pb11',	13: 'cm11',	14: 'p2/m11',	15: 'p21/m11', 
    16: 'p2/b11',	17: 'p21/b11',	18: 'c2/m11',	19: 'p222',	20: 'p2122', 
    21: 'p21212',	22: 'c222',	23: 'pmm2',	24: 'pma2',	25: 'pba2', 
    26: 'cmm2',	27: 'pm2m',	28: 'pm21b',	29: 'pb21m',	30: 'pb2b', 
    31: 'pm2a',	32: 'pm21n',	33: 'pb21a',	34: 'pb2n',	35: 'cm2m', 
    36: 'cm2e',	37: 'pmmm',	38: 'pmaa',	39: 'pban',	40: 'pmam', 
    41: 'pmma',	42: 'pman',	43: 'pbaa',	44: 'pbam',	45: 'pbma', 
    46: 'pmmn',	47: 'cmmm',	48: 'cmme',	49: 'p4',	50: 'p-4', 
    51: 'p4/m',	52: 'p4/n',	53: 'p422',	54: 'p4212',	55: 'p4mm', 
    56: 'p4bm',	57: 'p-42m',	58: 'p-421m',	59: 'p-4m2',	60: 'p-4b2', 
    61: 'p4/mmm',	62: 'p4/nbm',	63: 'p4/mbm',	64: 'p4/nmm',	65: 'p3', 
    66: 'p-3',	67: 'p312',	68: 'p321',	69: 'p3m1',	70: 'p31m', 
    71: 'p-31m',	72: 'p-3m1',	73: 'p6',	74: 'p-6',	75: 'p6/m', 
    76: 'p622',	77: 'p6mm',	78: 'p-6m2',	79: 'p-62m',	80: 'p6/mmm'
}


from ase.spacegroup import Spacegroup
mapper = {}
for i in range(1,230):
    sg = Spacegroup(i)
    mapper[''.join(sg.symbol.split()).lower()] = i

lyr_spg = {
    1: 'p1',	2: 'p-1',	3: 'p2',	4: 'pm',	5: 'pc', 
    6: 'p2/m',	7: 'p2/c',	8: 'p2',	9: 'p21',	10: 'c2', 
    11: 'pm',	12: 'pc',	13: 'cm',	14: 'p2/m',	15: 'p21/m', 
    16: 'p2/c',	17: 'p21/c',	18: 'c2/m',	19: 'p222',	20: 'p2221', 
    21: 'p21212',	22: 'c222',	23: 'pmm2',	24: 'pma2',	25: 'pba2', 
    26: 'cmm2',	27: 'pmm2',	28: 'pmc21',	29: 'pmc21',	30: 'pcc2', 
    31: 'pma2',	32: 'pmn21',	33: 'pca21',	34: 'pnc2',	35: 'cmm2', 
    36: 'ccc2',	37: 'pmmm',	38: 'pmaa',	39: 'pban',	40: 'pmam', 
    41: 'pmma',	42: 'pman',	43: 'pbaa',	44: 'pbam',	45: 'pbma', 
    46: 'pmmn',	47: 'cmmm',	48: 'cmme',	49: 'p4',	50: 'p-4', 
    51: 'p4/m',	52: 'p4/n',	53: 'p422',	54: 'p4212',	55: 'p4mm', 
    56: 'p4bm',	57: 'p-42m',	58: 'p-421m',	59: 'p-4m2',	60: 'p-4b2', 
    61: 'p4/mmm',	62: 'p4/nbm',	63: 'p4/mbm',	64: 'p4/nmm',	65: 'p3', 
    66: 'p-3',	67: 'p312',	68: 'p321',	69: 'p3m1',	70: 'p31m', 
    71: 'p-31m',	72: 'p-3m1',	73: 'p6',	74: 'p-6',	75: 'p6/m', 
    76: 'p622',	77: 'p6mm',	78: 'p-6m2',	79: 'p-62m',	80: 'p6/mmm'
}
for lg in lyr_spg:
    print(lg, mapper[lyr_spg[lg]])
