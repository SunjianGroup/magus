from .crossovers import *
from .mutations import *


op_list = ['CutAndSplicePairing', 'ReplaceBallPairing', 
           'SoftMutation', 'PermMutation', 'LatticeMutation', 'RippleMutation', 'SlipMutation',
           'RotateMutation', 'RattleMutation', 'FormulaMutation', 
           'LyrSlipMutation', 'LyrSymMutation', 'ShellMutation', 'CluSymMutation',
           ]

def remove_end(op_name):
    if op_name.endswith('Pairing'):
        return op_name[:-7].lower()
    elif op_name.endswith('Mutation'):
        return op_name[:-8].lower()

locals_ = locals()    # 部分python版本把locals()写入生成器会导致问题
op_dict = {remove_end(op_name): locals_[op_name] for op_name in op_list}


def get_default_op(p_dict):
    operators = {}
    for key in ['cutandsplice', 'slip', 'lattice', 'ripple', 'rattle']:
        operators[key] = {}
    if len(p_dict['symbols']) > 1:
        operators['perm'] = {}
    if p_dict['molDetector'] > 0:
        operators['rotate'] = {}
    if p_dict['formulaType'] == 'var':
        operators['formula'] = {}
    # if self.parameters.calcType=='rcs':
    #     op_nums['lattce'] = 0
    #     op_nums['formula'] = num if not self.parameters.chkMol and len(self.parameters.symbols) > 1 else 0
    #     op_nums['lyrslip'] = num
    #     op_nums['lyrsym'] = num
    # if self.parameters.calcType=='clus':
    #     op_nums['slip'] = 0
    #     #op_nums['soft'] = num
    #     op_nums['shell'], op_nums['clusym'] = [num]*2
    #     operations['ripple'] = RippleMutation(rho=0.05)
    #     operations['rattle'] = RattleMutation(p=0.25,rattle_range=0.8,dRatio=self.parameters.dRatio)
    return operators
