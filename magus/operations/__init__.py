from .crossovers import *
from .mutations import *


op_list = ['CutAndSplicePairing', 'ReplaceBallPairing', 
           'SoftMutation', 'PermMutation', 'LatticeMutation', 'RippleMutation', 'SlipMutation',
           'RotateMutation', 'RattleMutation', 'FormulaMutation', 
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
    return operators
