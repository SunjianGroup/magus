from magus.utils import load_plugins
from pathlib import Path
from magus.utils import COMPARATOR_PLUGIN, COMPARATOR_CONNECT_PLUGIN, FINGERPRINT_PLUGIN, CALCULATOR_PLUGIN, CALCULATOR_CONNECT_PLUGIN

check_list = {
    'calculators': [CALCULATOR_PLUGIN, CALCULATOR_CONNECT_PLUGIN], 
    'comparators': [COMPARATOR_PLUGIN, COMPARATOR_CONNECT_PLUGIN], 
    'fingerprints': [FINGERPRINT_PLUGIN],
    }
def checkpack(tocheck='all', *args, **kwargs):
    if tocheck == 'all':
        for pack in check_list:
            path = Path(__file__).parent.parent.joinpath(pack, '__init__.py')
            load_plugins(path, 'magus.' + pack, verbose=True)
            for plugin in check_list[pack]:
                print(plugin)
    else:
        path = Path(__file__).parent.parent.joinpath(tocheck, '__init__.py')
        load_plugins(path, 'magus.' + tocheck, verbose=True)
        for plugin in check_list[tocheck]:
                print(plugin)
