from magus.utils import load_plugins
from pathlib import Path


can_check = ['calculators', 'comparators', 'fingerprints']
def checkpack(tocheck='all', *args, **kwargs):
    if tocheck == 'all':
        for pack in can_check:
            path = Path(__file__).parent.parent.joinpath(pack, '__init__.py')
            load_plugins(path, 'magus.' + pack, verbose=True)
    else:
        path = Path(__file__).parent.parent.joinpath(tocheck, '__init__.py')
        load_plugins(path, 'magus.' + tocheck, verbose=True)
