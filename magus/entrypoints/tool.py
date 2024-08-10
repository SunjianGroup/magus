from ..reconstruct.tools import getslab, analyze, mine_substrate, inputslab

def tool(*args, **kwargs):
    if kwargs['getslab']: 
        kwargs['filename'] = kwargs['filename'] or 'Ref/layerslices.traj'
        getslab(*args, **kwargs)
    elif kwargs['analyze']:
        kwargs['filename'] = kwargs['filename'] or 'results'
        analyze(*args, **kwargs)
    elif kwargs['mine_substrate']:
        kwargs['filename'] = kwargs['filename'] or 'Ref/layerslices.traj'
        mine_substrate(*args, **kwargs)
    elif kwargs['inputslab']:
        kwargs['filename'] = kwargs['filename'] or 'inputslab.vasp'
        kwargs['sliceslab'] = kwargs['sliceslab']
        inputslab(*args, **kwargs)
