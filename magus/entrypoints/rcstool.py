from ..reconstruct.tools import getslab, analyze

def rcstool(*args, **kwargs):
    if kwargs['getslab']: 
        kwargs['filename'] = kwargs['filename'] or 'Ref/layerslices.traj'
        getslab(*args, **kwargs)
    elif kwargs['analyze']:
        kwargs['filename'] = kwargs['filename'] or 'results'
        analyze(*args, **kwargs)
