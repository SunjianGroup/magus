import logging
from magus.utils import load_plugins, FINGERPRINT_PLUGIN


load_plugins(__file__, 'magus.fingerprints')


log = logging.getLogger(__name__)


def get_fingerprint(p_dict):    
    if 'Fingerprint' in p_dict:
        name = p_dict['Fingerprint']['name']
        return FINGERPRINT_PLUGIN[name](**{**p_dict, **p_dict['Fingerprint']})
    else:
        return FINGERPRINT_PLUGIN['soap'](p_dict['symbols'])
