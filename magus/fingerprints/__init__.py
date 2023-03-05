import logging
from magus.utils import load_plugins, FINGERPRINT_PLUGIN


log = logging.getLogger(__name__)


def get_fingerprint(p_dict):
    load_plugins(__file__, 'magus.fingerprints')
    if 'Fingerprint' in p_dict:
        name = p_dict['Fingerprint']['name']
        return FINGERPRINT_PLUGIN[name](**{**p_dict, **p_dict['Fingerprint']})
    else:
        return FINGERPRINT_PLUGIN['nepdes'](symbols=p_dict['symbols'])
