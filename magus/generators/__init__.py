from .random import SPGGenerator, MoleculeSPGGenerator
import logging


__all__ = ['SPGGenerator', 'MoleculeSPGGenerator']

log = logging.getLogger(__name__)


def get_random_generator(p_dict):
    if p_dict['molMode']:
        return MoleculeSPGGenerator(**p_dict)
    else:
        return SPGGenerator(**p_dict)

