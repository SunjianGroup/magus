from magus.parameters import magusParameters
from ase.io import read, write

def calculate(*args, filename=None, input_file='input.yaml', 
              mode='relax', **kwargs):
    parameters = magusParameters('input.yaml')
    to_calc = read(filename)
    try:
        calc = parameters.get_MLCalculator()
    except:
        calc = parameters.get_MainCalculator()
    if mode == 'relax':
        calced = calc.relax(to_calc)
    else:
        calced = calc.scf(to_calc)
    write('out.traj', calced)

