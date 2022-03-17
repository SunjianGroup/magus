from magus.parameters import magusParameters
from ase.io import read, write


def calculate(*args, filename=None, input_file='input.yaml', 
              output_file='out.traj', mode='relax', pressure=None, **kwargs):
    parameters = magusParameters(input_file)
    if pressure is not None:
        parameters.p_dict['pressure'] = pressure
    to_calc = read(filename, index=':')
    try:
        calc = parameters.MLCalculator
    except:
        calc = parameters.MainCalculator
    if mode == 'relax':
        calced = calc.relax(to_calc)
    else:
        calced = calc.scf(to_calc)
    write(output_file, calced)

