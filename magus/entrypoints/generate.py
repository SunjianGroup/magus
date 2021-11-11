from magus.parameters import magusParameters
from ase.io import write

def generate(*args, input_file='input.yaml', number=10, 
             output_file='gen.traj', **kwargs):
    parameters = magusParameters(input_file)
    atoms_generator = parameters.RandomGenerator
    new_frames = atoms_generator.generate_pop(number)
    write(output_file, new_frames)
