from magus.parameters import magusParameters
from ase.io import write

def generate(*args, input_file='input.yaml', number=10, 
             output_file='gen.traj', **kwargs):
    parameters = magusParameters(input_file)
    population = parameters.Population
    atoms_generator = population.atoms_generator
    
    new_frames = atoms_generator.generate_pop(number)
    for i, atoms in enumerate(new_frames):
        new_frames[i] = population.Ind(atoms).for_calculate()

    write(output_file, new_frames)
