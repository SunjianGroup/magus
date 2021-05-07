from ase.io import write


def write_traj(filename, images, delTraj=True):
    writeImages = []
    for atoms in images:
        writeAtoms = atoms.copy()
        info = writeAtoms.info
        # delete the trajectories in info to reduce size
        if delTraj and 'trajs' in info.keys():
            info['trajs'] = []
        writeImages.append(writeAtoms)
    write(filename, images, format='traj')
