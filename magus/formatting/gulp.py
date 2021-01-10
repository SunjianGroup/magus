import numpy as np
from ase.atoms import Atoms


def dump_gulp(frames, filename, symbol_to_type, mode='w'):
    with open(filename, mode) as f:
        f.write('cell\n')
        a, b, c, alpha, beta, gamma = frames.get_cell_lengths_and_angles()
        f.write("%g %g %g %g %g %g\n" %(a, b, c, alpha, beta, gamma))
        f.write('fractional\n')
        for atom in frames:
            f.write("%s %.6f %.6f %.6f\n" %(atom.symbol, atom.a, atom.b, atom.c))
        f.write('\n')


def load_gulp(filename):
    with open(filename) as f:
        lines = f.readlines()

    cycles = -1
    self.optimized = None
    for i, line in enumerate(lines):
        m = re.match(r'\s*Total lattice energy\s*=\s*(\S+)\s*eV', line)
        if m:
            energy = float(m.group(1))
            self.results['energy'] = energy
            self.results['free_energy'] = energy

        elif line.find('Optimisation achieved') != -1:
            self.optimized = True

        elif line.find('Final Gnorm') != -1:
            self.Gnorm = float(line.split()[-1])

        elif line.find('Cycle:') != -1:
            cycles += 1

        elif line.find('Final Cartesian derivatives') != -1:
            s = i + 5
            forces = []
            while(True):
                s = s + 1
                if lines[s].find("------------") != -1:
                    break
                if lines[s].find(" s ") != -1:
                    continue
                g = lines[s].split()[3:6]
                G = [-float(x) * eV / Ang for x in g]
                forces.append(G)
            forces = np.array(forces)
            self.results['forces'] = forces

        elif line.find('Final cartesian coordinates of atoms') != -1:
            s = i + 5
            positions = []
            while True:
                s = s + 1
                if lines[s].find("------------") != -1:
                    break
                if lines[s].find(" s ") != -1:
                    continue
                xyz = lines[s].split()[3:6]
                XYZ = [float(x) * Ang for x in xyz]
                positions.append(XYZ)
            positions = np.array(positions)
            self.atoms.set_positions(positions)

    self.steps = cycles

        fout = open('optimized.structure', 'r')
        output = fout.readlines()
        fout.close()

        for i, line in enumerate(output):
            if 'cell' in line:
                cellIndex = i + 1
            if 'fractional' in line:
                posIndex = i + 1

        cellpar = output[cellIndex].split()
        cellpar = [float(par) for par in cellpar]

        pos = []
        for line in output[posIndex:posIndex + len(calcInd)]:
            pos.append([eval(i) for i in line.split()[2:5]])

        optInd = crystal(symbols=calcInd, cellpar=cellpar)
        optInd.set_scaled_positions(pos)
        optInd.info = calcInd.info.copy()

        # relaxsteps = os.popen("grep Cycle output | tail -1 | awk '{print $2}'").readlines()[0]
        # logging.debug('gulp relax steps:{}'.format(relaxsteps))
        enthalpy = os.popen("grep 'Total lattice enthalpy .* eV' output | tail -1 | awk '{print $5}'").readlines()[0]
        enthalpy = float(enthalpy)
        volume = optInd.get_volume()
        energy = enthalpy - pressure * GPa * volume
        optInd.info['energy'] = energy
        optInd.info['enthalpy'] = round(enthalpy/len(optInd), 3)

        #TODO The following code are adapted from ASE, need modification
        with open('output') as f:
            lines = f.readlines()
        for i,line in enumerate(lines):
            if 'Final internal derivatives' in line:
                s = i + 5
                break
        forces = []
        while(True):
            s = s + 1
            if "------------" in lines[s]:
                break
            g = lines[s].split()[3:6]                    
            G = [-float(x) * eV / Ang for x in g]
            forces.append(G)
        forces = np.array(forces)
        optInd.info['forces'] = forces
        return optInd

    except:
        logging.warning("traceback.format_exc():\n{}".format(traceback.format_exc()))
        logging.warning("GULP fail")
        return None