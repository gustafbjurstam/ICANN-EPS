import numpy as np
import random
from material_models import sigeps332 as sigeps33
# random.seed(1)

def run_path(uparams, rho0, yfunc):
    # Generate random a setup
    max_strain = random.uniform(0.0, 0.8)
    min_strain = random.uniform(-1.0, 0.0)
    dt = 1e-2
    direction = random.choice([1, -1])

    random_ratio = 0.0

    #  Initialize the stress and strain
    strain = np.zeros(6)
    stress = np.zeros(6)
    epsp = np.zeros(6)
    strains = []
    stresses = []
    rho = rho0 * 1.0
    # Run the path
    if direction == 1:
        while strain[0] < max_strain:
            strains.append(strain[0])
            stresses.append(stress[0])
            epsp[0] = (1.0+random_ratio*random.uniform(-1.0, 1.0))
            strain += dt * epsp
            rho = rho0 / np.prod(1 + np.diag(strain))
            stress, _ = sigeps33(dt, uparams, rho, rho0, epsp, strain, stress, yfunc)
        while strain[0] > min_strain:
            strains.append(strain[0])
            stresses.append(stress[0])
            epsp[0] = -(1.0+random_ratio*random.uniform(-1.0, 1.0))
            strain += dt * epsp
            rho = rho0 / np.prod(1 + np.diag(strain))
            stress, _ = sigeps33(dt, uparams, rho, rho0, epsp, strain, stress, yfunc)
    else:
        while strain[0] > min_strain:
            strains.append(strain[0])
            stresses.append(stress[0])
            epsp[0] = -(1.0+random_ratio*random.uniform(-1.0, 1.0))
            strain += dt * epsp
            rho = rho0 / np.prod(1 + np.diag(strain))
            stress, _ = sigeps33(dt, uparams, rho, rho0, epsp, strain, stress, yfunc)
        while strain[0] < max_strain:
            strains.append(strain[0])
            stresses.append(stress[0])
            epsp[0] = (1.0+random_ratio*random.uniform(-1.0, 1.0))
            strain += dt * epsp
            rho = rho0 / np.prod(1 + np.diag(strain))
            stress, _ = sigeps33(dt, uparams, rho, rho0, epsp, strain, stress, yfunc)

    return strains, stresses



if __name__ == "__main__":
    rho0 = 1.0
    data = np.vstack(np.loadtxt("Foam_volstr_stress.txt"))
    def yfunc(eps):
        return np.interp(eps, -data[::-1,0],data[::-1,1])
    uparams = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, yfunc(0.0)]

    strains, stresses = run_path(uparams, rho0, yfunc)

    import matplotlib.pyplot as plt
    plt.figure()
    colors = np.linspace(0, 1, len(strains))
    plt.scatter(strains, stresses, c=colors, cmap='plasma', marker='o')
    plt.xlabel('Strain')
    plt.ylabel('Stress (MPa)')
    plt.title('Load Path')
    plt.grid(True)
    plt.show()
