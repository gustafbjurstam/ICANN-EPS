import numpy as np
from material_models import sigeps330 as sigeps33

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # global data
    data = np.vstack(np.loadtxt("Foam_volstr_stress.txt"))

    def uniaxial_test(strain_rate, total_strain, dt, uparams, rho, rho0, sigo, yfunc):
        num_steps = int(total_strain / (strain_rate * dt))
        epsp = np.zeros(6)
        eps = np.zeros(6)
        stress_strain = []

        # Loading phase (positive strain)
        for step in range(num_steps):
            epsp[0] = strain_rate
            eps[0] += strain_rate * dt
            rho = rho0/(1+eps[0])
            sign, _ = sigeps33(dt, uparams, rho, rho0, epsp, eps, sigo, yfunc)
            stress_strain.append((eps[0], sign[0]))
            sigo = sign

        # Unloading phase (negative strain)
        for step in range(2*num_steps):
            epsp[0] = -strain_rate
            eps[0] -= strain_rate * dt
            rho = rho0/(1+eps[0])
            sign, _ = sigeps33(dt, uparams, rho, rho0, epsp, eps, sigo, yfunc)
            stress_strain.append((eps[0], sign[0]))
            sigo = sign

        # Reloading to zero strain
        for step in range(num_steps):
            epsp[0] = strain_rate
            eps[0] += strain_rate * dt
            rho = rho0/(1+eps[0])
            sign, _ = sigeps33(dt, uparams, rho, rho0, epsp, eps, sigo, yfunc)
            stress_strain.append((eps[0], sign[0]))
            sigo = sign

        return stress_strain

    def cyclic_test(uparams, rho, rho0, sigo, yfunc):
        epsp = np.zeros(6)
        eps = np.zeros(6)
        dt = 1e-3
        strains = -np.concatenate((np.linspace(0.0,0.2,21),np.linspace(0.2,0.0,21), np.linspace(0.0,0.7,71), np.linspace(0.7,0.0,71), np.linspace(0.0,0.95,96),np.linspace(0.95,0.0,101)))
        stress_strain = []
        E = uparams[6]
        for strain in strains:
            uparams[6] = np.max([E,np.abs(yfunc(strain) - yfunc(eps[0]))/np.abs(strain - eps[0]+1e-10)])
            epsp[0] = (strain-eps[0])/dt
            eps[0] = strain
            rho = rho0/(1+eps[0])
            sign, _ = sigeps33(dt, uparams, rho, rho0, epsp, eps, sigo, yfunc)
            stress_strain.append((-eps[0], -sign[0]))
            sigo = sign
        
        return stress_strain
    
    def yfunc(eps):
        return np.interp(eps, -data[::-1,0],data[::-1,1])
    
    # Parameters for the tests
    strain_rates = [-0.1]
    total_strain = -0.95
    dt = 1e-3
    uparams = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0]
    rho = 1.0
    rho0 = 1.0
    sigo = np.zeros(6)

    # Perform the tests and store the results
    # stress_strain_results = []
    # for strain_rate in strain_rates:
    #     stress_strain = uniaxial_test(strain_rate, total_strain, dt, uparams, rho, rho0, sigo, yfunc)
    #     stress_strain_results.append((strain_rate, stress_strain))

    # plt.figure()
    # i = 0

    # for strain_rate, stress_strain in stress_strain_results:
    #     strains, stresses = zip(*stress_strain)
    #     colors = np.linspace(0, 1, len(stress_strain))
    #     maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    #     plt.scatter(strains, stresses, c=colors, cmap=maps[i], label=f'Strain rate: {strain_rate}', marker='o')
    #     i += 1

    # plt.xlabel('Strain')
    # plt.ylabel('Stress (MPa)')
    # plt.title('Stress-Strain Curves for Different Strain Rates')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    stress_strain_results2 = []
    for strain_rate in strain_rates:
        stress_strain = cyclic_test(uparams, rho, rho0, sigo, yfunc)
        stress_strain_results2.append((strain_rate, stress_strain))

    plt.figure()
    i = 0

    for strain_rate, stress_strain in stress_strain_results2:
        strains, stresses = zip(*stress_strain)
        colors = np.linspace(0, 1, len(stress_strain))
        maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        plt.scatter(strains, stresses, c=colors, cmap=maps[i], label=f'Strain rate: {strain_rate}', marker='o')
        i += 1

    plt.xlabel('Compressive Strain')
    plt.ylabel('Compressive Stress (MPa)')
    plt.title('Stress-Strain Curves for Different Strain Rates')
    plt.legend()
    plt.grid(True)
    plt.show()