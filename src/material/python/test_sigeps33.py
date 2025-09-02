import numpy as np
from material_models import sigeps331 as sigeps33

# Testing sigeps33defaultYield
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Function to perform uniaxial tension-compression test
    def uniaxial_test(strain_rate, total_strain, dt, uparams, rho, rho0, sigo):
        num_steps = int(total_strain / (strain_rate * dt))
        epsp = np.zeros(6)
        eps = np.zeros(6)
        stress_strain = []

        # Loading phase (positive strain)
        for step in range(num_steps):
            epsp[0] = strain_rate
            eps[0] += strain_rate * dt
            rho = rho0/(1+eps[0])
            sign, _ = sigeps33(dt, uparams, rho, rho0, epsp, eps, sigo)
            stress_strain.append((eps[0], sign[0]))
            sigo = sign

        # Unloading phase (negative strain)
        for step in range(2*num_steps):
            epsp[0] = -strain_rate
            eps[0] -= strain_rate * dt
            rho = rho0/(1+eps[0])
            sign, _ = sigeps33(dt, uparams, rho, rho0, epsp, eps, sigo)
            stress_strain.append((eps[0], sign[0]))
            sigo = sign

        # Reloading to zero strain
        for step in range(num_steps):
            epsp[0] = strain_rate
            eps[0] += strain_rate * dt
            rho = rho0/(1+eps[0])
            sign, _ = sigeps33(dt, uparams, rho, rho0, epsp, eps, sigo)
            stress_strain.append((eps[0], sign[0]))
            sigo = sign

        return stress_strain

    # Function to perform uniaxial tension-hold test
    def uniaxial_tension_hold_test(strain_rate, total_time, total_strain, dt, uparams, rho, rho0, sigo):
        num_steps_total = int(total_time / dt)
        num_steps_loading = int((total_strain) / (strain_rate * dt))
        num_steps_hold = num_steps_total - num_steps_loading
        epsp = np.zeros(6)
        eps = np.zeros(6)
        stress_time = []

        # Loading phase (positive strain)
        for step in range(num_steps_loading):
            epsp[0] = strain_rate
            eps[0] += strain_rate * dt
            rho = rho0/(1+eps[0])
            sign, _ = sigeps33(dt, uparams, rho, rho0, epsp, eps, sigo)
            stress_time.append((step * dt, sign[0]))
            sigo = sign

        # Holding phase (constant strain)
        for step in range(num_steps_hold):
            epsp[0] = 0.0
            rho = rho0/(1+eps[0])
            sign, _ = sigeps33(dt, uparams, rho, rho0, epsp, eps, sigo)
            stress_time.append(((num_steps_loading + step) * dt, sign[0]))
            sigo = sign

        return stress_time

    # Parameters for the tests
    strain_rates = [0.1, 0.01]
    total_strain = 0.2
    dt = 1e-3
    uparams = [0.0, 0.0, 0.0, 1e30, 0.0, 0.0, 200, 0, 0, 2, 1000.0, 1000.0]
    rho = 1.0
    rho0 = 1.0
    sigo = np.zeros(6)

    # Perform the tests and store the results
    stress_strain_results = []
    stress_time_results = []

    for strain_rate in strain_rates:
        stress_strain = uniaxial_test(strain_rate, total_strain, dt, uparams, rho, rho0, sigo)
        stress_strain_results.append((strain_rate, stress_strain))

    # Calculate total time based on the lowest strain rate
    lowest_strain_rate = min(strain_rates)
    total_time = total_strain / lowest_strain_rate + 3 * (total_strain / lowest_strain_rate)

    for strain_rate in strain_rates:
        stress_time = uniaxial_tension_hold_test(strain_rate, total_time, total_strain, dt, uparams, rho, rho0, sigo)
        stress_time_results.append((strain_rate, stress_time))

    # Plot the results
    plt.figure()
    i = 0

    for strain_rate, stress_strain in stress_strain_results:
        strains, stresses = zip(*stress_strain)
        colors = np.linspace(0, 1, len(stress_strain))
        maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        plt.scatter(strains, stresses, c=colors, cmap=maps[i], label=f'Strain rate: {strain_rate}', marker='o')
        i += 1

    plt.xlabel('Strain')
    plt.ylabel('Stress (Pa)')
    plt.title('Stress-Strain Curves for Different Strain Rates')
    plt.legend()
    plt.grid(True)

    plt.figure()
    i = 0

    for strain_rate, stress_time in stress_time_results:
        times, stresses = zip(*stress_time)
        colors = np.linspace(0, 1, len(stress_time))
        maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        plt.scatter(times, stresses, c=colors, cmap=maps[i], label=f'Strain rate: {strain_rate}', marker='o')
        i += 1

    plt.xlabel('Time (s)')
    plt.ylabel('Stress (Pa)')
    plt.title('Stress-Time Curves for Different Strain Rates')
    plt.legend()
    plt.grid(True)

    plt.show()
