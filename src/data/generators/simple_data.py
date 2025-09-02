import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.material.python.material_models import sigeps332
import os

def update_other_axes(stretch,variant = 0):
    if variant == 0:
        # only one axis moves
        return stretch
    elif variant == 1:
        # No volume change
        stretch[1] = 1 / np.sqrt(stretch[0])
        stretch[2] = 1 / np.sqrt(stretch[0])
        return stretch
    elif variant == 2:
        # Two axes moves together# - third geometric mean
        stretch[1] = stretch[0]
        # stretch[2] = 1 / np.sqrt(stretch[0] * stretch[1])
        return stretch
    elif variant == 3:
        # One two axes moves as cube root
        stretch[1] = stretch[0] ** (-1 / 4)
        stretch[2] = stretch[0] ** (-1 / 4)
        return stretch
    elif variant == 4:
        # All move together
        stretch[1] = stretch[0]
        stretch[2] = stretch[0]
        return stretch

def test1(variant = 0):
    # Cyclic uniaxial test

    # Inputs
    dt = 6.5e-3
    # stops = [0.7, 0.9, 0.5, 0.7, 0.2, 1.4, 1.0, 1.7, 1.0]
    # stops = [0.7, 0.77, 0.3, 0.44]
    # stops = [0.4, 0.42]
    stops = [0.55, 0.57]


    # Material parameters
    data = np.vstack(np.loadtxt("data/raw/Foam_volstr_stress.txt"))
    def yfunc(eps):
        return np.interp(eps, -data[::-1,0],data[::-1,1])
    # uparams = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, yfunc(0.0)]
    uparams = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20, yfunc(0.0)]

    # Initial conditions
    rho = 1.0
    rho0 = 1.0
    sigo = np.zeros(6)
    stretch = np.array([1.0,1.0,1.0,0.0,0.0,0.0])
    stretch0 = stretch.copy()
    speed = 1.0
    stretches = np.array([stretch])
    stresses  = np.array([sigo])
    old_stretch = stretch.copy()

    # Loop
    for stop in stops:
        direction = np.sign(stop - stretch[0])
        strain_rate = direction * speed
        while direction * (stop - stretch[0]) > 0:
            stretch[0] += strain_rate * dt
            stretch = update_other_axes(stretch, variant)
            epsp = (stretch - old_stretch) / dt
            eps = stretch - stretch0
            rho = rho0/(np.prod(stretch[:3]))
            sigo, _ = sigeps332(dt, uparams, rho, rho0, epsp, eps, sigo, yfunc)
            stretches = np.vstack([stretches, stretch])
            stresses = np.vstack([stresses, sigo])
            old_stretch = stretch.copy()

    return stretches, stresses

def plot_path(stretches, stresses):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    colors = np.linspace(0, 1, len(stretches))
    components = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
    for i in range(6):
        ax = axs[i // 3, i % 3]
        ax.scatter(stretches[:, i], stresses[:, i], c=colors, cmap='plasma', marker='o')
        ax.set_xlabel('Stretch')
        ax.set_ylabel('Stress (MPa)')
        ax.set_title(f'Load Path Component ' + components[i])
        ax.grid(True)
    plt.tight_layout()
    plt.show()

    return None

if __name__ == "__main__":
    variant = 2
    stretches, stresses = test1(variant)
    plot_path(stretches**2, stresses)
    print("Shape: ",stretches.shape)

    for k in range(2,3):
        stretches, stresses = test1(variant=k)
        # Save the data
        directory = "data/simple_paths"
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savez(f"{directory}/path_{k}.npz", strains=stretches, stresses=stresses)
        print("Variant ",k," saved!")
    print("All data saved successfully!")

