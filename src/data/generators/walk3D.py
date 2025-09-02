# 3Dwalk.py

import numpy as np
# import random
from material_models import sigeps332 as sigeps33

def full_load_path(uparams, rho0, yfunc, principal=False, rng = np.random.default_rng()):
    # Determinisitic factors
    k = 6
    if principal: k = 3
    dt = 1e-3
    random_ratio = 0.0
    turns_per_component = 3

    # Generate random setup
    # rng = np.random.default_rng()
    direction = rng.integers(0,600, k)
    direction = (-1)**direction
    target = direction * rng.uniform(0.0, 0.9, k)

    # Initialise
    eps = np.zeros(6)
    sig = np.zeros(6)
    strains = np.array([eps])
    stresses = np.array([sig])
    turns = np.zeros(k)
    
    def walk(eps, sig, target, rho0, dt, random_ratio, uparams, yfunc, direction, k):
        maxed = np.array([False]*k)
        strains = np.array([eps])
        stresses = np.array([sig])
        epsp = np.zeros(6)
        while not np.any(maxed):
            epsp[:k] = (direction + random_ratio*rng.uniform(-1,1,k))
            eps = eps + dt * epsp
            rho = rho0 / np.prod(1 + eps[:3])
            # rho = rho0 / (1 + np.sum(eps[:3]))
            sig,_ = sigeps33(dt, uparams, rho, rho0, epsp, eps, sig, yfunc)
            strains = np.vstack([strains, eps])
            stresses = np.vstack([stresses, sig])
            maxed = eps[:k] * direction > target * direction
        return strains, stresses, maxed
    
    while np.any(turns < turns_per_component):
        new_strains, new_stresses, maxed = walk(eps, sig, target, rho0, dt, random_ratio, uparams, yfunc, direction,k)
        strains = np.vstack([strains, new_strains])
        stresses = np.vstack([stresses, new_stresses])
        turns += maxed
        eps = new_strains[-1]
        sig = new_stresses[-1]
        for i in range(k):
            if maxed[i]:
                direction[i] = -direction[i]
                target[i] = direction[i] * rng.uniform(0.0, 0.9)
    return strains, stresses, turns

if __name__ == "__main__":
    principal = False # Choose whether to walk in principal space
    rho0 = 1.0
    data = np.vstack(np.loadtxt("Foam_volstr_stress.txt"))
    def yfunc(eps):
        return np.interp(eps, -data[::-1,0],data[::-1,1])
    uparams = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, yfunc(0.0)]

    strains, stresses, turns = full_load_path(uparams, rho0, yfunc, principal)
    print(turns)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    colors = np.linspace(0, 1, len(strains))
    components = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
    for i in range(6):
        ax = axs[i // 3, i % 3]
        ax.scatter(strains[:, i], stresses[:, i], c=colors, cmap='plasma', marker='o')
        ax.set_xlabel('Strain')
        ax.set_ylabel('Stress (MPa)')
        ax.set_title(f'Load Path Component ' + components[i])
        ax.grid(True)
    plt.tight_layout()
    plt.show()
