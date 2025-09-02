import numpy as np
from walk3D import full_load_path
import os

if __name__ == "__main__":
    principal = False # Choose whether to walk in principal space
    rho0 = 1.0
    data = np.vstack(np.loadtxt("Foam_volstr_stress.txt"))
    def yfunc(eps):
        return np.interp(eps, -data[::-1,0],data[::-1,1])
    uparams = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, yfunc(0.0)]

    rng = np.random.default_rng(seed = 42)
    output_dir = "random_paths_6D"
    if principal:
        output_dir = "random_paths_3D"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(50):
        strains, stresses, turns = full_load_path(uparams, rho0, yfunc, principal, rng=rng)
        strains += np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        np.savez(os.path.join(output_dir, f"path_{i}.npz"), strains=strains, stresses=stresses)
        print(len(strains))


