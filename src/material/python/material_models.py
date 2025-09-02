# material_models.py

import numpy as np

def sigeps331(dt,uparams, rho, rho0, epsp, eps, sigo):
    # Inputs:
        # dt: time step
        # uparams: material parameters
        # rho: density
        # rho0: reference density
        # epsp: strain rate tensor
        # eps: strain tensor
        # sigo: original stress tensor
    # Outputs:
        # sign: updated stress tensor
        # soundsp: sound speed

    # Material parameters
        # gamma0: initial volumetric strain
        # phi: ratio of foam to polymer density
        # P0: initial air pressure
        # A,B,C: yield parameters
        # KEN: analysis type (always 1 here)
        # EY: Young's modulus
        # C1: Coefficient for Young's modulus update. [Pa s]
        # C2: Coefficient for Young's modulus update. [Pa]
        # ET: Tangent modulus
        # Vmu: Viscocity in pure compression
        # Vmu0: viscosity in pure shear

    # Use vectors instead of tensors

    # Unpack material parameters
    gamma0 = uparams[0]
    phi = uparams[1]
    P0 = uparams[2]
    A = uparams[3]
    B = uparams[4]
    C = uparams[5]
    EY = uparams[6]
    C1 = uparams[7]
    C2 = uparams[8]
    ET = uparams[9]
    VMU = uparams[10]
    VMU0 = uparams[11]

    # Compute the volumetric strain-air pressure
    gama = rho0/rho -1+gamma0
    sigair = np.maximum(0, -(P0*gama)/(1+gama - phi))

    # Case 2 - viscoelastic region
    # compute stiffnesses
    edot    = np.max(np.abs(epsp))
    E       = max(C1*edot+C2,EY)
    EPET    = (E+ET)     / VMU
    EMET    = (E*ET)     / VMU
    EPETS   = (E+ET)     / VMU0
    EMETS   = 2 * (E*ET) / VMU0

    # Compute yield stress
    SYIELD = np.abs(A+B*(1+C*gama)) # This is just one case, see line 318 in fortran code
    sigs = sigo.copy()
    sigs[:3] += sigair 
    # shear components (xy, xz, yz) remain from sigo

    de = eps.copy()
    de[3:] *= 0.5  # halve the shear terms only

    dedt = epsp.copy()
    dedt[3:] *= 0.5  # similarly halve the shear terms 
    # de is the strain tensor (that is e12 = gamma12/2)
    # dedt is the strain rate tensor

    # Compute stress rates
    dsdt = np.zeros(6)
    dsdt[:3] = E * dedt[:3] - EPET * sigs[:3] + EMET * de[:3]
    dsdt[3:] = E * dedt[3:] - EPETS * sigs[3:] + EMETS * de[3:]

    # Compute trail stress
    sigs = sigs+dsdt*dt

    # Transform stress to principal coordinates
    stress_matrix = np.matrix([
    [sigs[0], sigs[3], sigs[4]],
    [sigs[3], sigs[1], sigs[5]],
    [sigs[4], sigs[5], sigs[2]]
    ])
    sigpr, dirpr = np.linalg.eig(stress_matrix)

    # Yield criteria - scaling
    sigpr = np.minimum(SYIELD, abs(sigpr))*np.sign(sigpr)

    # Transform back to original coordinates
    # line 406
    stress_matrix = np.transpose(dirpr * np.diag(sigpr) * np.transpose(dirpr))
    sigs[:3] = np.diag(stress_matrix)
    sigs[3]  = stress_matrix[0,1]
    sigs[4]  = stress_matrix[0,2]
    sigs[5]  = stress_matrix[1,2]

    # Compute updated stress tensor and sound speed
    soundsp = np.sqrt(E/rho0)
    sign = sigs.copy()
    sign[:3] -= sigair

    return sign, soundsp

def sigeps330(dt,uparams, rho, rho0, epsp, eps, sigo, yfunc):
    # Inputs:
        # dt: time step
        # uparams: material parameters
        # rho: density
        # rho0: reference density
        # epsp: strain rate tensor
        # eps: strain tensor
        # sigo: original stress tensor
        # yfunc: custom yield function
    # Outputs:
        # sign: updated stress tensor
        # soundsp: sound speed

    # Material parameters
        # gamma0: initial volumetric strain
        # phi: ratio of foam to polymer density
        # P0: initial air pressure
        # A,B,C: yield parameters Irelevant for custom yield function
        # KEN: analysis type (always 0 here)
        # EY: Young's modulus
        # C1: Coefficient for Young's modulus update. [Pa s] Irrellevant for custom yield function
        # C2: Coefficient for Young's modulus update. [Pa]  Irrellevant for custom yield function
        # ET: Tangent modulus Irrellevant for custom yield function
        # Vmu: Viscocity in pure compression Irrellevant for elastic material
        # Vmu0: viscosity in pure shear Irrellevant for elastic material

    # Use vectors instead of tensors

    # Unpack material parameters
    gamma0 = uparams[0]
    phi = uparams[1]
    P0 = uparams[2]
    EY = uparams[6]

    # Compute the volumetric strain-air pressure
    gama = rho0/rho -1+gamma0
    sigair = np.maximum(0, -(P0*gama)/(1+gama - phi)) #zero

    # Compute yield stress
    SYIELD = yfunc(gama) # This is just one case, see line 318 in fortran code
    sigs = sigo.copy()
    sigs[:3] += sigair 
    # shear components (xy, xz, yz) remain from sigo

    de = eps.copy()
    de[3:] *= 0.5  # halve the shear terms only

    dedt = epsp.copy()
    dedt[3:] *= 0.5  # similarly halve the shear terms 
    # de is the strain tensor (that is e12 = gamma12/2)
    # dedt is the strain rate tensor

    # Compute stress rates
    dsdt = np.zeros(6)
    dsdt = EY * dedt

    # Compute trail stress
    sigs = sigs+dsdt*dt

    # Transform stress to principal coordinates
    stress_matrix = np.matrix([
    [sigs[0], sigs[3], sigs[4]],
    [sigs[3], sigs[1], sigs[5]],
    [sigs[4], sigs[5], sigs[2]]
    ])
    sigpr, dirpr = np.linalg.eig(stress_matrix)

    # Yield criteria - scaling
    sigpr = np.minimum(SYIELD, abs(sigpr))*np.sign(sigpr)

    # Transform back to original coordinates
    # line 406
    stress_matrix = np.transpose(dirpr * np.diag(sigpr) * np.transpose(dirpr))
    sigs[:3] = np.diag(stress_matrix)
    sigs[3]  = stress_matrix[0,1]
    sigs[4]  = stress_matrix[0,2]
    sigs[5]  = stress_matrix[1,2]

    # Compute updated stress tensor and sound speed
    soundsp = np.sqrt(EY/rho0)
    sign = sigs.copy()
    sign[:3] -= sigair

    return sign, soundsp

def sigeps332(dt,uparams, rho, rho0, epsp, eps, sigo, yfunc):
    # Inputs:
        # dt: time step
        # uparams: material parameters
        # rho: density
        # rho0: reference density
        # epsp: strain rate tensor
        # eps: strain tensor
        # sigo: original stress tensor
        # yfunc: custom yield function
    # Outputs:
        # sign: updated stress tensor
        # soundsp: sound speed

    # Material parameters
        # gamma0: initial volumetric strain
        # phi: ratio of foam to polymer density
        # P0: initial air pressure
        # A,B,C: yield parameters Irelevant for custom yield function
        # KEN: analysis type (always 0 here)
        # EY: Young's modulus
        # C1: Coefficient for Young's modulus update. [Pa s] Irrellevant for custom yield function
        # C2: Coefficient for Young's modulus update. [Pa]  Irrellevant for custom yield function
        # ET: Tangent modulus Irrellevant for custom yield function
        # Vmu: Viscocity in pure compression Irrellevant for elastic material
        # Vmu0: viscosity in pure shear Irrellevant for elastic material
        # ten_max: maximum tensile stress

    # Use vectors instead of tensors

    # Unpack material parameters
    gamma0 = uparams[0]
    phi = uparams[1]
    P0 = uparams[2]
    EY = uparams[6]
    sigt_cutoff = uparams[7]

    # Compute the volumetric strain-air pressure
    gama = rho0/rho -1+gamma0
    sigair = np.maximum(0, -(P0*gama)/(1+gama - phi)) #zero

    # Compute yield stress
    SYIELD = yfunc(gama) # This is just one case, see line 318 in fortran code
    sigs = sigo.copy()
    sigs[:3] += sigair 
    # shear components (xy, xz, yz) remain from sigo

    de = eps.copy()
    de[3:] *= 0.5  # halve the shear terms only

    dedt = epsp.copy()
    dedt[3:] *= 0.5  # similarly halve the shear terms 
    # de is the strain tensor (that is e12 = gamma12/2)
    # dedt is the strain rate tensor

    # Compute stress rates
    dsdt = np.zeros(6)
    dsdt = EY * dedt

    # Compute trail stress
    sigs = sigs+dsdt*dt

    # Transform stress to principal coordinates
    stress_matrix = np.matrix([
    [sigs[0], sigs[3], sigs[4]],
    [sigs[3], sigs[1], sigs[5]],
    [sigs[4], sigs[5], sigs[2]]
    ])
    sigpr, dirpr = np.linalg.eig(stress_matrix)

    # Yield criteria - scaling
    sigpr = np.minimum(SYIELD, abs(sigpr))*np.sign(sigpr)
    sigpr = np.minimum(sigt_cutoff, sigpr)
    # if gama > 0: # Creates discontinuties for multi-axial loading
    #     sigpr = np.maximum(0, sigpr)

    # Transform back to original coordinates
    # line 406
    stress_matrix = np.transpose(dirpr * np.diag(sigpr) * np.transpose(dirpr))
    sigs[:3] = np.diag(stress_matrix)
    sigs[3]  = stress_matrix[0,1]
    sigs[4]  = stress_matrix[0,2]
    sigs[5]  = stress_matrix[1,2]

    # Compute updated stress tensor and sound speed
    soundsp = np.sqrt(EY/rho0)
    sign = sigs.copy()
    sign[:3] -= sigair

    return sign, soundsp