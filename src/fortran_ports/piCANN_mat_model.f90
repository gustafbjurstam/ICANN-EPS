module piCANN_mat_model_module
    use iso_fortran_env, only: real64
    use piCANN_weights
    implicit none
    private

    ! LAPACK interface
    interface
        subroutine dsyev(jobz, uplo, n, a, lda, w, work, lwork, info)
            import real64
            character, intent(in) :: jobz, uplo
            integer, intent(in) :: n, lda, lwork
            integer, intent(out) :: info
            real(real64), intent(inout) :: a(lda,*)
            real(real64), intent(out) :: w(*), work(*)
        end subroutine dsyev
    end interface

    public :: piCANN_mat_model
contains

subroutine piCANN_mat_model(C, history, sigma, nElem)
    ! This function implements the piCANN material model
    ! Inputs
    ! C: The right Cauchy-Green deformation tensor in Voigt notation (6,nElem)
    ! history: The history vector contains at entries (1:6,nElem) the inelastic strain tensor in Voigt notation
    !          and at entry (7,nElem) initial guess for the plastic multiplier
    ! nElem: The number of elements in the simulation
    ! Outputs
    ! sigma: Sigma contains the second Piola-Kirchhoff stress tensor in Voigt notation (6,nElem)
    ! Input and output variables
    integer, intent(in)         :: nElem
    real(real64), intent(in)    :: C(6,nElem)
    real(real64), intent(inout) :: history(7,nElem)
    real(real64), intent(out)   :: sigma(6,nElem)

    ! Variables used in the trail step
    real(real64) :: Ci_old(6,nElem)
    real(real64) :: C_matrix(3,3,nElem)
    real(real64) :: Ci_old_matrix(3,3,nElem), D_matrix(3,3)
    real(real64) :: Ce_matrix(3,3,nElem), sigma_matrix(3,3,nElem)
    real(real64) :: Ui_matrix(3,3,nElem), Ui_inv_matrix(3,3,nElem)
    real(real64) :: dpsi_dCe(3,3,nElem), dpsi_dCi(3,3,nElem)
    real(real64) :: mandel(3,3,nElem), backstress(3,3,nElem), gamma(3,3,nElem)
    real(real64) :: phi(nElem)

    ! Variables used in the plastic correction step
    real(real64) :: lambda_dot1, lambda_dot2, phi1, phi2, dlambda_dot, dphi, update
    real(real64) :: mandel_temp(3,3), backstress_temp(3,3), gamma_temp(3,3)
    real(real64) :: Ci_temp(3,3), Ce_temp(3,3), Ui_temp(3,3), Ui_inv_temp(3,3)
    real(real64) :: dpsi_dCe_temp(3,3), dpsi_dCi_temp(3,3)

    ! Variables used for final stress output
    real(real64) :: S_matrix(3,3), U_matrix(3,3), U_inv_matrix(3,3)

    ! Parameters
    real(real64), parameter :: dx = 1.0e-8_real64
    real(real64), parameter :: tol = 1.0e-7_real64
    integer,      parameter :: max_steps = 500 ! Perhaps this should be an argument passed to the subroutine

    ! Iterators
    integer ::  i, k

    ! Initialize matrices
    
    ! Expect input (6,nElem) -- if this is not the case, adjust later code accordingly
    Ci_old = history(1:6, :)

    ! Initialize symmetric matrices
    do i = 1,nElem 
        ! Total RCG matrix
        C_matrix(1,1,i) = C(1,i)
        C_matrix(2,2,i) = C(2,i)
        C_matrix(3,3,i) = C(3,i)
        C_matrix(1,2,i) = C(4,i)
        C_matrix(2,1,i) = C(4,i)
        C_matrix(1,3,i) = C(5,i)
        C_matrix(3,1,i) = C(5,i)
        C_matrix(2,3,i) = C(6,i)
        C_matrix(3,2,i) = C(6,i)

        ! Old inelastic strain matrix
        Ci_old_matrix(1,1,i) = Ci_old(1,i)
        Ci_old_matrix(2,2,i) = Ci_old(2,i)
        Ci_old_matrix(3,3,i) = Ci_old(3,i)
        Ci_old_matrix(1,2,i) = Ci_old(4,i)
        Ci_old_matrix(2,1,i) = Ci_old(4,i)
        Ci_old_matrix(1,3,i) = Ci_old(5,i)
        Ci_old_matrix(3,1,i) = Ci_old(5,i)
        Ci_old_matrix(2,3,i) = Ci_old(6,i)
        Ci_old_matrix(3,2,i) = Ci_old(6,i)
    end do

    ! Compute matrix square root of Ci_old
    do i = 1,nElem
        call matrix_sqrt_spd(Ci_old_matrix(:,:,i), Ui_matrix(:,:,i), Ui_inv_matrix(:,:,i))
    end do

    ! Compute multiplicative split
    do i = 1,nElem
        Ce_matrix(:,:,i) = matmul(matmul(Ui_inv_matrix(:,:,i), C_matrix(:,:,i)), Ui_inv_matrix(:,:,i))
    end do

    ! Compute derivatives of Helmholtz free energy to get stress state
    do k = 1,nElem
        dpsi_dCe(:,:,k)     = matrix_derivative_of_psi(Ce_matrix(:,:,k), Ci_old_matrix(:,:,k), dx, 1)
        dpsi_dCi(:,:,k)     = matrix_derivative_of_psi(Ce_matrix(:,:,k), Ci_old_matrix(:,:,k), dx, 2)
        mandel(:,:,k)       = 2 * matmul(Ce_matrix(:,:,k), dpsi_dCe(:,:,k))
        backstress(:,:,k)   = 2 * matmul(dpsi_dCi(:,:,k), Ci_old_matrix(:,:,k))
    end do
    gamma = mandel - backstress

    ! Check if the stress state is plastically admissible
    do i = 1,nElem
        phi(i) = yield_function(gamma(:,:,i))
        if (phi(i) > 0) then
            ! Stress state is not admissible, plastic correction needed
            ! Compute the derivative of the yield function
            D_matrix = matrix_derivative_of_yield_function(gamma(:,:,i), dx)

            ! Find the correct multiplier for the plastic correction using secant method
            lambda_dot1 = 0.0_real64
            if (history(7,i) <= 1.0e-7_real64) then
                ! If the previous multiplier is zero initialise it to 1e-3
                lambda_dot2 = 1.0e-2_real64
            else
                lambda_dot2 = history(7,i) ! Use the previous multiplier as the initial guess
            end if
            ! Initialising lambda_dot1 as 0 means that the initial state is already computed
            phi1 = phi(i)
            k = 0
            do while (abs(phi1) > tol) ! Convergence criterion can be modified
                ! Compute the new stress state
                Ci_temp = plastic_flow(D_matrix, Ui_matrix(:,:,i), lambda_dot2)
                call matrix_sqrt_spd(Ci_temp, Ui_temp, Ui_inv_temp)
                Ce_temp = matmul(matmul(Ui_inv_temp, C_matrix(:,:,i)), Ui_inv_temp)
                
                dpsi_dCe_temp   = matrix_derivative_of_psi(Ce_temp, Ci_temp, dx, 1)
                dpsi_dCi_temp   = matrix_derivative_of_psi(Ce_temp, Ci_temp, dx, 2)
                mandel_temp     = 2 * matmul(Ce_temp, dpsi_dCe_temp)
                backstress_temp = 2 * matmul(dpsi_dCi_temp, Ci_temp)
                gamma_temp      = mandel_temp - backstress_temp
                phi2 = yield_function(gamma_temp)
                ! Update the lambda_dot using secant method
                dlambda_dot = lambda_dot2 - lambda_dot1
                lambda_dot1 = lambda_dot2
                dphi = phi2 - phi1
                phi1 = phi2
                update = phi2 * dlambda_dot / dphi
                if (abs(update) > 0.99 * lambda_dot2) then
                    ! This stabilisation comes with the nice property that the multiplier is always positive
                    update = 0.9 * sign(lambda_dot2, update)
                end if

                lambda_dot2 = lambda_dot2 - update
                k = k + 1
                if (k > max_steps) then
                    ! The print statement is for debugging purposes, it can be removed in production code
                    print *, "Warning: Maximum number of iterations reached in plastic correction at element ", i, " with lambda_dot2 = ", &
                    lambda_dot2, " and phi2 = ", phi2
                    exit ! Exit the loop if maximum iterations are reached
                end if
            end do
            history(7,i) = lambda_dot2 ! Store the plastic multiplier in history as initial guess for the next step
            S_matrix = 2 * matmul(matmul(Ui_inv_temp, dpsi_dCe_temp), Ui_inv_temp) ! Compute the second Piola-Kirchhoff stress
            history(1,i) = Ci_temp(1,1)
            history(2,i) = Ci_temp(2,2)
            history(3,i) = Ci_temp(3,3)
            history(4,i) = Ci_temp(1,2)
            history(5,i) = Ci_temp(1,3)
            history(6,i) = Ci_temp(2,3)
        else
            S_matrix = 2 * matmul(matmul(Ui_inv_matrix(:,:,i), dpsi_dCe(:,:,i)), Ui_inv_matrix(:,:,i)) ! Compute the second Piola-Kirchhoff stress
            history(1:6,i) = Ci_old(:,i) ! Update history to the old inelastic strain
        end if
        sigma_matrix(:,:,i) = S_matrix
    end do

    ! Compress the stress matrix to the output vector
    do i = 1,nElem
        sigma(1,i) = sigma_matrix(1,1,i)
        sigma(2,i) = sigma_matrix(2,2,i)
        sigma(3,i) = sigma_matrix(3,3,i)
        ! Ensure that the output is symmetric
        ! This is not strictly necessary, but it is good practice
        sigma(4,i) = (sigma_matrix(1,2,i) + sigma_matrix(2,1,i)) * 0.5_real64
        sigma(5,i) = (sigma_matrix(1,3,i) + sigma_matrix(3,1,i)) * 0.5_real64
        sigma(6,i) = (sigma_matrix(2,3,i) + sigma_matrix(3,2,i)) * 0.5_real64
    end do

    
    end subroutine piCANN_mat_model

subroutine matrix_sqrt_spd(A,B,B_inv)
    real(real64), intent(in)  :: A(3,3)
    real(real64), intent(out) :: B(3,3), B_inv(3,3)
    real(real64) :: Q(3,3), D(3,3), D_inv(3,3), A_temp(3,3)
    real(real64) :: D_vec(3)
    integer :: info, i
    real(real64) :: work(128) ! Workspace for LAPACK, uncertain about what size to use, I want to avoid dynamic allocation
    integer :: lwork = 128 ! Size of the workspace array
    ! computes the square root of an spd matrix A using eigen decomposition and LAPACK
    A_temp = A
    call dsyev("V","U",3,A_temp,3,D_vec,work,lwork,info)
    if (info /= 0) then
        print *, "Error in matrix_sqrt_spd: "
        print *, "Error in dysev: ", info
        stop
    end if
    ! Construct diagonal matrix D
    D = 0.0
    D_inv = 0.0
    Q = A_temp
    do i = 1,3
        D(i,i) = sqrt(D_vec(i))
        D_inv(i,i) = 1.0_real64 / D(i,i)
    end do
    ! Compute B = Q * D * Q^T
    B       = matmul(Q, matmul(D, transpose(Q)))
    B_inv   = matmul(Q, matmul(D_inv, transpose(Q)))

end subroutine matrix_sqrt_spd

function Helmholtz(Ce,Ci) result(psi)
    ! Compute the Helmholtz free energy based on the weights of the piCANN model
    real(real64), intent(in)  :: Ce(3,3), Ci(3,3)
    real(real64)              :: psi
    ! Potentially, this function could be split into elastic and inelastic contributions
    ! This would accelerate the computation of the derivatives

    ! Get invariants of the input tensors
    real(real64) :: I1_e, I2_e, I3_e
    real(real64) :: I1_i, I2_i, I3_i
    ! Get modified invariants
    real(real64) :: I1_bar_e, I2_bar_e
    real(real64) :: I1_bar_i, I2_bar_i

    I1_e = Ce(1,1) + Ce(2,2) + Ce(3,3)
    I2_e = Ce(1,1)*Ce(2,2) + Ce(1,1)*Ce(3,3) + Ce(2,2)*Ce(3,3) - Ce(1,2)**2- Ce(1,3)**2 - Ce(2,3)**2
    I3_e = Ce(1,1)*Ce(2,2)*Ce(3,3) + 2.0*Ce(1,2)*Ce(1,3)*Ce(2,3) - Ce(1,1)*Ce(2,3)**2 - Ce(1,2)**2*Ce(3,3) - Ce(1,3)**2*Ce(2,2)

    I1_i = Ci(1,1) + Ci(2,2) + Ci(3,3)
    I2_i = Ci(1,1)*Ci(2,2) + Ci(1,1)*Ci(3,3) + Ci(2,2)*Ci(3,3) - Ci(1,2)**2- Ci(1,3)**2 - Ci(2,3)**2
    I3_i = Ci(1,1)*Ci(2,2)*Ci(3,3) + 2.0*Ci(1,2)*Ci(1,3)*Ci(2,3) - Ci(1,1)*Ci(2,3)**2 - Ci(1,2)**2*Ci(3,3) - Ci(1,3)**2*Ci(2,2)

    I1_bar_e = I1_e / (I3_e**(1.0/3.0)) - 3.0
    I2_bar_e = I2_e / (I3_e**(2.0/3.0)) - 3.0
    I1_bar_i = I1_i / (I3_i**(1.0/3.0)) - 3.0
    I2_bar_i = I2_i / (I3_i**(2.0/3.0)) - 3.0

    ! Compute elastic contribution
    psi = 0.0
    psi = psi + hNet_e_iso_net1_iso1_cp1(1) * I1_bar_e + &
          hNet_e_iso_net1_iso1_cp1(2) * I1_bar_e**2
    psi = psi + hNet_e_iso_net1_iso1_cp3(1) * exp(hNet_e_iso_net1_iso1_cp2(1) * I1_bar_e) + &
          hNet_e_iso_net1_iso1_cp3(2) * exp(hNet_e_iso_net1_iso1_cp2(2) * I1_bar_e**2)

    psi = psi + hNet_e_iso_net2_iso2_cp1(1) * I2_bar_e + &
          hNet_e_iso_net2_iso2_cp1(2) * I2_bar_e**2
    psi = psi + hNet_e_iso_net2_iso2_cp3(1) * exp(hNet_e_iso_net2_iso2_cp2(1) * I2_bar_e) + &
          hNet_e_iso_net2_iso2_cp3(2) * exp(hNet_e_iso_net2_iso2_cp2(2) * I2_bar_e**2)

    psi = psi + convex_poly_expansion(I3_e - 1.0, hNet_e_vol_net_vol_coefficients, &
        hNet_e_vol_net_vol_biases)

    ! Compute inelastic contribution
    psi = psi + hNet_i_iso_net1_iso1_cp1(1) * I1_bar_i + &
          hNet_i_iso_net1_iso1_cp1(2) * I1_bar_i**2
    psi = psi + hNet_i_iso_net1_iso1_cp3(1) * exp(hNet_i_iso_net1_iso1_cp2(1) * I1_bar_i) + &
          hNet_i_iso_net1_iso1_cp3(2) * exp(hNet_i_iso_net1_iso1_cp2(2) * I1_bar_i**2)

    psi = psi + hNet_i_iso_net2_iso2_cp1(1) * I2_bar_i + &
          hNet_i_iso_net2_iso2_cp1(2) * I2_bar_i**2
    psi = psi + hNet_i_iso_net2_iso2_cp3(1) * exp(hNet_i_iso_net2_iso2_cp2(1) * I2_bar_i) + &
          hNet_i_iso_net2_iso2_cp3(2) * exp(hNet_i_iso_net2_iso2_cp2(2) * I2_bar_i**2)

    psi = psi + convex_poly_expansion(I3_i - 1.0, hNet_i_vol_net_vol_coefficients, &
        hNet_i_vol_net_vol_biases)

end function Helmholtz

function convex_poly_expansion(x, coeffs, biases) result(result)
    ! Compute the convex polynomial expansion
    real(real64), intent(in)  :: x
    real(real64), intent(in)  :: coeffs(:)
    real(real64), intent(in)  :: biases(:)
    real(real64) :: result, p0, p0_prime, c, b
    integer :: i, terms, p

    result = 0.0_real64
    p0 = 0.0_real64
    p0_prime = 0.0_real64
    terms = size(coeffs)
    do i = 1, terms
        p = 2 * i
        c = coeffs(i) / i
        b = biases(i)
        result      = result + (c * (x + b))**p
        p0          = p0 + (c * b)**p
        p0_prime    = p0_prime + p * c * (c * b)**(p - 1)
    end do
    ! Ensure that the polynomial is 0 at x=0 and non-negative
    result = result - p0 - p0_prime * x

end function convex_poly_expansion

function matrix_derivative_of_psi(Ce, Ci, dx, flag) result(derivative_matrix)
    real(real64), intent(in)  :: Ce(3,3), Ci(3,3)
    real(real64), intent(in)  :: dx
    real(real64) :: temp_matrix1(3,3), temp_matrix2(3,3)
    real(real64) :: derivative_matrix(3,3)
    real(real64) :: psi1, psi2
    integer :: flag, i, j

    if (flag == 1) then
        ! Compute the derivative with respect to the first component
        ! Perturb the diagonal elements
        do i = 1,3
            temp_matrix1 = Ce
            temp_matrix2 = Ce
            temp_matrix1(i,i) = temp_matrix1(i,i) + dx
            temp_matrix2(i,i) = temp_matrix2(i,i) - dx
            psi1 = Helmholtz(temp_matrix1, Ci)
            psi2 = Helmholtz(temp_matrix2, Ci)
            derivative_matrix(i,i) = (psi1 - psi2) / (2.0 * dx)
        end do
        ! Perturb the off-diagonal elements symmetrically
        do i = 1,3
            do j = i+1,3
                temp_matrix1 = Ce
                temp_matrix2 = Ce
                temp_matrix1(i,j) = temp_matrix1(i,j) + dx
                temp_matrix1(j,i) = temp_matrix1(j,i) + dx
                temp_matrix2(i,j) = temp_matrix2(i,j) - dx
                temp_matrix2(j,i) = temp_matrix2(j,i) - dx
                psi1 = Helmholtz(temp_matrix1, Ci)
                psi2 = Helmholtz(temp_matrix2, Ci)
                derivative_matrix(i,j) = (psi1 - psi2) / (2.0 * dx)
                derivative_matrix(j,i) = derivative_matrix(i,j)
            end do
        end do
    else if (flag == 2) then
        ! Compute the derivative with respect to the second component
        ! Perturb the diagonal elements
        do i = 1,3
            temp_matrix1 = Ci
            temp_matrix2 = Ci
            temp_matrix1(i,i) = temp_matrix1(i,i) + dx
            temp_matrix2(i,i) = temp_matrix2(i,i) - dx
            psi1 = Helmholtz(Ce, temp_matrix1)
            psi2 = Helmholtz(Ce, temp_matrix2)
            derivative_matrix(i,i) = (psi1 - psi2) / (2.0 * dx)
        end do
        ! Perturb the off-diagonal elements symmetrically
        do i = 1,3
            do j = i+1,3
                temp_matrix1 = Ci
                temp_matrix2 = Ci
                temp_matrix1(i,j) = temp_matrix1(i,j) + dx
                temp_matrix1(j,i) = temp_matrix1(j,i) + dx
                temp_matrix2(i,j) = temp_matrix2(i,j) - dx
                temp_matrix2(j,i) = temp_matrix2(j,i) - dx
                psi1 = Helmholtz(Ce, temp_matrix1)
                psi2 = Helmholtz(Ce, temp_matrix2)
                derivative_matrix(i,j) = (psi1 - psi2) / (2.0 * dx)
                derivative_matrix(j,i) = derivative_matrix(i,j)
            end do
        end do
    end if
end function matrix_derivative_of_psi

function yield_function(gamma) result(phi)
    real(real64), intent(in) :: gamma(3,3)
    real(real64) :: phi ! yield function value
    real(real64) :: I1, I2, I3, J2, J3
    real(real64) :: p1, p2, p3

    ! Compute the invariants of the gamma tensor
    I1 = gamma(1,1) + gamma(2,2) + gamma(3,3)
    I2 = gamma(1,1)*gamma(2,2) + gamma(1,1)*gamma(3,3) + gamma(2,2)*gamma(3,3) &
       - gamma(1,2)**2 - gamma(1,3)**2 - gamma(2,3)**2
    I3 = gamma(1,1)*gamma(2,2)*gamma(3,3) + 2.0*gamma(1,2)*gamma(1,3)*gamma(2,3) &
       - gamma(1,1)*gamma(2,3)**2 - gamma(1,2)**2*gamma(3,3) - gamma(1,3)**2*gamma(2,2)

    I1 = I1 * uniform_yield_weight
    I2 = I2 * (uniform_yield_weight**2)
    I3 = I3 * (uniform_yield_weight**3)
    J2 = I1**2 - 2*I2
    J3 = I1**3 - 3 * I1*I2 + 3 * I3

    ! Compute the yield function based on the invariants
    phi = 0.0_real64
    
    ! Contributions from the first invariant
    p1 = convex_poly_expansion(I1, gNet1_coefficients, gNet1_biases)
    phi = phi + gNet1_weights_g(1) * p1
    phi = phi + (cosh(p1) - 1.0) * gNet1_weights_g(2)
    phi = phi + log(cosh(p1)) * gNet1_weights_g(3)
    phi = phi + (exp(p1) - 1.0) * gNet1_weights_g(4)
    phi = phi + (p1**2) * gNet1_weights_g(5)
    phi = phi + (p1**3) * gNet1_weights_g(6)
    phi = phi + (p1**4) * gNet1_weights_g(7)

    ! Contributions from the second invariant
    p2 = convex_poly_expansion(J2, gNet2_coefficients, gNet2_biases)
    phi = phi + gNet2_weights_g(1) * p2
    phi = phi + (cosh(p2) - 1.0) * gNet2_weights_g(2)
    phi = phi + log(cosh(p2)) * gNet2_weights_g(3)
    phi = phi + (exp(p2) - 1.0) * gNet2_weights_g(4)
    phi = phi + (p2**2) * gNet2_weights_g(5)
    phi = phi + (p2**3) * gNet2_weights_g(6)
    phi = phi + (p2**4) * gNet2_weights_g(7)

    ! Contributions from the third invariant
    p3 = convex_poly_expansion(J3, gNet3_coefficients, gNet3_biases)
    phi = phi + gNet3_weights_g(1) * p3
    phi = phi + (cosh(p3) - 1.0) * gNet3_weights_g(2)
    phi = phi + log(cosh(p3)) * gNet3_weights_g(3)
    phi = phi + (exp(p3) - 1.0) * gNet3_weights_g(4)
    phi = phi + (p3**2) * gNet3_weights_g(5)
    phi = phi + (p3**3) * gNet3_weights_g(6)
    phi = phi + (p3**4) * gNet3_weights_g(7)

    ! Move down
    phi = phi - 1.0_real64
end function yield_function

function matrix_derivative_of_yield_function(gamma, dx) result(derivative_matrix)
    real(real64), intent(in)  :: gamma(3,3)
    real(real64), intent(in)  :: dx
    real(real64) :: temp_matrix1(3,3), temp_matrix2(3,3)
    real(real64) :: derivative_matrix(3,3)
    real(real64) :: phi1, phi2
    real(real64) :: max_d
    integer :: i, j

    ! Compute the derivative with respect to the first component
    ! Perturb the diagonal elements
    do i = 1,3
        temp_matrix1 = gamma
        temp_matrix2 = gamma
        temp_matrix1(i,i) = temp_matrix1(i,i) + dx
        temp_matrix2(i,i) = temp_matrix2(i,i) - dx
        phi1 = yield_function(temp_matrix1)
        phi2 = yield_function(temp_matrix2)
        derivative_matrix(i,i) = (phi1 - phi2) / (2.0 * dx)
    end do

    ! Perturb the off-diagonal elements symmetrically
    do i = 1,3
        do j = i+1,3
            temp_matrix1 = gamma
            temp_matrix2 = gamma
            temp_matrix1(i,j) = temp_matrix1(i,j) + dx
            temp_matrix1(j,i) = temp_matrix1(j,i) + dx
            temp_matrix2(i,j) = temp_matrix2(i,j) - dx
            temp_matrix2(j,i) = temp_matrix2(j,i) - dx
            phi1 = yield_function(temp_matrix1)
            phi2 = yield_function(temp_matrix2)
            derivative_matrix(i,j) = (phi1 - phi2) / (2.0 * dx)
            derivative_matrix(j,i) = derivative_matrix(i,j)
        end do
    end do

    ! Normalizze the derivative matrix by the maximum value
    max_d = maxval(abs(derivative_matrix)) + 1e-4
    derivative_matrix = derivative_matrix / max_d
end function matrix_derivative_of_yield_function

function plastic_flow(D, Ui, lambda) result(Ci_new)
    ! Compute the plastic flow based on the derivative of the yield function
    ! A matrix exponential is needed, look into potential libraries for this
    ! For now, it will be built on eigen decomposition
    real(real64), intent(in)  :: D(3,3)
    real(real64), intent(in)  :: Ui(3,3)
    real(real64), intent(in)  :: lambda
    real(real64) :: Ci_new(3,3)

    Ci_new = matmul(Ui, matmul(expm(lambda * D), Ui))
end function plastic_flow

function expm(A) result(expA)
    ! Compute the matrix exponential using eigen decomposition
    real(real64) :: A(3,3)
    real(real64) :: expA(3,3), D(3,3)
    real(real64) :: w(3), work(128)
    integer :: info, n = 3, lda = 3, lwork = 128
    integer :: i, j, k

    ! Call LAPACK routine to compute eigenvalues and eigenvectors
    call dsyev("V", "U", n, A, lda, w, work, lwork, info)
    if (info /= 0) then
        print *, "Error in expm"
        print *, "Error in dsyev: ", info
        stop
    end if
    ! Construct the matrix exponential from eigenvalues and eigenvectors
    w = exp(w) ! Exponentiate the eigenvalues
    D = 0.0_real64
    do i = 1, n
        D(i,i) = w(i)
    end do
    expA =matmul(A, matmul(D, transpose(A))) ! Compute the matrix exponential
end function expm

end module piCANN_mat_model_module