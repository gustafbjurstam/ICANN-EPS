!test_piCANN.f90
program test_piCANN
    use iso_fortran_env, only: real64
    use piCANN_mat_model_module, only: piCANN_mat_model
    implicit none

    ! Simulation parameters
    integer, parameter :: nElem = 24
    integer, parameter :: nTimeSteps = 60
    real(real64), parameter :: initialC11 = 1.0_real64
    real(real64), parameter :: finalC11 = 0.09_real64
    
    ! Variables for simulation
    real(real64) :: C(6, nElem)         ! Right Cauchy-Green deformation tensor
    real(real64) :: history(7, nElem)   ! Internal variables (inelastic strain)
    real(real64) :: sigma(6, nElem)     ! Stress tensor
    real(real64) :: C11_current         ! Current value of C11 for first element
    real(real64) :: deltaC11            ! Change in C11 per time step
    integer :: i, t                     ! Loop indices
    
    ! Output file
    integer :: fileUnit
    
    ! Initialize C tensor to identity for both elements
    C = 0.0_real64
    C(1,:) = 1.0_real64  ! C11
    C(2,:) = 1.0_real64  ! C22
    C(3,:) = 1.0_real64  ! C33
    
    ! Initialize history (inelastic strain) to identity tensor
    history = 0.0_real64
    history(1:3,:) = 1.0_real64  ! Diagonal components to 1.0
    
    ! Calculate the change in C11 per time step
    deltaC11 = (finalC11 - initialC11) / real(nTimeSteps, real64)
    
    ! Open output file
    open(newunit=fileUnit, file="piCANN_results.txt", status="replace")
    
    ! Write header to file
    write(fileUnit, '(A)') "TimeStep,Element,C11,C22,C33,C12,C13,C23,S11,S22,S33,S12,S13,S23,Ep11,Ep22,Ep33,Ep12,Ep13,Ep23"
    
    ! Time stepping loop
    do t = 0, nTimeSteps
        ! Update C11 for the first element only
        C11_current = initialC11 + deltaC11 * t
        C(1, 1) = C11_current
        
        ! Call the material model
        call piCANN_mat_model(C, history, sigma, nElem)
        
        ! Write results to file
        do i = 1, 1
            write(fileUnit, '(I4,",",I1,",",18(ES14.6,","))') &
                t, i, &
                C(1,i), C(2,i), C(3,i), C(4,i), C(5,i), C(6,i), &
                sigma(1,i), sigma(2,i), sigma(3,i), sigma(4,i), sigma(5,i), sigma(6,i), &
                history(1,i), history(2,i), history(3,i), history(4,i), history(5,i), history(6,i)
        end do
    end do
    
    ! Close the output file
    close(fileUnit)
    
    print *, "Simulation completed. Results written to piCANN_results.txt"
    
end program test_piCANN