! Auto-generated Fortran module for piCANN weights
! Generated from model: outputs/models/icann_staged_refactored

module piCANN_weights
  use iso_fortran_env, only: real64
  implicit none

  ! Network sizes
  integer, parameter :: n_e = 1
  integer, parameter :: n_i = 1
  integer, parameter :: n_g1 = 1
  integer, parameter :: n_g2 = 1
  integer, parameter :: n_g3 = 1

  ! Configuration parameters
  real(real64), parameter :: tolerance = 9.9999997473787516e-06_real64
  real(real64), parameter :: lambda_dot = 4.9999998882412910e-03_real64

  ! cp1
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_e_iso_net1_iso1_cp1(2) = &
    [2.7998644113540649e-01_real64, 9.8063848912715912e-02_real64]

  ! cp2
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_e_iso_net1_iso1_cp2(2) = &
    [-0.0000000000000000e+00_real64, -0.0000000000000000e+00_real64]

  ! cp3
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_e_iso_net1_iso1_cp3(2) = &
    [-0.0000000000000000e+00_real64, -0.0000000000000000e+00_real64]

  ! cp1
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_e_iso_net2_iso2_cp1(2) = &
    [5.2185541391372681e-01_real64, 2.0153100788593292e-01_real64]

  ! cp2
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_e_iso_net2_iso2_cp2(2) = &
    [0.0000000000000000e+00_real64, -0.0000000000000000e+00_real64]

  ! cp3
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_e_iso_net2_iso2_cp3(2) = &
    [-0.0000000000000000e+00_real64, -0.0000000000000000e+00_real64]

  ! coefficients
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_e_vol_net_vol_coefficients(1) = &
    [5.3063213825225830e-01_real64]

  ! biases
  real(real64), parameter :: hNet_e_vol_net_vol_biases(1) = &
    [0.0000000000000000e+00_real64]

  ! coefficients
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_e_vol_net_power_expansion_coefficients(1) = &
    [5.3063213825225830e-01_real64]

  ! biases
  real(real64), parameter :: hNet_e_vol_net_power_expansion_biases(1) = &
    [0.0000000000000000e+00_real64]

  ! cp1
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_i_iso_net1_iso1_cp1(2) = &
    [-0.0000000000000000e+00_real64, -0.0000000000000000e+00_real64]

  ! cp2
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_i_iso_net1_iso1_cp2(2) = &
    [-0.0000000000000000e+00_real64, -0.0000000000000000e+00_real64]

  ! cp3
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_i_iso_net1_iso1_cp3(2) = &
    [-0.0000000000000000e+00_real64, -0.0000000000000000e+00_real64]

  ! cp1
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_i_iso_net2_iso2_cp1(2) = &
    [-0.0000000000000000e+00_real64, -0.0000000000000000e+00_real64]

  ! cp2
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_i_iso_net2_iso2_cp2(2) = &
    [-0.0000000000000000e+00_real64, -0.0000000000000000e+00_real64]

  ! cp3
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_i_iso_net2_iso2_cp3(2) = &
    [-0.0000000000000000e+00_real64, -0.0000000000000000e+00_real64]

  ! coefficients
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_i_vol_net_vol_coefficients(1) = &
    [4.9196518375538290e-05_real64]

  ! biases
  real(real64), parameter :: hNet_i_vol_net_vol_biases(1) = &
    [0.0000000000000000e+00_real64]

  ! coefficients
  ! This weight has non-negative constraint
  real(real64), parameter :: hNet_i_vol_net_power_expansion_coefficients(1) = &
    [4.9196518375538290e-05_real64]

  ! biases
  real(real64), parameter :: hNet_i_vol_net_power_expansion_biases(1) = &
    [0.0000000000000000e+00_real64]

  ! weights_g
  ! This weight has non-negative constraint
  real(real64), parameter :: gNet1_weights_g(7) = &
    [6.9206418991088867e+00_real64, 6.9455614089965820e+00_real64, 9.8508358001708984e-01_real64, &
     9.7625150680541992e+00_real64, 6.5952386856079102e+00_real64, 6.3300919532775879e+00_real64, &
     6.2319507598876953e+00_real64]

  ! coefficients
  ! This weight has non-negative constraint
  real(real64), parameter :: gNet1_coefficients(1) = &
    [3.5039126873016357e-02_real64]

  ! biases
  real(real64), parameter :: gNet1_biases(1) = &
    [0.0000000000000000e+00_real64]

  ! coefficients
  ! This weight has non-negative constraint
  real(real64), parameter :: gNet1_poly_coefficients(1) = &
    [3.5039126873016357e-02_real64]

  ! biases
  real(real64), parameter :: gNet1_poly_biases(1) = &
    [0.0000000000000000e+00_real64]

  ! weights_g
  ! This weight has non-negative constraint
  real(real64), parameter :: gNet2_weights_g(7) = &
    [8.5657405853271484e+00_real64, 2.1528244018554688e+00_real64, 8.8735790252685547e+00_real64, &
     7.6669511795043945e+00_real64, 5.7439937591552734e+00_real64, 3.2413935661315918e+00_real64, &
     8.4830532073974609e+00_real64]

  ! coefficients
  ! This weight has non-negative constraint
  real(real64), parameter :: gNet2_coefficients(1) = &
    [8.0402606725692749e-01_real64]

  ! biases
  real(real64), parameter :: gNet2_biases(1) = &
    [0.0000000000000000e+00_real64]

  ! coefficients
  ! This weight has non-negative constraint
  real(real64), parameter :: gNet2_poly_coefficients(1) = &
    [8.0402606725692749e-01_real64]

  ! biases
  real(real64), parameter :: gNet2_poly_biases(1) = &
    [0.0000000000000000e+00_real64]

  ! weights_g
  ! This weight has non-negative constraint
  real(real64), parameter :: gNet3_weights_g(7) = &
    [3.9727568626403809e+00_real64, 3.6217951774597168e+00_real64, 2.4194717407226562e-02_real64, &
     4.9558601379394531e+00_real64, 9.7568178176879883e+00_real64, 7.7740883827209473e+00_real64, &
     3.5889530181884766e+00_real64]

  ! coefficients
  ! This weight has non-negative constraint
  real(real64), parameter :: gNet3_coefficients(1) = &
    [6.6032785177230835e-01_real64]

  ! biases
  real(real64), parameter :: gNet3_biases(1) = &
    [0.0000000000000000e+00_real64]

  ! coefficients
  ! This weight has non-negative constraint
  real(real64), parameter :: gNet3_poly_coefficients(1) = &
    [6.6032785177230835e-01_real64]

  ! biases
  real(real64), parameter :: gNet3_poly_biases(1) = &
    [0.0000000000000000e+00_real64]

  ! uniform_yield_weight
  real(real64), parameter :: uniform_yield_weight = 2.0649909973144531e+00_real64

end module piCANN_weights
