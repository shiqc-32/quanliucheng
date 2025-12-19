SUBROUTINE Sdz0(nL, Have, &
		          H,  eta, lambda, &
					 d_o_H, z0_o_H)

 IMPLICIT NONE

 integer, INTENT(in)  :: nL         ! divided building layers
 real,    INTENT(in)  :: Have       ! average building height (m)
 real,    INTENT(in)  :: H(-1:nL)   ! height of each layer (m)
 real,    INTENT(in)  :: eta(nL)    ! plan area index (eta) profile (-)
 real,    INTENT(in)  :: lambda(nL) ! frontal area index (lambda) profile (-)

 real,    INTENT(out) :: d_o_H      ! displacement height over Have
 real,    INTENT(out) :: z0_o_H     ! roughness length over Have

 real, parameter :: Cd = 1.2, &
	                 kappa = 0.4
 real, parameter :: a_min = 1.E-6
 real, parameter :: alpha = 0.81 !sqrt(0.55)

!Local
 real :: Hmax, deltaH(0:nL), al(0:nL)
 real :: eta_base, eta_eff, lambda_tot
 real :: a, AA, B, ctmp, dH, z
 integer :: i,j,k,kt


 !
 !Hmax; deltaH: H(k) - H(k-1)
 !
  Hmax = H(nL)
  DO k = 0, nL
    deltaH(k) = H(k) - H(k-1)
  ENDDO

 !
 !eta_base, lambda_tot
 !
  eta_base   = eta(1)
  lambda_tot = sum(lambda(:))


 !+++++++++++++++++++++++++
 !The main algorithm
 !
  a = 9.6*lambda_tot
  al(0:nL) = 0.
  eta_eff  = 0.
  DO k = 1, nL
   eta_eff = eta_eff + eta(k) * lambda(k) / lambda_tot
  ENDDO
  DO k = 1, nL
    al(k) = a * (eta(k)/eta_eff)**(-0.05) !eta_eff
  ENDDO

  AA = alpha**2 * (Cd*2.*a) / (4.*kappa**2) * exp(-2.*a)
  B  = 0.
  DO k = 1, nL
    ctmp = 0.
	 DO kt = 0, k-1
	  ctmp = ctmp + 2.*al(kt)*deltaH(kt)/Hmax
	 ENDDO
    B = B + AA * lambda(k)*Hmax/al(k)/deltaH(k) * exp(ctmp) * (exp(2.*al(k)*deltaH(k)/Hmax) - 1.)
  ENDDO !k=1,nL

  d_o_H = ( 1. + 4.43**(-eta_base) * (eta_base - 1.) ) * Hmax/Have
  z0_o_H = (Hmax/Have - d_o_H) * exp(-B**(-0.5))
 !
 !End of the main algorithm
 !+++++++++++++++++++++++++


 END SUBROUTINE Sdz0
