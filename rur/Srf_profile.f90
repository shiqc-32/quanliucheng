SUBROUTINE Srf_profile(nk, dz, rp, rf_tot, &
		                rf_density)

 IMPLICIT NONE

 integer, INTENT(in)  :: nk
 real,    INTENT(in)  :: dz
 real,    INTENT(in)  :: rp(nk)
 real,    INTENT(in)  :: rf_tot
 real,    INTENT(out) :: rf_density(nk)

!Local
 real :: atmp, rp_sum
 integer :: i,j,k

  rp_sum = sum(rp(:))
  rf_density(:) = 0.

  DO k = 1, nk
    rf_density(k) = rf_tot * rp(k)/rp_sum / dz
  ENDDO

 END SUBROUTINE Srf_profile
