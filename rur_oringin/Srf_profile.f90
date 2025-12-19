SUBROUTINE Srf_profile(nk, dz, rp, rf_tot, &
		                rf_density)

 IMPLICIT NONE

 integer, INTENT(in)  :: nk         ! total original building layers
 real,    INTENT(in)  :: dz         ! original urban layer depth (m)
 real,    INTENT(in)  :: rp(nk)     ! local plan area index
 real,    INTENT(in)  :: rf_tot     ! total frontal area index
 real,    INTENT(out) :: rf_density(nk) ! local frontal area index density (frontal area index/dz) (m-1)

!Local
 real :: atmp, rp_sum
 integer :: i,j,k

  
  rp_sum = sum(rp(:))
  rf_density(:) = 0.

  DO k = 1, nk
   rf_density(k) = rf_tot * rp(k)/rp_sum / dz
  ENDDO


 END SUBROUTINE Srf_profile
