SUBROUTINE Srp_debug(nk, rp)

 IMPLICIT NONE

 integer, INTENT(in)  :: nk         ! total original building layers
 real,    INTENT(inout)  :: rp(nk)  ! plan area index (rp) (-)

!Local
 real :: atmp
 integer :: i,j,k

  DO k = nk-1, 1, -1
   atmp = rp(k)
   IF (atmp.ge.1.) rp(k) = rp(k+1)
  ENDDO


 END SUBROUTINE Srp_debug
