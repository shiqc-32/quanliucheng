SUBROUTINE Spre_dz0(nk, dz, rp, rf_density, nL, &
		              drp_threshold, &
		              Have, H, eta, lambda, eta2, lambda2, zr)

 IMPLICIT NONE

 integer, INTENT(in)  :: nk             ! total original building layers
 real,    INTENT(in)  :: dz             ! original urban layer depth (m)
 real,    INTENT(in)  :: rp(nk)         ! local plan area index (f)
 real,    INTENT(in)  :: rf_density(nk) ! local frontal area index density (frontal area index/dz) (m-1)
 integer, INTENT(in)  :: nL             ! reconstructed building layers
 real,    INTENT(in)  :: drp_threshold
 real,    INTENT(out) :: Have        ! average building height (m)
 real,    INTENT(out) :: H(-1:nL)    ! upper height of reconstructed layers (m)
 real,    INTENT(out) :: eta(nL)     ! local plan area idnex of reconstructed layers (-)
 real,    INTENT(out) :: lambda(nL)  ! local frontal area index (lambda) profile (-)
 real,    INTENT(out) :: eta2(nk)    ! eta(1:nL) -> eta2(1:nk) for analysis
 real,    INTENT(out) :: lambda2(nk) ! lambda(1:nL)/dz -> lambda2(1:nk) for analysis
 real,    INTENT(out) :: zr(nk)      ! heights of the reconstructed layers for analysis

!Local
 real :: atmp, rp2(nk), rf2(nk)
 real, allocatable :: delta_rp(:)
 real :: rp_tmp(nk), rf_tmp(nk), dH, dzr
 integer :: i,j,k,ik,kr, indices(nL-1), ik0, ik1, kl_low(nL), kl_up(nL), kl_top

!the building height is removed when the local plan area index < 0.02, following Cheng and Yang (2023,BLM)
  rp2(:) = rp(:)
  rf2(:) = rf_density(:)
  DO k = 1, nk
    IF(rp2(k) .lt. drp_threshold) THEN
      rp2(k) = 0.
      rf2(k) = 0.
    ENDIF
  ENDDO

  DO k = nk, 1, -1
    IF(rp2(k) .ge. drp_threshold) THEN
      kl_top = k
      EXIT
    ENDIF
  ENDDO

  allocate(delta_rp(kl_top-1))
  delta_rp(:) = 0.
  DO k = 1, kl_top-1
    delta_rp(k) = rp2(k) - rp2(k+1)
  ENDDO

!find and sort the nL largest indices of delta_rp
  CALL Sfind_sort_largest(kl_top-1, delta_rp, nL-1, &
	                       indices)
  deallocate(delta_rp)

!Have
  Have = 0.
  DO k = 1, nk-1
    Have = Have + real(k)*dz * (rp2(k) - rp2(k+1))/rp2(1)
  ENDDO

!!!!! Key for setting:
! H(-1:nL), eta(nL), lambda(nL), and
! eta2(nk), lambda2(nk)
  DO k = 1, nL-1
    kl_up(k) = indices(k)
  ENDDO
  kl_up(nL) = kl_top
  !
  kl_low(1) = 1
  DO k = 2, nL
    kl_low(k) = kl_up(k-1) + 1
  ENDDO

  H(-1:nL)  = 0.
  eta(1:nL) = 0.
  eta2(:)   = 0.
  lambda(1:nL) = 0.
  lambda2(:)   = 0.

  DO k = 1, nL
    ik0 = kl_low(k)
    ik1 = kl_up(k)
    eta(k) = rp2(ik0)
    H(k)   = real(ik1)*dz  ! test for version 2
    DO ik = ik0, ik1
      lambda(k) = lambda(k) + rf2(ik)*dz !(-)
    ENDDO !ik
  ENDDO !k=1,nL

  !eta2, lambda2 for analysis
  dzr = H(nL)/real(nk)
  DO kr = 1, nk
    zr(kr) = real(kr)*dzr
    DO k = 1, nL
      IF (zr(kr).le.H(k) .and. zr(kr).gt.H(k-1)) THEN
        dH = H(k) - H(k-1)
        eta2(kr) = eta(k)
        lambda2(kr) = lambda(k)/dH  !(m-1)
      ENDIF
    ENDDO !k=1,nL
  ENDDO !kr=1,nk
!!!!!
 END SUBROUTINE Spre_dz0
