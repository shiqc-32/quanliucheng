SUBROUTINE Sreconstructed_layers(nk, rp, nL_defaut, &
		                           drp_threshold, &
		                           nL) 

 IMPLICIT NONE

 integer, INTENT(in)  :: nk             ! total original building layers
 real,    INTENT(in)  :: rp(nk)         ! local plan area index (f)
 integer, INTENT(in)  :: nL_defaut     ! reconstructed building layers by default
 real,    INTENT(in)  :: drp_threshold
 integer, INTENT(out) :: nL             ! reconstructed building layers

!Local
 real, allocatable :: delta_rp(:)
 real :: atmp, rp2(nk)
 integer :: nL_max, kl_top
 integer :: i,j,k

  
  rp2(:) = rp
  DO k = 1, nk
   IF(rp2(k) .lt. drp_threshold) THEN
		rp2(k) = 0.
	ENDIF
  ENDDO

  DO k = nk, 1, -1
   IF(rp2(k) .ge. drp_threshold) THEN
      kl_top = k
      EXIT
   ENDIF
  ENDDO

  allocate(delta_rp(kl_top))
  delta_rp(:) = 0.
  DO k = 1, kl_top
   delta_rp(k) = rp2(k) - rp2(k+1)
  ENDDO

  nL_max = 0
  DO k = 1, kl_top
   IF(delta_rp(k) .ge. drp_threshold) THEN
		nL_max = nL_max + 1
	ENDIF
  ENDDO

  print *, 'The maximum building layers can be effectively reconstructed is:', nL_max

  IF(nL_defaut .le. nL_max) THEN
	  nL = nL_defaut
	  print *, 'The reconstructed building layers is by default:', nL
  ELSE
	  print *, 'nL_defaut:', nL_defaut, '> nL_max:', nL_max
	  print *, 'Suggest setting the layers no more than', nL_max
!	  print *, 'Input the layer number for nL:'
!	  read(6,*) nL
	  nL = nL_max
	  print *, 'The reconstructed building layers is equal to nL_max = ', nL
  ENDIF

  IF(nL.gt.kl_top) THEN
	  print *, 'nL MUST be no more than', kl_top
     STOP
  ENDIF


 END SUBROUTINE Sreconstructed_layers
