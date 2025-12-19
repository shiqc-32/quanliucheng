SUBROUTINE Sfind_sort_largest(nk, arr, nL, &
		                        indices)

 IMPLICIT NONE

 real, parameter ::  Infinitsml = -1.E6
 integer, INTENT(in)  :: nk
 real,    INTENT(in)  :: arr(nk)
 integer, INTENT(in)  :: nL
 integer, INTENT(out) :: indices(nL)

!Local
 real :: temp
 integer :: i,j,k

  indices(1:nL) = 0
  DO i = 1, nL
    temp = Infinitsml
    DO k = 1, nk
      IF(arr(k) > temp .and. all(indices /= k)) THEN
        temp = arr(k)
        indices(i) = k
      ENDIF
    ENDDO
  ENDDO

END SUBROUTINE Sfind_sort_largest
