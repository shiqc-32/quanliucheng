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

  CALL sort_indices(nL, indices)

 END SUBROUTINE Sfind_sort_largest

subroutine sort_indices(k, indices)
    implicit none
    integer, intent(in) :: k
    integer, intent(inout) :: indices(k)
    integer :: i, j, temp

    ! 使用简单的冒泡排序对索引进行排序
    do i = 1, k-1
        do j = 1, k-i
            if (indices(j) > indices(j+1)) then
                ! 交换索引
                temp = indices(j)
                indices(j) = indices(j+1)
                indices(j+1) = temp
            end if
        end do
    end do
end subroutine sort_indices

