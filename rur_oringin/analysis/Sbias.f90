
 subroutine Sbias(nx, aref, a, &
                  abias)
 implicit none
 integer, intent(in) :: nx
 real, intent(in)    :: aref(nx)
 real, intent(in)    :: a(nx)
 real, intent(out)   :: abias

!local
 real :: amean, var_tmp, amean_ref
 integer :: i,k,t, k1, k2, ki
 integer :: loca(1), kl, kmin


 amean_ref = 0.
 do i = 1, nx
   amean_ref = amean_ref + aref(i)/real(nx)
 enddo !i

 amean = 0.
 do i = 1, nx
   amean = amean + a(i)/real(nx)
 enddo !i

 abias = amean - amean_ref


 end subroutine Sbias
