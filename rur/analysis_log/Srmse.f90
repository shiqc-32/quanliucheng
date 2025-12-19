
 subroutine Srmse(nx, a_ref, a, &
                  armse)
 implicit none
 integer, intent(in) :: nx
 real, intent(in)    :: a_ref(nx)
 real, intent(in)    :: a(nx)
 real, intent(out)   :: armse

!local
 real :: mean, var_tmp
 integer :: i,k,t, k1, k2, ki
 integer :: loca(1), kl, kmin

 var_tmp = 0.
 do i = 1, nx
   var_tmp = var_tmp + (a(i) - a_ref(i))**2/real(nx)
 enddo !i

 armse = sqrt(var_tmp)


 end subroutine Srmse
