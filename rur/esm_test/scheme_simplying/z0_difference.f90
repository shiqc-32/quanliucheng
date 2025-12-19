 IMPLICIT NONE

 integer, parameter :: nsites = 107 !130   ! number of total urban sites 
 real, parameter :: kappa = 0.4
 real, parameter :: zref  = 30.  ! reference height (m), zatm = zref + d + z0 (according to CLM5)

!Read in
 real :: dz0_ref(12), dz0_rur(4)

!Output
 real :: ust(4), bias(4), rmse(4), ust2(4), bias2(4)
 real :: ust_les, ust_mac, ust_kanda, ust_rur
 real :: bias_mac, bias_kanda, bias_rur
 real :: rmse_mac, rmse_kanda, rmse_rur

!Local
 real :: z0_les, z0_mac, z0_kanda, z0_rur
 real :: ust_o_U
 real :: ust_les_site, ust_mac_site, ust_kanda_site, ust_rur_site ! u*/Uatm
 
 real :: rf_ref, rp_ref, Have_ref, Hmax_ref, Hstd_ref, Have, Hmax
 real :: atmp
 integer :: i,j,k,id, idtmp, ifg
 character*9   :: idscheme(4)
 character*20  :: varlist(19)
 character*20  :: directory
 character*100 :: filename1, filename2

 data idscheme /'les', 'mac', 'kanda', 'rur'/


!Open files for write out
 open(10, file='./z0_difference.dat', status='unknown',action='write')
 write(10,'(7a10,a5)') "rf", "rp", "rf*rp", "z0_les", "z0_rur", "bias", "bias_2", "id"

!Open files for read in
 open(31, file='../../data_out/dz0_rur-vs-ref.csv', status='old',action='read')
 read(31,*)

!======================
! Loop for urban sites
!=====================
 DO id = 1, nsites
!
  print *, '----------------------------------'
  print *, 'Reading the site ID:', id
 
  read(31, *)idtmp, ifg, rf_ref, rp_ref, Have_ref, Hmax_ref, Hstd_ref, dz0_ref(:), &
	                                 Have, Hmax, dz0_rur(:) !d_o_H*Have, z0_o_H*Have, d_o_H, z0_o_H

  z0_les   = dz0_ref(2)
  z0_mac   = dz0_ref(4)
  z0_kanda = dz0_ref(6)
  z0_rur   = dz0_rur(2)

  write(10,'(3f10.3, 3f10.2, f10.3, i5)') rf_ref, rp_ref, rf_ref*rp_ref, z0_les, z0_rur, z0_rur-z0_les, (z0_rur-z0_les)/z0_les, id

 ENDDO ! id

 close(10)
 close(31)



 END
