 IMPLICIT NONE

 integer, parameter :: nsites = 107 !130   ! number of total urban sites 
 real, parameter :: kappa = 0.4
 real, parameter :: zref  = 30.  ! reference height (m), zatm = zref + d + z0 (according to CLM5)
 real, parameter :: z0_threshold = 7.

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
 integer :: i,j,k,id, idtmp, ifg, next
 character*9   :: idscheme(4)
 character*20  :: varlist(19)
 character*20  :: directory
 character*100 :: filename1, filename2

 data idscheme /'les', 'mac', 'kanda', 'rur'/


!Open files for write out
 open(10, file='./u10m_sites.dat', status='unknown',action='write')
 open(11, file='./u10m_mean.dat', status='unknown',action='write')
 write(10,'(5a10)') "site_id", "les", "Mac", "Kanda", "RUR"
 write(11,'(a5,4a10)') "id", "Scheme", "mean", "bias", "rmse"

!Open files for read in
 open(31, file='../../data_out/dz0_rur-vs-ref.csv', status='old',action='read')
 read(31,*)

 ust_les = 0.
 ust_mac = 0.
 ust_kanda = 0.
 ust_rur   = 0.
 rmse_mac   = 0. 
 rmse_kanda = 0.
 rmse_rur   = 0.

 next = 0
 DO id = 1, nsites
!
  print *, '----------------------------------'
  print *, 'Reading the site ID:', id
 
  read(31, *)idtmp, ifg, rf_ref, rp_ref, Have_ref, Hmax_ref, Hstd_ref, dz0_ref(:), &
	                                 Have, Hmax, dz0_rur(:) !d_o_H*Have, z0_o_H*Have, d_o_H, z0_o_H

  z0_les   = dz0_ref(2)
  IF(z0_les .ge. z0_threshold)THEN
	  print *, id
	  next = next + 1
  ENDIF
 ENDDO ! id
 close(31)

 print *, 'next=', next

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

  ust_les_site = log(10./z0_les + 1) / log(zref/z0_les + 1)
  ust_mac_site = log(10./z0_les + 1) / log(zref/z0_mac + 1)
  ust_kanda_site = log(10./z0_les + 1) / log(zref/z0_kanda + 1)
  ust_rur_site   = log(10./z0_les + 1) / log(zref/z0_rur + 1)

  write(10,'(i10,4f10.4)') idtmp, ust_les_site, ust_mac_site, ust_kanda_site, ust_rur_site

  IF(z0_les .ge. z0_threshold) THEN
  ust_les = ust_les + ust_les_site/real(next)
  ust_mac = ust_mac + ust_mac_site/real(next)
  ust_kanda = ust_kanda + ust_kanda_site/real(next)
  ust_rur   = ust_rur + ust_rur_site/real(next)

  rmse_mac = rmse_mac + (ust_mac_site - ust_les_site)**2/real(next)
  rmse_kanda = rmse_kanda + (ust_kanda_site - ust_les_site)**2/real(next)
  rmse_rur = rmse_rur + (ust_rur_site - ust_les_site)**2/real(next)
  ENDIF

 ENDDO ! id

 ust(1) = ust_les
 ust(2) = ust_mac
 ust(3) = ust_kanda
 ust(4) = ust_rur
 ust2(:) = ust(:)**2

 bias_mac = (ust_mac - ust_les) / ust_les
 bias_kanda = (ust_kanda - ust_les) / ust_les
 bias_rur = (ust_rur - ust_les) / ust_les
 bias(:) = 0.
 bias(2) = bias_mac * 100.
 bias(3) = bias_kanda * 100.
 bias(4) = bias_rur * 100.

 rmse_mac = sqrt(rmse_mac)
 rmse_kanda = sqrt(rmse_kanda)
 rmse_rur   = sqrt(rmse_rur)
 rmse(:) = 0.
 rmse(2) = rmse_mac
 rmse(3) = rmse_kanda
 rmse(4) = rmse_rur

 DO k = 1, 4
  write(11,'(i5, a10,f10.3,f10.1,f10.4)') k, trim(idscheme(k)), ust(k), bias(k), rmse(k)
 ENDDO

 close(10)
 close(11)
 close(31)



 END
