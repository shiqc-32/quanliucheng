 IMPLICIT NONE

 integer, parameter :: nsites = 130   ! number of total urban sites 
 integer, parameter :: nsites_r = 107   ! number of total real urban sites 

!Output
 real :: d_obs_log(nsites), z0_obs_log(nsites)
 real :: d_mac_log(nsites), z0_mac_log(nsites)
 real :: d_kanda_log(nsites), z0_kanda_log(nsites)
 real :: d_rur_log(nsites), z0_rur_log(nsites)
!
 real :: bias_d(3), bias_z0(3), rmse_d(3), rmse_z0(3)

!Local
 real, allocatable :: aref(:), a(:)
 real :: abias, armse
 real :: atmp, ztmp, eta_tmp, lamda_tmp
 real :: rf_ref, rp_ref, Have_ref, Hmax_ref, Hstd_ref, dz0_ref(12) !reference for comparison
 real :: Have, Hmax
 real :: delta_d, delta_z0
 real :: d_obs, z0_obs, d_mean, z0_mean, d_rur, z0_rur, d_rur2, z0_rur2
 integer :: nL, nx
 integer :: is, i,j,k,id, idtmp, ifg
 character*3   :: idcha
 character*10  :: schemecha(3)
 character*20  :: varlist(19)
 character*20  :: directory
 character*100 :: filename1, filename2
 data schemecha /'Mac', 'Kanda', 'RUR'/


!Open files for read in
 open(10, file='../data_out/dz0_rur-vs-ref.csv', status='unknown',action='read')
 read(10,*)

!Open file for write
 open(20, file='./dz0_rur-vs-ref_log.dat')
 write(20,'(2a5, 8a10)')'ID', 'FG', 'd_les', 'z0_les', 'd_mac', 'z0_mac', 'd_kanda', 'z0_kanda', 'd_rur', 'z0_rur'

!======================
! Loop for urban sites
!=====================
 DO id = 1, nsites
!
  print *, '----------------------------------'
  print *, 'Reading the site ID:', id

 !
 !Read
 !
  read(10, *)idtmp, ifg, rf_ref, rp_ref, Have_ref, Hmax_ref, Hstd_ref, dz0_ref(:), &
	                                 Have, Hmax, d_rur, z0_rur, d_rur2, z0_rur2

  d_obs_log(id)  = log(dz0_ref(1))
  z0_obs_log(id) = log(dz0_ref(2))

  d_mac_log(id)  = log(dz0_ref(3))
  z0_mac_log(id) = log(dz0_ref(4))

  d_kanda_log(id)  = log(dz0_ref(5))
  z0_kanda_log(id) = log(dz0_ref(6))

  d_rur_log(id)  = log(d_rur)
  z0_rur_log(id) = log(z0_rur)

  write(20,'(2i5, 8f10.3)') idtmp, ifg, d_obs_log(id), z0_obs_log(id), d_mac_log(id), z0_mac_log(id), &
													  d_kanda_log(id), z0_kanda_log(id), d_rur_log(id), z0_rur_log(id)

 ENDDO ! id
 close(10)
 close(20)


!1. RMSE, Bias for real urban sites
 nx = nsites_r
 allocate(aref(nx), a(nx))

 !1.1 for d
 aref(:) = d_obs_log(1:nx)

 a(:) = d_mac_log(1:nx)
 CALL Sbias(nx, aref, a, abias)
 CALL Srmse(nx, aref, a, armse)
 bias_d(1) = abias
 rmse_d(1) = armse

 a(:) = d_kanda_log(1:nx)
 CALL Sbias(nx, aref, a, abias)
 CALL Srmse(nx, aref, a, armse)
 bias_d(2) = abias
 rmse_d(2) = armse

 a(:) = d_rur_log(1:nx)
 CALL Sbias(nx, aref, a, abias)
 CALL Srmse(nx, aref, a, armse)
 bias_d(3) = abias
 rmse_d(3) = armse

 !1.2 for z0
 aref(:) = z0_obs_log(1:nx)

 a(:) = z0_mac_log(1:nx)
 CALL Sbias(nx, aref, a, abias)
 CALL Srmse(nx, aref, a, armse)
 bias_z0(1) = abias
 rmse_z0(1) = armse

 a(:) = z0_kanda_log(1:nx)
 CALL Sbias(nx, aref, a, abias)
 CALL Srmse(nx, aref, a, armse)
 bias_z0(2) = abias
 rmse_z0(2) = armse

 a(:) = z0_rur_log(1:nx)
 CALL Sbias(nx, aref, a, abias)
 CALL Srmse(nx, aref, a, armse)
 bias_z0(3) = abias
 rmse_z0(3) = armse

 deallocate(aref, a)

!Open file for write
 open(21, file='./bias_rmse_rur-kanda-mac_real.dat')
 write(21,'(5a10)')'scheme','bias_d', 'bias_z0', 'rmse_d', 'rmse_z0' 
 DO is = 1, 3
   write(21,'(a10,4f10.3)') trim(schemecha(is)), bias_d(is), bias_z0(is), rmse_d(is), rmse_z0(is)
 ENDDO
 close(21)


!2. RMSE, Bias for all urban sites
 nx = nsites
 allocate(aref(nx), a(nx))

 !2.1 for d
 aref(:) = d_obs_log(1:nx)

 a(:) = d_mac_log(1:nx)
 CALL Sbias(nx, aref, a, abias)
 CALL Srmse(nx, aref, a, armse)
 bias_d(1) = abias
 rmse_d(1) = armse

 a(:) = d_kanda_log(1:nx)
 CALL Sbias(nx, aref, a, abias)
 CALL Srmse(nx, aref, a, armse)
 bias_d(2) = abias
 rmse_d(2) = armse

 a(:) = d_rur_log(1:nx)
 CALL Sbias(nx, aref, a, abias)
 CALL Srmse(nx, aref, a, armse)
 bias_d(3) = abias
 rmse_d(3) = armse

 !2.2 for z0
 aref(:) = z0_obs_log(1:nx)

 a(:) = z0_mac_log(1:nx)
 CALL Sbias(nx, aref, a, abias)
 CALL Srmse(nx, aref, a, armse)
 bias_z0(1) = abias
 rmse_z0(1) = armse

 a(:) = z0_kanda_log(1:nx)
 CALL Sbias(nx, aref, a, abias)
 CALL Srmse(nx, aref, a, armse)
 bias_z0(2) = abias
 rmse_z0(2) = armse

 a(:) = z0_rur_log(1:nx)
 CALL Sbias(nx, aref, a, abias)
 CALL Srmse(nx, aref, a, armse)
 bias_z0(3) = abias
 rmse_z0(3) = armse

 deallocate(aref, a)

!Open file for write
 open(21, file='./bias_rmse_rur-kanda-mac.dat')
 write(21,'(5a10)')'scheme','bias_d', 'bias_z0', 'rmse_d', 'rmse_z0' 
 DO is = 1, 3
   write(21,'(a10,4f10.3)') trim(schemecha(is)), bias_d(is), bias_z0(is), rmse_d(is), rmse_z0(is)
 ENDDO
 close(21)


 END
