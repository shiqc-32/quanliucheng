 IMPLICIT NONE

 integer, parameter :: nsites = 130   ! number of total urban sites 
 integer, parameter :: nk     = 300   ! total layers of the original urban data
 real, parameter    :: dz = 2.        ! original urban layer depth (m)
 integer, parameter :: nL_defaut = 8  ! the default number of reconstructed building layers
 real, parameter    :: drp_threshold = 0.01 ! the rp threshold to remove ineffective building heights

!Read in
 real :: rp(nk)  ! local plan area index 
 real :: rf_tot  ! total frontal area index

 real :: rf_density(nk)  ! local frontal area index density (frontal area index/dz) (m-1)

!Output
 real, allocatable :: H(:), eta(:), lambda(:)
 real :: eta2(nk), lambda2(nk), C
 real :: d_o_H, z0_o_H
 real :: Have, Hmax

!Local
 real :: eta0(nk)
 real :: atmp, ztmp, eta_tmp, lamda_tmp
 real :: rf_ref, rp_ref, Have_ref, Hmax_ref, Hstd_ref, dz0_ref(12) !reference for comparison
 integer :: nL
 integer :: i,j,k,id, idtmp, ifg
 character*3   :: idcha
 character*20  :: varlist(19)
 character*20  :: directory
 character*100 :: filename1, filename2


!Open files for read in
 open(10, file='./data_in/rp_profiles.csv', status='old',action='read')
 open(20, file='./data_in/r_f_total.csv', status='old',action='read')
 open(30, file='./data_in/Table1-Kanda_R2.csv', status='old',action='read') !reference for comparison
 read(10,*)
 read(20,*)
 read(30,*)varlist

!Open files for write out
 open(31, file='./data_out/dz0_rur-vs-ref.csv', status='unknown',action='write')
 write(31,'(*(G0, :, ",", 2X))') (trim(varlist(i)),i=1,19), 'Have_rur', 'Hmax_rur', &
	                              'd_rur(m)', 'z0_rur(m)', 'd_rur_o_Have', 'z0_rur_o_Have'

!======================
! Loop for urban sites
!=====================
 DO id = 1, nsites
!
  print *, '----------------------------------'
  print *, 'Reading the site ID:', id
  read(10,*)atmp, rp(:)
  read(20,*)rf_tot
  read(30,*)idtmp, ifg, rf_ref, rp_ref, Have_ref, Hmax_ref, Hstd_ref, dz0_ref(:)

 !debug rp values larger than 1
  CALL Srp_debug(nk, rp)

 !get rf_density profile
  CALL Srf_profile(nk, dz, rp, rf_tot, &
                   rf_density)

 !
 !prepare for the main computing algorithm:
 ! 1.reconstruct building layers and set relevant parameters
 ! 2.set effective maximum building height
 ! out: average building height (Have), 
 !      reconstructed building layers (nL), upper height of each layer H(-1:nL),
 !      plan area index of each layer (eta(nL)), 
 !      frontal area index of each layer (lambda(nL))
 !
  CALL Sreconstructed_layers(nk, rp, nL_defaut, &
	                          drp_threshold, &
	                          nL)
  allocate(H(-1:nL), eta(nL), lambda(nL))
  CALL Spre_dz0(nk, dz, rp, rf_density, nL, &
	             drp_threshold, &
	             Have, H, eta, lambda, eta2, lambda2, zr)

 !
 !Main computing algorithm
 !
  CALL Sdz0(nL, Have, &
            H,  eta, lambda, &
				d_o_H, z0_o_H)

  Hmax = H(nL)
  print *, 'Have=', Have
  print *, 'Hmax=', Hmax
  print *, 'd_o_H =', d_o_H
  print *, 'z0_o_H =', z0_o_H
  print *, ' '

  deallocate(H, eta, lambda)

 !
 !Write
 !
  write(31, '(*(G0.6, :, ",", 2X))')idtmp, ifg, rf_ref, rp_ref, Have_ref, Hmax_ref, Hstd_ref, dz0_ref(:), &
	                                 Have, Hmax, d_o_H*Have, z0_o_H*Have, d_o_H, z0_o_H

  IF(id.ge.1 .and. id.lt.10)THEN
	  write(idcha,'(i1)') id
  elseif(id.ge.10 .and. id.lt.100)THEN
	  write(idcha,'(i2)') id
  elseif(id.ge.100 .and. id.lt.1000)THEN
	  write(idcha,'(i3)') id
  ELSE
	  print *, 'id ', id, 'is out of the upper limit ', nsites
	  STOP
  ENDIF

  directory = './data_out/'
  filename1 = trim(directory)//'ID'//trim(idcha)//'_rpf_profiles_ori.dat'
  filename2 = trim(directory)//'ID'//trim(idcha)//'_rpf_profiles_reconstructed.dat'
  open(11, file=trim(filename1), status='unknown',action='write')
  open(21, file=trim(filename2), status='unknown',action='write')

  write(11,'(3a12)') 'Height', 'rp', 'rf'
  write(21,'(3a12)') 'Height', 'rp', 'rf'

  DO k = 1, nk
   ztmp = real(k) * dz
	write(11,'(f12.2,2f12.6)') ztmp, rp(k), rf_density(k)
	write(21,'(f12.4,2f12.6)') zr(k), eta2(k), lambda2(k)
  ENDDO

  close(11)
  close(21)


 ENDDO ! id
 close(10)
 close(20)
 close(30)
 close(31)

 !
 !Write
 !

! open(11,file='r_ftotal_test.csv')
! write(11,'(*(G0, :, ",", 2X))') 
! write(11, '(*(G0.6, :, ",", 2X))')
! close(11)


 END
