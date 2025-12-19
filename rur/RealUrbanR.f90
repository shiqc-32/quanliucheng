IMPLICIT NONE

 integer, parameter :: nsites = 538   ! number of total urban sites
 integer, parameter :: nk     = 89    ! total layers of the original urban data
 real,    parameter :: dz = 2.        ! original urban layer depth (m)
 integer, parameter :: nL_defaut = 8  ! the default number of reconstructed building layers
 real,    parameter :: drp_threshold = 0.01 ! the rp threshold to remove ineffective building heights

!Read in
 real :: rp(nk)            ! local plan area index
 real :: rf_tot            ! total frontal area index
 real :: rf_density(nk)    ! local frontal area index density (frontal area index/dz) (m-1)

!Output
 real, allocatable :: H(:), eta(:), lambda(:)
 real :: eta2(nk), lambda2(nk), zr(nk)
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
 character*100 :: filename1, filename2, filename3
 character*200 :: remark

 ! Sdz0 总体诊断（4 大原因）
 integer :: ierr_sdz0
 integer :: chk_have, chk_lambda, chk_B, chk_result

 ! B 内部细分诊断
 integer :: chk_B_geom      ! 1: Hmax>0 且 deltaH(1:nL)>0
 integer :: chk_B_etaeff    ! 1: eta_eff>0
 integer :: chk_B_al        ! 1: 所有 al(k)>0 且 max(al)/min(al) 不极端
 integer :: chk_B_layers    ! 1: 每层 B_layer(k) 基本非负

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
	                              'd_rur(m)', 'z0_rur(m)', 'd_rur_o_Have', 'z0_rur_o_Have', &
                                  'chk_Have_gt0', 'chk_lambda_tot_gt0', 'chk_B_gt0', 'chk_result_valid', &
                                  'chk_B_geom_ok', 'chk_B_etaeff_ok', 'chk_B_al_ok', 'chk_B_layers_nonneg', &
                                  'remark'

!======================
! Loop for urban sites
!======================
 DO id = 1, nsites
!
  print *, '----------------------------------'
  print *, 'Reading the site ID:', id

  ! ===== 生成案例编号字符串 idcha，并构造输出文件名（提前）=====
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
  filename3 = trim(directory)//'ID'//trim(idcha)//'_B_al_diag.dat'

  ! 初始化诊断量
  remark       = 'OK'
  ierr_sdz0    = 0
  chk_have     = 0
  chk_lambda   = 0
  chk_B        = 0
  chk_result   = 0
  chk_B_geom   = 0
  chk_B_etaeff = 0
  chk_B_al     = 0
  chk_B_layers = 0

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
 !Main computing algorithm + B 相关诊断 + 逐层 al(k) 输出文件
 !
  CALL Sdz0(nL, Have, &
            H,  eta, lambda, &
            d_o_H, z0_o_H, ierr_sdz0, &
            chk_have, chk_lambda, chk_B, chk_result, &
            chk_B_geom, chk_B_etaeff, chk_B_al, chk_B_layers, &
            filename3)

  Hmax = H(nL)
  print *, 'Have      =', Have
  print *, 'Hmax      =', Hmax
  print *, 'd_o_H     =', d_o_H
  print *, 'z0_o_H    =', z0_o_H
  print *, 'ierr_sdz0 =', ierr_sdz0
  print *, 'chk_have  =', chk_have
  print *, 'chk_lambda=', chk_lambda
  print *, 'chk_B     =', chk_B
  print *, 'chk_result=', chk_result
  print *, 'chk_B_geom   =', chk_B_geom
  print *, 'chk_B_etaeff =', chk_B_etaeff
  print *, 'chk_B_al     =', chk_B_al
  print *, 'chk_B_layers =', chk_B_layers

  ! ===== 生成 remark：重点细分 B<=0 的原因 =====
  if ( (z0_o_H /= z0_o_H) .or. (z0_o_H <= 0.) .or. (ierr_sdz0 /= 0) ) then

     select case (ierr_sdz0)

     case (1)
        remark = 'ERR1: Have <= 0 in Sdz0 (invalid average height)'

     case (2)
        remark = 'ERR2: lambda_tot <= 0 in Sdz0 (sum(lambda)=0)'

     case (3)
        ! B <= 0，进一步看哪个检查不过
        if (chk_B_geom == 0) then
           remark = 'ERR3-geom: B <= 0 (Hmax or deltaH invalid or non-positive)'
        elseif (chk_B_etaeff == 0) then
           remark = 'ERR3-eta: B <= 0 (eta_eff <= 0, invalid effective plan area index)'
        elseif (chk_B_al == 0) then
           remark = 'ERR3-al: B <= 0 (al(k) non-positive or extremely uneven)'
        elseif (chk_B_layers == 0) then
           remark = 'ERR3-layer: B <= 0 (some layer contribution to B is negative)'
        else
           remark = 'ERR3: B <= 0 in Sdz0 (other numerical issue)'
        end if

     case (4)
        remark = 'ERR4: d_o_H or z0_o_H is NaN/Inf in Sdz0'

     case default
        remark = 'ERR?: unknown numerical issue in Sdz0'

     end select

  else
     remark = 'OK'
  end if

  print *, 'remark    =', trim(remark)
  print *, ' '

 !
 !Write summary CSV (保持原逻辑)
 !
  write(31, '(*(G0.6, :, ",", 2X))', advance='no') &
       idtmp, ifg, rf_ref, rp_ref, Have_ref, Hmax_ref, Hstd_ref, dz0_ref(:), &
	   Have, Hmax, d_o_H*Have, z0_o_H*Have, d_o_H, z0_o_H, &
       chk_have, chk_lambda, chk_B, chk_result, &
       chk_B_geom, chk_B_etaeff, chk_B_al, chk_B_layers
  write(31, '(",", 1X, A)') trim(remark)

  ! ===== 原有 rpf 输出文件（保持原逻辑，仅依赖已生成的 idcha/directory）=====
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

  deallocate(H, eta, lambda)

 ENDDO ! id

 close(10)
 close(20)
 close(30)
 close(31)

 END
