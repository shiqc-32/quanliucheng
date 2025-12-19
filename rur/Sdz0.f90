SUBROUTINE Sdz0(nL, Have, &
		          H,  eta, lambda, &
					 d_o_H, z0_o_H, ierr, &
                     chk_have, chk_lambda, chk_B, chk_result, &
                     chk_B_geom, chk_B_etaeff, chk_B_al, chk_B_layers, &
                     diagfile)

 IMPLICIT NONE

 ! ==== 输入 ====
 integer, INTENT(in)  :: nL         ! divided building layers
 real,    INTENT(in)  :: Have       ! average building height (m)
 real,    INTENT(in)  :: H(-1:nL)   ! height of each layer (m)
 real,    INTENT(in)  :: eta(nL)    ! plan area index (eta) profile (-)
 real,    INTENT(in)  :: lambda(nL) ! frontal area index (lambda) profile (-)
 character*(*) , INTENT(in) :: diagfile

 ! ==== 输出：原有结果 ====
 real,    INTENT(out) :: d_o_H      ! displacement height over Have
 real,    INTENT(out) :: z0_o_H     ! roughness length over Have

 ! ==== 输出：总体诊断 ====
 integer, INTENT(out) :: ierr            ! 0=OK, 1:Have, 2:lambda_tot, 3:B, 4:result NaN
 integer, INTENT(out) :: chk_have        ! 1: Have>0
 integer, INTENT(out) :: chk_lambda      ! 1: lambda_tot>0
 integer, INTENT(out) :: chk_B           ! 1: B>0
 integer, INTENT(out) :: chk_result      ! 1: d_o_H & z0_o_H 非 NaN

 ! ==== 输出：B 路径细分诊断 ====
 integer, INTENT(out) :: chk_B_geom      ! 1: Hmax>0 且所有 deltaH(1:nL)>0
 integer, INTENT(out) :: chk_B_etaeff    ! 1: eta_eff>0
 integer, INTENT(out) :: chk_B_al        ! 1: 所有 al(k)>0 且 max(al)/min(al) 不极端
 integer, INTENT(out) :: chk_B_layers    ! 1: 每层 B_layer(k) 基本非负

 ! ==== 常数 ====
 real, parameter :: Cd    = 1.2
 real, parameter :: kappa = 0.4
 real, parameter :: a_min = 1.E-6
 real, parameter :: alpha = 0.81 !sqrt(0.55)
 real, parameter :: tinyB = 1.0E-12   ! 仅用于诊断 B_layer 是否略为负数

 ! ==== 局部变量 ====
 real :: Hmax, deltaH(0:nL), al(0:nL)
 real :: eta_base, eta_eff, lambda_tot
 real :: a, AA, B, ctmp
 real :: B_layer(nL)
 real :: min_deltaH, min_al, max_al, min_B_layer
 logical :: al_positive, al_finite
 integer :: k, kt
 integer :: iu, ios

 ! 初始化诊断变量（不影响计算）
 ierr         = 0
 chk_have     = 0
 chk_lambda   = 0
 chk_B        = 0
 chk_result   = 0
 chk_B_geom   = 0
 chk_B_etaeff = 0
 chk_B_al     = 0
 chk_B_layers = 0

 !==========================
 ! 以下为原始算法本体：一行不改
 !==========================

 !Hmax; deltaH: H(k) - H(k-1)
 Hmax = H(nL)
 DO k = 0, nL
    deltaH(k) = H(k) - H(k-1)
 ENDDO

 !eta_base, lambda_tot
 eta_base   = eta(1)
 lambda_tot = sum(lambda(:))

 !Main algorithm
 a = 9.6*lambda_tot
 al(0:nL) = 0.
 eta_eff  = 0.
 DO k = 1, nL
   eta_eff = eta_eff + eta(k) * lambda(k) / lambda_tot
 ENDDO
 DO k = 1, nL
    al(k) = a * (eta(k)/eta_eff)**(-0.05) !eta_eff
 ENDDO

 AA = alpha**2 * (Cd*2.*a) / (4.*kappa**2) * exp(-2.*a)
 B  = 0.
 DO k = 1, nL
    ctmp = 0.
	 DO kt = 0, k-1
	  ctmp = ctmp + 2.*al(kt)*deltaH(kt)/Hmax
	 ENDDO
    B_layer(k) = AA * lambda(k)*Hmax/al(k)/deltaH(k) * exp(ctmp) * &
                 (exp(2.*al(k)*deltaH(k)/Hmax) - 1.)
    B = B + B_layer(k)
 ENDDO !k=1,nL

 d_o_H  = ( 1. + 4.43**(-eta_base) * (eta_base - 1.) ) * Hmax/Have
 z0_o_H = (Hmax/Have - d_o_H) * exp(-B**(-0.5))

 !==========================
 ! 诊断：只“看”结果，不改上面的计算
 !==========================

 ! 四个总体检查
 IF (Have       > 0.0) chk_have   = 1
 IF (lambda_tot > 0.0) chk_lambda = 1
 IF (B          > 0.0) chk_B      = 1
 IF ( (d_o_H == d_o_H) .AND. (z0_o_H == z0_o_H) ) chk_result = 1

 ! B 路径细分：
 ! 1) 几何层 thickness 是否合理：Hmax>0 且所有 deltaH(1:nL)>0
 min_deltaH = 1.0E30
 DO k = 1, nL
    IF (deltaH(k) < min_deltaH) min_deltaH = deltaH(k)
 ENDDO
 IF (Hmax > 0.0 .AND. min_deltaH > 0.0) chk_B_geom = 1

 ! 2) eta_eff 是否为正
 IF (eta_eff > 0.0) chk_B_etaeff = 1

 ! 3) al(k) 是否为正且不过于极端
 al_positive = .TRUE.
 al_finite   = .TRUE.
 min_al = 1.0E30
 max_al = -1.0E30
 DO k = 1, nL
    IF (al(k) <= 0.0) al_positive = .FALSE.
    IF (al(k) /= al(k)) al_finite = .FALSE.   ! NaN 检查
    IF (al(k) < min_al) min_al = al(k)
    IF (al(k) > max_al) max_al = al(k)
 ENDDO
 IF (al_positive .AND. al_finite) THEN
    IF (max_al / min_al < 1.0E6) chk_B_al = 1
 ENDIF

 ! 4) 各层对 B 的贡献是否基本非负
 min_B_layer = 1.0E30
 DO k = 1, nL
    IF (B_layer(k) < min_B_layer) min_B_layer = B_layer(k)
 ENDDO
 IF (min_B_layer >= -tinyB) chk_B_layers = 1

 ! 总体错误码（4 大类）
 IF (Have <= 0.0) THEN
    ierr = 1
 ELSEIF (lambda_tot <= 0.0) THEN
    ierr = 2
 ELSEIF (B <= 0.0) THEN
    ierr = 3
 ELSEIF ( (d_o_H /= d_o_H) .OR. (z0_o_H /= z0_o_H) ) THEN
    ierr = 4
 ELSE
    ierr = 0
 END IF

 !==========================
 ! 额外输出：逐层 al(k) 诊断文件（固定小数 10 位）
 ! 不改变任何原计算与判定逻辑
 !==========================
 iu  = 99
 ios = 0
 open(iu, file=trim(diagfile), status='unknown', action='write', iostat=ios)
 if (ios == 0) then
    write(iu,'(A)') '# Layer-wise local drag coefficient al(k) based on RECONSTRUCTED layers (fixed 10 decimals)'
    write(iu,'(A)') '# k  z_low(m)  z_up(m)  deltaH(m)  eta(k)  lambda(k)  al(k)'
    do k = 1, nL
       ! k 为整数；其余全部用 10 位小数（F 格式）
       write(iu,'(I6,1X,F16.10,1X,F16.10,1X,F16.10,1X,F16.10,1X,F16.10,1X,F16.10)') &
            k, H(k-1), H(k), deltaH(k), eta(k), lambda(k), al(k)
    end do
    close(iu)
 end if

END SUBROUTINE Sdz0
