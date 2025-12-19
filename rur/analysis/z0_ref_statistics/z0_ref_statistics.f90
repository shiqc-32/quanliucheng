 IMPLICIT NONE

 integer, parameter :: nsites = 130   ! number of total urban sites 
 integer, parameter :: nsites_r = 107   ! number of total real urban sites 
 real, parameter :: z0min = 0.
 real, parameter :: z0max = 26.
 real, parameter :: nz = 13  !dz0 = 2


!In
 real :: d(nsites_r), z0(nsites_r)

!Output
 real :: num(nz), p, cp
!

!Local
 integer :: iz, ns(nz)
 real :: dz0, a0, a1
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


!Open files for read in
 open(10, file='../../data_out/dz0_rur-vs-ref.csv', status='unknown',action='read')
 read(10,*)

!======================
! Loop for urban sites
!=====================
 DO id = 1, nsites_r
!
  print *, '----------------------------------'
  print *, 'Reading the site ID:', id

 !
 !Read
 !
  read(10, *)idtmp, ifg, rf_ref, rp_ref, Have_ref, Hmax_ref, Hstd_ref, dz0_ref(:), &
	                                 Have, Hmax, d_rur, z0_rur, d_rur2, z0_rur2

  d(id)  = dz0_ref(1)
  z0(id) = dz0_ref(2)

 ENDDO ! id
 close(10)

! counts, probability, cp
      dz0 = (z0max - z0min)/real(nz)
      ns(:) = 0
      DO i = 1, nsites_r
        DO iz = 1, nz
         a0 = z0min + real(iz-1)*dz0
         a1 = a0 + dz0
         IF(z0(i).gt.a0 .and. z0(i).le.a1) THEN
           ns(iz) = ns(iz) + 1
         ENDIF
        ENDDO !iz
      ENDDO !i

! probability compputing and write out
 open(20, file='./z0_ref_statistics.dat')
 write(20,'(2a8, 2a13)')'z0(m)', 'Count', 'Probability', 'CP'
 DO iz = 1, nz
   atmp = z0min+real(iz-1+0.5)*dz0
   p = real(ns(iz))/real(nsites_r)
   cp   = real(sum(ns(1:iz)))/real(nsites_r)
   write(20,'(f8.1, i8, 2f13.4)')atmp, ns(iz), p, cp
 ENDDO
 close(20)


 END
