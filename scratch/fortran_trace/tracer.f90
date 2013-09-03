! -*- f90 -*-

!---------------------------------------------------------------
SUBROUTINE get_topo(gx, gy, gz, bx, by, bz, topo, x1, x2, y1, y2, z1, z2, &
                    nx, ny, nz, outnx, outny, outnz, nsegs)
!---------------------------------------------------------------
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: nx, ny, nz
  INTEGER, INTENT(IN) :: outnx, outny, outnz
  REAL, INTENT(IN) :: gx(nx), gy(ny), gz(nz)
  REAL, INTENT(IN) :: bx(nx, ny, nz), by(nx, ny, nz), bz(nx, ny, nz)
  INTEGER, INTENT(INOUT) :: topo(outnx, outny, outnz)
  REAL, INTENT(IN) :: x1, x2, y1, y2, z1, z2
  INTEGER, INTENT(INOUT) :: nsegs
  REAL :: x, y, z
  REAL :: vl(3, 100000)
  ! REAL :: vl2(3, 100000)
  INTEGER :: ix, iy, iz
  INTEGER :: nv, ifl, ihem
  INTEGER :: nopen, nclose, nsw
  ! INTEGER :: nsegs

  ! x1 = -2.0
  ! x2 = 8.0
  ! y1 = -12.0
  ! y2 = 0.0
  ! z1 = -5.0
  ! z2 = 5.0

  !write(0,*) 'bound2:  ', x1, x2, y1, y2, z1, z2

  nopen = 0
  nclose = 0
  nsw = 0

  do ix=1, outnx
    do iy=1, outny
      do iz=1, outnz
        x = x1 + float(ix - 1) * (x2 - x1) / float(outnx - 1)
        y = y1 + float(iy - 1) * (y2 - y1) / float(outny - 1)
        z = z1 + float(iz - 1) * (z2 - z1) / float(outnz - 1)

        call xcl4(gx, gy, gz, bx, by, bz, x, y, z, ifl, ihem, 0, 100, 2, vl, nv, nx, ny, nz, nsegs)
        topo(ix, iy, iz) = ifl

        ! if(ifl.eq.0) then
        !   nclose = nclose + 1
        ! endif
        ! if(ifl.eq.1) then
        !   nopen = nopen + 1
        ! endif
        ! if(ifl.eq.2) then
        !   nsw = nsw + 1
        ! endif

        ! if(ix.eq.iy) then
        !   write(0, *) x, y, z, ix, iy, iz, ifl
        ! endif

      end do
    end do
  end do

  WRITE (*, *) "Total segments calculated: ", nsegs 

END SUBROUTINE


SUBROUTINE xcl4(gx, gy, gz, bx, by, bz, x, y, z, &
                ifl, ihem, imo, xmaxlen, ndezi, vl, nv, &
                nx, ny, nz, nsegs)
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: nx, ny, nz
  REAL, INTENT(IN) :: bx(nx,ny,nz), by(nx,ny,nz), bz(nx,ny,nz)
  REAL, INTENT(IN) :: gx(nx), gy(ny), gz(nz)
  REAL, INTENT(IN) :: x, y, z
  INTEGER, INTENT(IN) :: imo, xmaxlen, ndezi
  REAL, INTENT(OUT) :: vl(3, 100000)
  INTEGER, INTENT(OUT) :: ifl, ihem, nv
  INTEGER, INTENT(INOUT) :: nsegs
  ! CHARACTER*256 cdum,rec
  REAL :: xmhd, ymhd, zmhd
  REAL :: xs, ys, zs
  REAL :: xe, ye, ze
  INTEGER :: ifl1, ifl2, ip, maxst, i, j, k, np
  REAL :: dir, ddir, rs
  INTEGER :: iv = 0

  if(iv.eq.1) then
    write(0,*) 'xcl4aa:  ', x, y, z, imo
  endif

  xmhd = x
  ymhd = y
  zmhd = z

  !.... first trace backward
  ddir = 0.020;
  maxst = int(xmaxlen / ddir)
  if(imo.eq.0) then
    maxst=10000
  endif

  ip = 2 * maxst + 2;
  vl(1, ip) = xmhd;
  vl(2, ip) = ymhd;
  vl(3, ip) = zmhd;
  dir = -ddir;
  ifl1 = 0;
  np = 1

  xs = xmhd;
  ys = ymhd;
  zs = zmhd

  ! trace backward
  do k=1, maxst
    if(iv.eq.1) then
      write(0,'(a,2i6,6(1x,f8.3))') 'backward1:  ', ip, k, xe, ye, ze 
    endif

    call trace2(gx, gy, gz, bx, by, bz, xs, ys, zs, xe, ye, ze, dir, nx, ny, nz)
    nsegs = nsegs + 1
    rs = sqrt(xe * xe + ye * ye + ze * ze)

    ! inner boundary
    if(rs.lt.3.5) then
      ifl1 = 1
      exit
    endif

    ! outer boundary
    if(xe.lt.gx(1).or.xe.gt.gx(nx)) then
      ifl1 = 2
      exit
    endif
    if(ye.lt.gy(1).or.ye.gt.gy(ny)) then
      ifl1 = 2
      exit
    endif
    if(ze.lt.gz(1).or.ze.gt.gz(nz)) then
      ifl1 = 2
      exit
    endif

    ip = ip-1;
    vl(1, ip) = xe;
    vl(2, ip) = ye;
    vl(3, ip) = ze;
    xs = xe;
    ys = ye;
    zs = ze;
    np = np + 1
  end do
  ! write(0,*)'xcl4a: ',xmhd,ymhd,zmhd,xe,ye,ze,imo,ifl,rs

  !...... repack array
  j = ip;
  do i=1, np
    vl(1, i) = vl(1, j);
    vl(2, i) = vl(2, j);
    vl(3, i) = vl(3, j)
    if(iv.eq.1) then
      write(0,'(a,2i6,6(1x,f8.3))')'backward: ', j, i, vl(1, j), vl(2, j), vl(3, j)
    endif
    j = j + 1
  enddo

  !..... now trace forward
  ip = np;
  dir = ddir;
  ifl2 = 0;
  
  xs = xmhd;
  ys = ymhd;
  zs = zmhd
  
  do k=1, maxst
    call trace2(gx, gy, gz, bx, by, bz, xs, ys, zs, xe, ye, ze, dir, nx, ny, nz)
    nsegs = nsegs + 1

    rs = sqrt(xe * xe + ye * ye + ze * ze)

    ! inner boundary
    if(rs.lt.3.7) then
      ifl2 = 1
      exit
    endif

    ! outer boundary
    if(xe.lt.gx(1).or.xe.gt.gx(nx)) then
      ifl2 = 2
      exit
    endif
    if(ye.lt.gy(1).or.ye.gt.gy(ny)) then
      ifl2 = 2
      exit
    endif
    if(ze.lt.gz(1).or.ze.gt.gz(nz)) then
      ifl2 = 2
      exit
    endif

    ip = ip + 1; 
    vl(1, ip) = xe;
    vl(2, ip) = ye;
    vl(3, ip) = ze
    if(iv.eq.1) then
      write(0,'(a,2i6,6(1x,f8.3))')'forward:  ',k,ip,xs,ys,zs,xe,ye,ze
    endif
    
    xs = xe;
    ys = ye;
    zs = ze;

  end do

  np = ip
  !.... determine topology
  !     ifl=0: closed  ifl=1:  open, ihem=-1: south  ihem=1: north  ifl=2:  SW
  ifl = 3;
  ihem = 0

  if(ifl1.eq.1.and.ifl2.eq.1) then
    ifl = 0
  endif
  if(ifl1.eq.1.and.ifl2.eq.2) then
    ifl = 1;
    ihem = -1;
  endif
  if(ifl1.eq.2.and.ifl2.eq.1) then
    ifl = 1;
    ihem = 1;
  endif
  if(ifl1.eq.2.and.ifl2.eq.2) then
    ifl=2
  endif
  if(iv.eq.1) then
    write(0,*)'xcl4b: ',ifl1,ifl2,ifl,ihem,imo,x,y,z,np
  endif
  
  nv = np;
  if(imo.eq.0) then
    return
  endif

  return

END SUBROUTINE


SUBROUTINE trace2(gx, gy, gz, bx, by, bz, &
                  x1, y1, z1, x2, y2, z2, dir, nx, ny, nz)
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: nx, ny, nz
  REAL, INTENT(IN) :: bx(nx,ny,nz), by(nx,ny,nz), bz(nx,ny,nz)
  REAL, INTENT(IN) :: gx(nx), gy(ny), gz(nz)
  REAL, INTENT(IN) :: x1, y1, z1
  REAL, INTENT(OUT) :: x2, y2, z2
  REAL, INTENT(IN) :: dir

  REAL :: r1
  REAL :: bb, bbx, bby, bbz, s
  INTEGER iout

  !..... trace one step
  r1 = sqrt(x1*x1 + y1*y1 + z1*z1)
  if(r1.gt.3.7) then
    call ipol3a(bx,bbx,gx,gy,gz,x1,y1,z1,iout,nx,ny,nz)
    if(iout.ne.0) goto 999
    call ipol3a(by,bby,gx,gy,gz,x1,y1,z1,iout,nx,ny,nz)
    if(iout.ne.0) goto 999
    call ipol3a(bz,bbz,gx,gy,gz,x1,y1,z1,iout,nx,ny,nz)
    if(iout.ne.0) goto 999
  else
    ! call cotr('gse','sm ',x1,y1,z1,xsm,ysm,zsm)
    ! xbbx = -3.0 * xsm * zsm
    ! xbby = -3.0 * ysm * zsm
    ! xbbz = r1 * r1 - 3.0 * zsm * zsm
    ! call cotr('sm ', 'gse', xbbx, xbby, xbbz, bbx, bby, bbz)
    x2 = 0
    y2 = 0
    z2 = 0
    return
  endif
  ! write(0,'(a,3(1x,g12.5))')' trace2: ',bbx,bby,bbz

  bb = sqrt(bbx*bbx + bby*bby + bbz*bbz)
  if(bb.le.0.0) goto 999
  s = dir / bb

  x2 = x1 + bbx * s
  y2 = y1 + bby * s
  z2 = z1 + bbz * s

  return

999   continue
  x2 = 1.e6
  y2 = 1.e6
  z2 = 1.e6
  return

END SUBROUTINE


SUBROUTINE ipol3a(a,b,gx,gy,gz,x,y,z,iout, nx,ny,nz)
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: nx, ny, nz
  REAL, INTENT(IN) :: a(nx, ny, nz)
  REAL, INTENT(IN) :: gx(nx), gy(ny), gz(nz)
  REAL, INTENT(IN) :: x, y, z
  REAL, INTENT(OUT) :: b
  INTEGER, INTENT(OUT) :: iout

  INTEGER, SAVE :: ixl = 1, iyl = 1, izl = 1
  ! INTEGER, SAVE :: 
  ! INTEGER, SAVE :: 
  INTEGER :: i, ix, iy, iz
  REAL :: dxi, dyi, dzi
  REAL :: x00, x01, x10, x11, y0, y1

  iout = 0

  ! find x index
  ix = ixl
  if(x.ge.gx(ix).and.x.le.gx(ix + 1)) goto 100
  ix = min0(ixl + 1, nx - 1)
  if(x.ge.gx(ix).and.x.le.gx(ix + 1)) goto 100
  ix = max0(ixl - 1, 1)
  if(x.ge.gx(ix).and.x.le.gx(ix + 1)) goto 100
  do i = 1, nx - 1
    ! WRITE(*, *) "i: ", i
    ! WRITE(*, *) "gx(i): ", gx(i)
    ! WRITE(*, *) "gx(i + 1): ", gx(i + 1)
    if(x.ge.gx(i).and.x.le.gx(i + 1)) then
      ix = i
      goto 100
    endif
  end do
  iout = 1
  return
100 continue
  ixl = ix
  ! WRITE(*, *) ">> x:", ix

  ! find y index
  iy = iyl
  if(y.ge.gy(iy).and.y.le.gy(iy + 1)) goto 200
  iy = min0(iyl + 1, ny - 1)
  if(y.ge.gy(iy).and.y.le.gy(iy + 1)) goto 200
  iy = max0(iyl - 1, 1)
  if(y.ge.gy(iy).and.y.le.gy(iy + 1)) goto 200
  do i = 1, ny - 1
    if(y.ge.gy(i).and.y.le.gy(i + 1)) then
      iy = i
      goto 200
    endif
  end do
  iout = 1
  return
200 continue
  iyl = iy

  ! find y index
  iz = izl
  if(z.ge.gz(iz).and.z.le.gz(iz + 1)) goto 300
  iz = min0(izl + 1, nz - 1)
  if(z.ge.gz(iz).and.z.le.gz(iz + 1)) goto 300
  iz = max0(izl - 1, 1)
  if(z.ge.gz(iz).and.z.le.gz(iz + 1)) goto 300
  do i = 1, nz - 1
    if(z.ge.gz(i).and.z.le.gz(i + 1)) then
      iz = i
      goto 300
    endif
  end do
  iout = 1
  return
300 continue
  izl = iz

  ! calc interpolation
  dxi = (x - gx(ix)) / (gx(ix + 1) - gx(ix))
  dyi = (y - gy(iy)) / (gy(iy + 1) - gy(iy))
  dzi = (z - gz(iz)) / (gz(iz + 1) - gz(iz))

  x00 = a(ix, iy, iz) + dxi * (a(ix + 1, iy, iz) - a(ix, iy, iz))
  x10 = a(ix, iy + 1, iz) + dxi * (a(ix + 1, iy + 1, iz) - a(ix, iy + 1, iz))
  x01 = a(ix, iy, iz + 1) + dxi * (a(ix + 1, iy, iz + 1) - a(ix, iy, iz + 1))
  x11 = a(ix, iy + 1, iz + 1) + dxi * (a(ix + 1, iy + 1, iz + 1) - a(ix, iy + 1, iz + 1))
  y0 = x00 + dyi * (x10 - x00)
  y1 = x01 + dyi * (x11 - x01)
  b = y0 + dzi * (y1 - y0)
  return

END SUBROUTINE
