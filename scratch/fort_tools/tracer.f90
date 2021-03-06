! -*- f90 -*-

!SUBROUTINE ffort_interp_trilin(a, b, gx, gy, gz, ptsx, ptsy, ptsz, nx, ny, nz, npts)
!  IMPLICIT NONE
!
!  INTEGER, INTENT(IN) :: nx, ny, nz
!  INTEGER, INTENT(IN) :: npts
!  REAL, INTENT(IN) :: a(nx, ny, nz)
!  REAL, INTENT(IN) :: gx(nx), gy(ny), gz(nz)
!  REAL, INTENT(IN) :: ptsx(npts), ptsy(npts), ptsz(npts)
!  REAL, INTENT(INOUT) :: b(npts)
!

!---------------------------------------------------------------
SUBROUTINE get_topo_at(topo, nsegs, ptsx, ptsy, ptsz, bx, by, bz, gx, gy, gz, &
                       npts, nx, ny, nz)
!---------------------------------------------------------------
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: npts
  INTEGER, INTENT(IN) :: nx, ny, nz
  REAL, INTENT(IN) :: gx(nx), gy(ny), gz(nz)
  REAL, INTENT(IN) :: bx(nx, ny, nz), by(nx, ny, nz), bz(nx, ny, nz)
  REAL, INTENT(IN) :: ptsx(npts), ptsy(npts), ptsz(npts)
  INTEGER, INTENT(INOUT) :: topo(npts)
  INTEGER, INTENT(INOUT) :: nsegs

  REAL :: x, y, z
  REAL :: vl(3, 100000)
  INTEGER :: i
  INTEGER :: nv, ifl, ihem
  INTEGER :: nopen, nclose, nsw

  nopen = 0
  nclose = 0
  nsw = 0

  do i=1, npts
    x = ptsx(i)
    y = ptsy(i)
    z = ptsz(i)

    call xcl4(gx, gy, gz, bx, by, bz, x, y, z, ifl, ihem, 0, 100, 2, vl, nv, nx, ny, nz, nsegs)
    topo(i) = ifl
  end do
END SUBROUTINE

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
  INTEGER :: ix, iy, iz
  INTEGER :: nv, ifl, ihem
  INTEGER :: nopen, nclose, nsw

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
      end do
    end do
  end do
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

  ! TOPOLOGY_INVALID = 1 # 1, 2, 3, 4, 9, 10, 11, 12, 15
  ! TOPOLOGY_CLOSED = 7 # 5 (both N), 6 (both S), 7(both hemispheres)
  ! TOPOLOGY_SW = 8
  ! TOPOLOGY_OPEN_NORTH = 13
  ! TOPOLOGY_OPEN_SOUTH = 14
  ! TOPOLOGY_OTHER = 16 # >= 16

  if(ifl1.eq.1.and.ifl2.eq.1) then
    ifl = 7
  endif
  if(ifl1.eq.1.and.ifl2.eq.2) then
    ifl = 14;
    ihem = -1;
  endif
  if(ifl1.eq.2.and.ifl2.eq.1) then
    ifl = 13;
    ihem = 1;
  endif
  if(ifl1.eq.2.and.ifl2.eq.2) then
    ifl = 8
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
  INTEGER :: iout = 0

  !..... trace one step
  r1 = sqrt(x1*x1 + y1*y1 + z1*z1)
  if(r1.gt.3.7) then
    call ipol3a(bx,bbx,gx,gy,gz,x1,y1,z1,iout,nx,ny,nz)
    call ipol3a(by,bby,gx,gy,gz,x1,y1,z1,iout,nx,ny,nz)
    call ipol3a(bz,bbz,gx,gy,gz,x1,y1,z1,iout,nx,ny,nz)
    if(iout.ne.0) then
      x2 = 1.e6
      y2 = 1.e6
      z2 = 1.e6
      return
    endif
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
  if(bb.gt.0.0) then
    s = dir / bb

    x2 = x1 + bbx * s
    y2 = y1 + bby * s
    z2 = z1 + bbz * s
  else
    x2 = 1.e6
    y2 = 1.e6
    z2 = 1.e6
  endif

  return

END SUBROUTINE

SUBROUTINE ffort_interp_trilin(a, b, gx, gy, gz, ptsx, ptsy, ptsz, nx, ny, nz, npts)
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: nx, ny, nz
  INTEGER, INTENT(IN) :: npts
  REAL, INTENT(IN) :: a(nx, ny, nz)
  REAL, INTENT(IN) :: gx(nx), gy(ny), gz(nz)
  REAL, INTENT(IN) :: ptsx(npts), ptsy(npts), ptsz(npts)
  REAL, INTENT(INOUT) :: b(npts)

  INTEGER :: i
  INTEGER :: iout
  REAL :: val = 0

  iout = 0

  DO i = 1, npts
    call ipol3a(a, val, gx, gy, gz, ptsx(i), ptsy(i), ptsz(i), iout, nx, ny, nz)
    if(iout.gt.0) then
      return
    endif
    b(i) = val
  END DO

END SUBROUTINE

SUBROUTINE ipol3a(a,b,gx,gy,gz,x,y,z,iout, nx,ny,nz)
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: nx, ny, nz
  REAL, INTENT(IN) :: a(nx, ny, nz)
  REAL, INTENT(IN) :: gx(nx), gy(ny), gz(nz)
  REAL, INTENT(IN) :: x, y, z
  REAL, INTENT(OUT) :: b
  INTEGER, INTENT(OUT) :: iout

  INTEGER, SAVE :: ix = 1, iy = 1, iz = 1
  REAL :: dxi, dyi, dzi
  REAL :: x00, x01, x10, x11, y0, y1

  call closest_ind(gx, nx, x, ix, iout)
  call closest_ind(gy, ny, y, iy, iout)
  call closest_ind(gz, nz, z, iz, iout)

  if(iout.gt.0) then
    write(*, *) "IOUT:", iout
    return
  endif

  ! calc interpolation
  dxi = (x - gx(ix)) / (gx(ix + 1) - gx(ix))
  dyi = (y - gy(iy)) / (gy(iy + 1) - gy(iy))
  dzi = (z - gz(iz)) / (gz(iz + 1) - gz(iz))

  call interp3(a, ix, iy, iz, dxi, dyi, dzi, b, nx, ny, nz)

  return

END SUBROUTINE

PURE SUBROUTINE closest_ind(gx, nx, x, ix, iout)
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: nx
  REAL, INTENT(IN) :: gx(nx)
  REAL, INTENT(IN) :: x
  INTEGER, INTENT(INOUT) :: ix
  INTEGER, INTENT(INOUT) :: iout
  INTEGER i

  ! already there?
  if(x.ge.gx(ix).and.x.le.gx(ix + 1)) then
    return
  endif
  ! what about one cell ahead?
  ix = min0(ix + 1, nx - 1)
  if(x.ge.gx(ix).and.x.le.gx(ix + 1)) then
    return
  endif
  ! and one behind?
  ix = max0(ix - 2, 1)
  if(x.ge.gx(ix).and.x.le.gx(ix + 1)) then
    return
  endif
  ! ok then search
  do i = 1, nx - 1
    if(x.ge.gx(i).and.x.le.gx(i + 1)) then
      ix = i
      return
    endif
  end do

  ! oh man, we never found it
  iout = iout + 1
  return
END SUBROUTINE

PURE SUBROUTINE interp3(a, ix, iy, iz, dxi, dyi, dzi, b, nx, ny, nz)
  IMPLICIT NONE

  INTEGER, INTENT(IN) :: nx, ny, nz
  REAL, INTENT(IN) :: a(nx, ny, nz)
  INTEGER, INTENT(IN) :: ix, iy, iz
  REAL, INTENT(IN) :: dxi, dyi, dzi
  REAL, INTENT(OUT) :: b
  REAL :: x00, x01, x10, x11, y0, y1

  x00 = a(ix, iy    , iz    ) + dxi * (a(ix + 1, iy    , iz    ) - a(ix, iy    , iz    ))
  x10 = a(ix, iy + 1, iz    ) + dxi * (a(ix + 1, iy + 1, iz    ) - a(ix, iy + 1, iz    ))
  x01 = a(ix, iy    , iz + 1) + dxi * (a(ix + 1, iy    , iz + 1) - a(ix, iy    , iz + 1))
  x11 = a(ix, iy + 1, iz + 1) + dxi * (a(ix + 1, iy + 1, iz + 1) - a(ix, iy + 1, iz + 1))
  y0 = x00 + dyi * (x10 - x00)
  y1 = x01 + dyi * (x11 - x01)
  b = y0 + dzi * (y1 - y0)
END SUBROUTINE
