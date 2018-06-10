subroutine inquire_next(found_field,IU,ndim,nx,ny,nz,it,varname,tstring)
  ! read the 'meta-data' from the next field in the file
  ! This will leave the file one field beyond where it was before
  ! called from python as:
  !
  ! was_found,ndim,nx,ny,nz,it = inquire_next(unit,vname,tstring)
  !
  IMPLICIT NONE
  integer :: IU
  !f2py intent(in) IU
  character(Len=80) :: varname,tstring,buff
  !f2py intent(inout) varname,tstring
  integer :: ndim,nx,ny,nz,it
  !f2py intent(out) ndim,nx,ny,nz,it
  integer :: found_field
  !f2py intent(out) found_field

  nx=-1;ny=-1;nz=-1;
  ndim=0
  found_field=0

  do while(.true.)
     read(IU,'(A)',end=888,err=888) buff
     if(buff.eq.'FIELD-1D-1')then
        found_field=1
        ndim=1
        read(IU,'(A)') varname
        read(IU,'(A)') tstring
        read(IU,*)it,nx
        exit
     elseif(buff.eq.'FIELD-2D-1')then
        found_field=1
        ndim=2
        read(IU,'(A)') varname
        read(IU,'(A)') tstring
        read(IU,*)it,nx,ny
        exit
     elseif(buff.eq.'FIELD-3D-1')then
        found_field=1
        ndim=3
        read(IU,'(A)') varname
        read(IU,'(A)') tstring
        read(IU,*)it,nx,ny,nz
        exit
     endif
  enddo
  !Get back to the beginning of this field
  backspace(IU)
  backspace(IU)
  backspace(IU)
  backspace(IU)
888 continue

  !  print*,ndim,nx,ny,nz,it,varname,tstring

end subroutine inquire_next

!----------------------------------------------------------------
subroutine read_jrrle1d(iu,a1,nx,l1,read_ascii,success)
  !----------------------------------------------------------------
  IMPLICIT NONE
  integer iu
  !f2py intent(in) iu
  integer nx
  !f2py intent(hidden) nx
  real a1(nx)
  !f2py intent(inplace) a1
  character(len=*) :: l1
  !f2py intent(inplace) l1
  logical read_ascii
  !f2py intent(in) read_ascii
  integer success
  !f2py intent(out) success

  !locals
  integer i
  integer it
  integer m
  real rid
  integer nn
  character*80 rec

  nn = nx
  rid=0.0
  if((iu.lt.0).or.(iu.gt.32768))then
     write(0,*)'Exception: Ambiguous I/O unit number.'
     return
  endif
  success=0
  !==============================================
  m=LEN(l1)
  call end0(l1,m)
100 continue
  read(iu,1000,end=190,err=190) rec
  if(rec(1:10).eq.'FIELD-1D-1') then
     read(iu,1000,end=190) rec
     if((l1(1:m).ne.rec(1:m))) goto 100
     l1=rec
     read(iu,1000,end=190) rec
     read(iu,*,end=190)it,nn
     call rdn2(iu,a1,nn,rec,it,rid)

     ! see if we have fullascii
     if (read_ascii .or. nn.ne.nx) then
       read(iu,'(a)')rec
       if(rec(2:17).eq.'fullasciifollows') then
         ! write(0,*)'reading:  ',rec(2:17)
         do i=1,nx; read(iu,*)a1(i); enddo
       else
         backspace(iu)
       endif
     endif

     if(nn.eq.nx)then
        success = 1
     endif
     return
  endif
  goto 100
190 continue
  success=0

1000 format(A)
  return
end subroutine read_jrrle1d

!----------------------------------------------------------------
subroutine read_jrrle2d(iu,a1,nx,l1,read_ascii,success)
  !----------------------------------------------------------------
  IMPLICIT NONE
  integer iu
  !f2py intent(in) iu
  integer nx !Total Number of points.  (note this is slightly different than getf21)
  !f2py intent(hidden) nx
  real a1(nx)
  !f2py intent(inplace) a1
  character(len=*) :: l1
  !f2py intent(inplace) l1
  logical read_ascii
  !f2py intent(in) read_ascii
  integer success
  !f2py intent(out) success

  !locals
  integer i
  integer it
  integer m
  real rid
  integer nn,nx1,ny1
  character*80 rec

  nn = nx
  rid=0.0
  if((iu.lt.0).or.(iu.gt.32768))then
     write(0,*)'Exception: Ambiguous I/O unit number.'
     return
  endif
  success=0
  !==============================================
  m=LEN(l1)
  call end0(l1,m)
100 continue
  read(iu,1000,end=190,err=190) rec
  if(rec(1:10).eq.'FIELD-2D-1') then
     read(iu,1000,end=190) rec
     if((l1(1:m).ne.rec(1:m))) goto 100
     l1=rec
     read(iu,1000,end=190) rec
     read(iu,*,end=190)it,nx1,ny1
     call rdn2(iu,a1,nn,rec,it,rid)

    ! see if we have fullascii
    if (read_ascii .or. nn.ne.nx) then
      read(iu,'(a)')rec
      if(rec(2:17).eq.'fullasciifollows') then
        ! write(0,*)'reading:  ',rec(2:17)
        do i=1,nx; read(iu,*)a1(i); enddo
      else
        backspace(iu)
      endif
    endif

    if(nx1*ny1.eq.nx)then
      success = 1
    endif
    return
  endif
  goto 100

190 continue
  success=0

1000 format(A)
  return
end subroutine read_jrrle2d

!----------------------------------------------------------------
subroutine read_jrrle3d(iu,a1,nx,l1,read_ascii,success)
  !----------------------------------------------------------------
  IMPLICIT NONE
  integer iu
  !f2py intent(in) iu
  integer nx
  !f2py intent(hidden) nx
  real a1(nx)
  !f2py intent(inplace) a1
  character(len=*) :: l1
  !f2py intent(inplace) l1
  logical read_ascii
  !f2py intent(in) read_ascii
  integer success
  !f2py intent(out) success

  !locals
  integer i
  integer it
  integer m
  real rid
  integer nn,nx1,ny1,nz1
  character*80 rec

  nn = nx
  rid=0.0
  if((iu.lt.0).or.(iu.gt.32768))then
     write(0,*)'Exception: Ambiguous I/O unit number.'
     return
  endif
  success=0
  !==============================================
  m=LEN(l1)
  call end0(l1,m)
100 continue
  read(iu,1000,end=190,err=190) rec
  if(rec(1:10).eq.'FIELD-3D-1') then
     read(iu,1000,end=190) rec
     if((l1(1:m).ne.rec(1:m))) goto 100
     l1=rec
     read(iu,1000,end=190) rec
     read(iu,*,end=190)it,nx1,ny1,nz1
     call rdn2(iu,a1,nn,rec,it,rid)

     ! see if we have fullascii
     if (read_ascii .or. nn.ne.nx) then
       read(iu,'(a)')rec
         if(rec(2:17).eq.'fullasciifollows') then
         ! write(0,*)'reading:  ',rec(2:17)
         do i=1,nx; read(iu,*)a1(i); enddo
       else
         backspace(iu)
       endif
     endif

     if((nx1*ny1*nz1.eq.nx))then
        success = 1
     endif
     return
  endif
  goto 100
190 continue
  success=0

1000 format(A)
  return
end subroutine read_jrrle3d


!-----------------------------------------------------------
subroutine write_jrrle1d(iu,a1,nx,l1,l2,it)
  !-----------------------------------------------------------
  IMPLICIT NONE
  integer iu                !io unit
  !f2py intent(in) iu
  character*80 l1,l2        !l1 variable name, l2: ascii time string
  !f2py intent(in) l1,l2
  real a1(nx)                !array to write
  !f2py intent(in) a1
  integer nx,it             !nx: dimension of array, it: integer time
  !f2py intent(in) nx,it

  write(iu,'(a)')'FIELD-1D-1'
  write(iu,'(a)') l1
  write(iu,'(a)') l2
  write(iu,*) it,nx
  call wrn2(iu,a1,nx,'FUNC-1-1',it,float(it))
  return
end subroutine write_jrrle1d

!-----------------------------------------------------------
subroutine write_jrrle2d(iu,a1,nx,ny,l1,l2,it)
!-----------------------------------------------------------
  IMPLICIT NONE
  integer iu                !io unit
  !f2py intent(in) iu
  character*80 l1,l2        !l1 variable name, l2: ascii time string
  !f2py intent(in) l1,l2
  real a1(nx*ny)                !array to write
  !f2py intent(in) a1
  integer nx,ny,it          !nx: dimension of array, it: integer time
  !f2py intent(in) nx,ny,it

  write(iu,'(a)')'FIELD-2D-1'
  write(iu,'(a)') l1
  write(iu,'(a)') l2
  write(iu,*)it,nx,ny
  call wrn2(iu,a1,nx*ny,'FUNC-2-1',it,float(ny))
  return
end subroutine write_jrrle2d

!-----------------------------------------------------------
subroutine write_jrrle3d(iu,a1,nx,ny,nz,l1,l2,it)
!-----------------------------------------------------------
  IMPLICIT NONE
  integer iu                !io unit
  !f2py intent(in) iu
  character*80 l1,l2        !l1 variable name, l2: ascii time string
  !f2py intent(in) l1,l2
  real a1(nx*ny*nz)                !array to write
  !f2py intent(in) a1
  integer nx,ny,nz,it          !nx: dimension of array, it: integer time
  !f2py intent(in) nx,ny,nz,it

  write(iu,'(a)')'FIELD-3D-1'
  write(iu,'(a)') l1
  write(iu,'(a)') l2
  write(iu,*)it,nx,ny,nz
  call wrn2(iu,a1,nx*ny*nz,'FUNC-3-1',it,float(ny))
  return
end subroutine write_jrrle3d

!!!!!!!!!!! These routines probably don't need to be converted to python !!!!!!!!!!!

!-----------------------------------------------------------
subroutine end0(r,m)
  !
  ! find the end of a string (ie first white space)
  !
  ! assumes strings are space padded and that 80 is the
  ! maximum length.
  !
  !-----------------------------------------------------------
  IMPLICIT NONE
  integer i,n,m
  character*80 r

  ! presumably m is a reasonable value to begin with...
  n = min0(m,len(r))

  ! find first space in string, return index just prior
  do i = 1,n
     m = i - 1
     if ( r(i:i).eq.' ' ) return
  enddo

  ! or return the smaller of the array length and
  ! 80 if no spaces are encountered
  m=n

  return

end subroutine end0

! Change log---
!
! Burlen Loring 2008-01-03
!
! added comments, indetantion, and implicit none
!.---------------------------------------
subroutine rdn2( fileNo,a,n,cid,it,rid )
  !
  !
  !
  ! character decoding
  !.---------------------------------------
  real a(n)
  integer n
  integer it
  real rid
  character*8 cid
  character*4 did
  character*8 nid
  real a1(0:63)
  integer intRec1(0:63),intRec2(0:63),i3(0:63)
  integer q
  integer fileNo

  read( UNIT=fileNo,FMT='(a4,a8,3e14.7,i8,a)',ERR=100,END=900 ) &
       did,nid,zmin,zmax,rid,it,cid
  ! The format of the header record is as follows:
  ! 1, 4 char string                                    ( did )
  ! 8 chars to 1 int                                    ( n )
  ! 3 times 14 chars to double w/ 7 digits precision    ( zmin,zmax,rid )
  ! 8 chars to 1 int                                    ( it )
  ! 1, 8 char string                                    ( cid )

  ! if nid is "********" then the number of grid cells is too large for
  ! the jrrle format, so we need to assume that the value passed in as
  ! `n` is correct, or else all is lost
  if (nid.ne."********") then
    read(nid,'(i8)') nid_asint
    if (nid_asint.ne.n) then
      write(0,*)'rdn2: nid (nid_asint) .ne. n ', nid, nid_asint, n
    endif
    n = nid_asint
  endif

  ! try read again, if we don't have WRN2 data
  if( did.ne.'WRN2' ) then
     goto 100
  endif

  ! data array is constant, initialize a and return
  if(zmin.eq.zmax) then
     do i=1,n
        a(i)=zmin
     enddo
     return
     ! compute data resolution
  else
     ! what is the significance of 4410??
     ! and why the explicit cast rather than 4410.0??
     dzi=(zmax-zmin)/float(4410)
  endif

  ! process data records, these are 64 characters long
  q=0 ! index into decompressed real*8 data
  do k=1,n,64

     ! records should be 64 ints long but last one may be shorter
     nk = min0(63,n-k)

     ! decompress a record( expands and converts ascii to hex)
     call wrndec(fileNo,intRec1,nn)

     ! error in record length bail
     ! note: if returned nn=-5 explicitly indicates an error
     if(nn.ne.nk) then
        write(0,*)'rdn2: nn .ne. nk ',nn,nk,n,k
        n=-2
        return
     endif

     ! decompress next record( expands and converts ascii to hex)
     call wrndec(fileNo,intRec2,nn)

     ! error in record length bail
     ! note: if returned nn=-5 explicitly indicates an error
     if(nn.ne.nk) then
        write(0,*)'rdn2: nn .ne. nk ',nn,nk
        n=-2
        return
     endif

     ! convert these records from hex-ascii to double
     ! not sure about the specifics here except that
     ! this is the propritary OpenGGCM format
     ! what is significance of 33,47 and 94 ??
     do i=0,nk
        intRec1(i) = intRec1(i) - 33
        intRec2(i) = intRec2(i) - 33
        sig        = 1.

        if( intRec1(i).ge.47 ) then
           sig        = -1.
           intRec1(i) = intRec1(i) - 47
        endif

        i3(i) = intRec2(i) + 94*intRec1(i)
        a1(i) = FLOAT(i3(i))
        a1(i) = dzi*a1(i) + zmin
        a1(i) = sig*exp(a1(i))

        q    = q + 1
        a(q) = a1(i)
     enddo
  enddo

  return

  ! there was no rle data (ascii only?)
100 continue
  n=-1
  backspace(fileNo)
  return

  ! io error occured
900 continue
  n=-1
  return
end subroutine rdn2


!---------------------------------------
subroutine wrndec(fileNo,expandedDatOut,n)
  !
  ! decode compressed data sequence read in from a file
  ! will decode a single record (typically 64 chars ion length)
  !
  ! the decompressed record is returned in exdpandedDataOut
  ! the legth of the de-compressed record is returned in n
  ! n=-5 indicates something bad happened
  !
  !.---------------------------------------
  IMPLICIT NONE
  integer fileNo
  integer n
  integer nRep,hexChar
  integer checkSum
  integer i,j,q
  integer expandedDatOut(0:63) ! results returned
  integer expandedDat(-2:67)   ! work array has expanded data + check sum
  character*72 asciiDat        ! read buffer

  ! initialize some of the variables
  ! note: read buffer is space padded, spaces indicate end of field
  n=-5
  expandedDatOut(0)=0
  asciiDat='                                                                        '

  ! read 72 characters from file(record assumed to be 64 chars)
  read( UNIT=fileNo,FMT='(A)',END=900,ERR=900 ) asciiDat


  ! expand sequences in buffer. An ascii value with 7th bit set
  ! is a count(2^7=128), this tells how many times following ascii
  ! char is repeated. If the 7th bit is not set then the ascii char
  ! appears once.
  ! this is an infinite loop which breaks once a space is encountered
  ! or too many chars have been processed (ie record length exceeded).
  i=-2 ! index into expanded data
  j=0  ! index into read buffer
100 continue
  j = j + 1

  ! error records should not be this long, bail
  if(j.gt.67) then
     write(0,*)'wrndec: cannot find end of encoded record'
     n=-5
     return
  endif

  ! convert j-th value from ascii to hex
  hexChar=ICHAR( asciiDat(j:j) )

  ! if its a space then we are done expanding this record, break
  if ( hexChar.eq.32 ) then
     goto 190
  endif

  ! if its not an 7 bit ascii character
  ! then it tells how many time to repeat the following ascii char
  if ( hexChar.gt.127 ) then

     nRep = hexChar-170 ! convert to count ??

     ! convert next j-th value from ascii to hex
     ! this is repeated nRep times
     j  = j + 1
     hexChar = ICHAR( asciiDat(j:j) )
     do q=1,nRep
        i = i + 1
        expandedDat(i) = hexChar
     enddo

     ! otherwise it is 7 bit ascii, and apears only once
  else
     i = i + 1
     expandedDat(i) = hexChar
  endif

  ! expand next sequence in buffer
  goto 100

190 continue

  ! error we expect field to be 64 chars if its more then this
  ! there is a problem, bail
  n=i
  if( n.gt.63 ) then
     write(0,*)'wrndec: n gt 63, n=',n
     n=-5
     write(0,'(a)')'rec:'
     write(0,'(a)')asciiDat
     return
  endif

  ! copy and compute check sum
  checkSum=0
  do i=0,n
     ! copy
     expandedDatOut(i) = expandedDat(i)
     ! compute check sum
     checkSum = checkSum + expandedDatOut(i)
  enddo

  ! check sum error bail
  if(expandedDat(-1).ne.33+mod(checkSum,92)) then
     write(0,*)'wrndec: checksum error '
     write(0,'(a,a)')'rec:',asciiDat
     write(0,*)expandedDat(-1),33+mod(checkSum,92)
     n=-5
     return
  endif

  return

  ! file io error
900 continue
  write(0,*)' wrndec eof/err '
  n=-5
  return

end subroutine wrndec


!.---------------------------------------
subroutine wrn2(iu,a,n,cid,it,rid)
  !.---------------------------------------
  IMPLICIT NONE
  integer n
  integer it
  integer iu
  real z0
  real a(n)
  real b
  real zmin,zmax
  real z1,z2
  real dz
  real rid
  character*8 cid
  real a1(0:63)
  integer nk
  integer k,i
  integer i1(0:63),i2(0:63),i3(0:63)
  zmin=1.e33
  zmax=1.e-33
  do i=1,n
     b=abs(a(i))
     zmin=amin1(zmin,b)
     zmax=amax1(zmax,b)
  enddo
  zmin=amax1(1.e-33,zmin)
  zmax=amin1( 1.e33,zmax)
  z2=alog(zmax)
  z1=-76.
  if(zmin.ne.0.)then
     z1=alog(zmin)
  endif
  z1=amax1(z1,z2-37.)
  if(abs(z2-z1).le.1.e-5)then
     z1=z1-1.0
     z2=z2+1.0
  endif
  z0=exp(z1)
  dz=float(4410)/(z2-z1)
1000 format(a,i8,3e14.7,i8,a)
  write(iu,1000)'WRN2',n,z1,z2,rid,it,cid
  do k=1,n,64
     nk=min0(63,n-k)
     do i=0,nk
        a1(i)=amin1(1.e33,abs(a(i+k)))
        a1(i)=amax1(z0,a1(i))
        a1(i)=dz*(alog(a1(i))-z1)+0.5
        i3(i)=INT(a1(i))
        i3(i)=max0(0,i3(i))
        i3(i)=min0(4414,i3(i))
        i1(i)=i3(i)/94
        i2(i)=i3(i)-94*i1(i)
        if(a(i+k).lt.0.)then
           i1(i)=i1(i)+47
        endif
        i1(i)=i1(i)+33
        i2(i)=i2(i)+33
     enddo
     call wrnenc(iu,i1,nk)
     call wrnenc(iu,i2,nk)
  enddo
  return
end subroutine wrn2


!.---------------------------------------
subroutine wrnenc(iu,i1,n)
  !.---------------------------------------
  integer iu
  integer n
  integer i1(0:63)
  integer i2(-1:63)
  character c*72
  ick=i1(0)
  ir=i1(0)
  ic=1
  k=-1
  do i=1,n
     ick=ick+i1(i)
     if(i1(i).eq.ir) then
        ic=ic+1
     else
        if(ic.eq.1) then
           k=k+1
           i2(k)=ir
        else
           k=k+1
           i2(k)=ic+170
           k=k+1
           i2(k)=ir
        endif
        ic=1
        ir=i1(i)
     endif
  enddo
  if(ic.eq.1) then
     k=k+1
     i2(k)=ir
  else
     k=k+1
     i2(k)=ic+170
     k=k+1
     i2(k)=ir
  endif
  i2(-1)=33+mod(ick,92)
  j=0
  do i=-1,k
     j=j+1
     c(j:j)=char(i2(i))
  enddo
  write(iu,'(a)')c(1:j)
  return
end subroutine wrnenc
