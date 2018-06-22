
#ifdef FSEEKABLE
#define HAVE_FSEEK 1
#define HAVE_FTELL 1
#endif

  subroutine seek(unit,offset,whence,status)
    implicit none
    integer unit
    INTEGER*8 offset
    integer whence
    integer status
    !f2py intent(in) unit
    !f2py intent(in) offset
    !f2py integer optional, intent(in) :: whence=0
    !f2py intent(out) status
    !Non-zero exit status on exit if this routine fails
    status = -1

#ifdef HAVE_FSEEK
    call fseek(unit,offset,whence,status)
#else
    read(unit,'()',ADVANCE='NO',POS=offset,IOSTAT=status)
#endif
    return

  end subroutine seek

  subroutine tell(unit,offset)
    implicit none
    integer unit
    INTEGER*8 offset
    !f2py intent(in) unit
    !f2py intent(out) offset
    offset = -1

#ifdef HAVE_FTELL
    call ftell(unit,offset)
#else
    inquire(UNIT=unit,POS=offset)
#endif

    return
  end subroutine tell

  subroutine freefileunit(uu,funit)
    IMPLICIT NONE
    integer uu
    !f2py integer optional uu=-1
    integer funit
    !f2py intent(out) funit

    logical isopen
    integer i

    ! Specifically check for this unit
    if(uu.gt.0)then
       inquire(unit=uu,opened=isopen)
       if(.not.isopen)then
          funit=uu
          return
       endif
    endif

    !look for a free file unit
    do i=10,10000
       if((i.ge.100) .and. (i.le.102))then
          !some implementations reserve these units
          cycle
       endif
       inquire(unit=i,opened=isopen)
       if(.not.isopen)then
          funit=i
          exit
       endif
    enddo

  end subroutine freefileunit

  subroutine frewind(funit,debug)
    IMPLICIT NONE
    integer :: funit
    !f2py intent(in) funit
    integer debug
    !f2py integer optional debug=0
    if(debug.gt.0) print*,"Rewinding unit:",funit

    rewind(funit)
  end subroutine frewind


  subroutine fbackspace(funit,debug)
    IMPLICIT NONE
    integer :: funit
    !f2py intent(in) funit
    integer debug
    !f2py integer optional debug=0

    if(debug.gt.1) print*,"backspacing unit:",funit

    backspace(funit)
  end subroutine fbackspace

  subroutine fopen(funit,uu,fname,debug)
    !TODO: open as binary or for appending
    ! open a fortran file
    IMPLICIT NONE
    integer :: funit,uu
    !f2py intent(out) funit
    !f2py integer optional uu=-1
    character(Len=*) :: fname
    !f2py intent(in) fname
    integer debug
    !f2py integer optional debug=0
    logical isopen
    integer openstat
    character*10 access_method

#ifdef HAVE_STREAM
    access_method = 'STREAM'
#else
    access_method = 'SEQUENTIAL'
#endif

    inquire(file=fname,opened=isopen,number=funit)
    if(isopen)then
       if(debug.gt.0) print*,"file already opened:",fname
       if(debug.gt.0) print*,"associated unit:",funit
       return
    endif

    call freefileunit(uu,funit)
    open(unit=funit,file=fname,status='UNKNOWN',form='FORMATTED',access=access_method,IOSTAT=openstat)

    if (openstat.ne.0) then
      funit = -1 * openstat
    endif

    if(debug.gt.0) print*,"opened file:",fname
    if(debug.gt.0) print*,"associated unit:",funit
    if(debug.gt.0) print*,"access method:",access_method

  end subroutine fopen

  subroutine fadvance_one_line(success,funit,debug)
    !advance one line in the file
    IMPLICIT NONE
    integer :: success,funit
    !f2py intent(out) success
    !f2py intent(in) funit
    character(Len=1) :: a
    integer debug
    !f2py integer optional debug=0
    if(debug.gt.1) print*,"advancing unit by 1 line:",funit


    success=0
    read(funit,'(A)',err=777,end=777) a
    success=1
    777 continue ! continue on error

  end subroutine fadvance_one_line

  subroutine fisopen(funit,isopen,debug)
    ! Check if a fortran file unit is open
    integer :: funit
    !f2py intent(in) funit
    integer :: isopen
    !f2py intent(out) isopen
    integer :: debug
    !f2py integer optional debug=0
    logical o

    inquire(unit=funit,opened=o)
    if(o)then
       isopen=1
    else
       isopen=0
    endif
  end subroutine fisopen

  subroutine fclose(funit,debug)
    ! close a fortran file
    integer :: funit
    !f2py intent(in) funit
    integer debug
    !f2py integer optional debug=0
    logical isopen
    if(debug.gt.0) print*,"closing unit:",funit

    inquire(unit=funit,opened=isopen)
    if(isopen)then
       close(unit=funit)
    else
       print*,"Cannot close unit that isn't open",funit
    endif

  end subroutine fclose

  !TODO: fseek a binary unit
