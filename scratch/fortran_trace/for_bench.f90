! -*- f90 -*-

subroutine fortran_entry(arr_in, arr_out, nfcalls, total, nx_in, nx_out)
  implicit none
  integer, intent(in) :: nx_in
  integer, intent(in) :: nx_out
  real, intent(in) :: arr_in(nx_in)
  real, intent(inout) :: arr_out(nx_out)
  real, intent(out) :: total
  integer, intent(in) :: nfcalls

  integer :: i

  do i = 0, nfcalls
    call worker(arr_in, arr_out, total, nx_in, nx_out)
    call worker2(arr_in, arr_out, total, nx_in, nx_out)
  end do

end subroutine


subroutine worker(arr_in, arr_out, total, nx_in, nx_out)
  implicit none
  integer, intent(in) :: nx_in
  integer, intent(in) :: nx_out
  real, intent(in) :: arr_in(nx_in)
  real, intent(inout) :: arr_out(nx_out)
  real, intent(out) :: total

  integer :: i = 2

  ! do i = 1, nx_in
  !   arr_out(i) = 2.0 * arr_in(i)
  !   total = total + arr_in(i)
  ! end do
  total = total + arr_out(i)

end subroutine

subroutine worker2(arr_in, arr_out, total, nx_in, nx_out)
  implicit none
  integer, intent(in) :: nx_in
  integer, intent(in) :: nx_out
  real, intent(in) :: arr_in(nx_in)
  real, intent(inout) :: arr_out(nx_out)
  real, intent(out) :: total

  integer :: i = 3

  ! do i = 1, nx_in
  !   arr_out(i) = 2.0 * arr_in(i)
  !   total = total + arr_in(i)
  ! end do
  total = total + arr_out(i)

end subroutine
