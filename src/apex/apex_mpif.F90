
       subroutine apex_mpi_fortran_init_predefined_constants()
       include 'mpif.h'
       call apex_mpi_predef_init_in_place(MPI_IN_PLACE)
       call apex_mpi_predef_init_bottom(MPI_BOTTOM)
       call apex_mpi_predef_init_status_ignore(MPI_STATUS_IGNORE)
       call apex_mpi_predef_init_statuses_ignore(MPI_STATUSES_IGNORE)
       call apex_mpi_predef_init_unweighted(MPI_UNWEIGHTED)
       return
       end subroutine apex_mpi_fortran_init_predefined_constants

       subroutine apex_mpi_fortran_init_predefined_constants_()
       include 'mpif.h'
       call apex_mpi_predef_init_in_place(MPI_IN_PLACE)
       call apex_mpi_predef_init_bottom(MPI_BOTTOM)
       call apex_mpi_predef_init_status_ignore(MPI_STATUS_IGNORE)
       call apex_mpi_predef_init_statuses_ignore(MPI_STATUSES_IGNORE)
       call apex_mpi_predef_init_unweighted(MPI_UNWEIGHTED)
       return
       end subroutine apex_mpi_fortran_init_predefined_constants_
