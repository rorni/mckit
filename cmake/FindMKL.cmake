################################################################################
#
# \file      cmake/FindMKL.cmake
# \author    J. Bakosi
# \copyright 2012-2015, Jozsef Bakosi, 2016, Los Alamos National Security, LLC.
# \brief     Find the Math Kernel Library from Intel
# \date      Thu 26 Jan 2017 02:05:50 PM MST
#
################################################################################

# Find the Math Kernel Library from Intel
#
#  MKL_FOUND - System has MKL
#  MKL_INCLUDE_DIRS - MKL include files directories
#  MKL_LIBRARIES - The MKL libraries
#  MKL_INTERFACE_LIBRARY - MKL interface library
#  MKL_SEQUENTIAL_LAYER_LIBRARY - MKL sequential layer library
#  MKL_CORE_LIBRARY - MKL core library
#
#  The environment variables MKLROOT and INTEL are used to find the library.
#  Everything else is ignored. If MKL is found "-DMKL_ILP64" is added to
#  CMAKE_C_FLAGS and CMAKE_CXX_FLAGS.
#
#  Example usage:
#
#  find_package(MKL)
#  if(MKL_FOUND)
#    target_link_libraries(TARGET ${MKL_LIBRARIES})
#  endif()

message("Find MKL from PyMesh")

# If already in cache, be silent
if (MKL_INCLUDE_DIRS AND MKL_LIBRARIES AND MKL_INTERFACE_LIBRARY AND
        MKL_SEQUENTIAL_LAYER_LIBRARY AND MKL_CORE_LIBRARY)
    set (MKL_FIND_QUIETLY TRUE)
endif()

#if(NOT BUILD_SHARED_LIBS)
#    set(INT_LIB "libmkl_intel_ilp64.a")
#    set(SEQ_LIB "libmkl_sequential.a")
#    set(THR_LIB "libmkl_intel_thread.a")
#    set(COR_LIB "libmkl_core.a")
#else()
#    set(INT_LIB "mkl_intel_ilp64")
#    set(SEQ_LIB "mkl_sequential")
#    set(THR_LIB "mkl_intel_thread")
#    set(COR_LIB "mkl_core")
#endif()


find_path(MKL_INCLUDE_DIR NAMES mkl.h
        HINTS $ENV{MKLROOT}/include
        ${MKLROOT}/include
        $ENV{INTEL}/mkl/include
        ${INTEL}/mkl/include)

cmake_print_variables(MKL_INCLUDE_DIR)

#find_library(MKL_INTERFACE_LIBRARY
#        NAMES ${INT_LIB}
#        PATHS $ENV{MKLROOT}/lib
#        $ENV{MKLROOT}/lib/intel64
#        $ENV{INTEL}/mkl/lib
#        $ENV{INTEL}/mkl/lib/intel64
#        ${MKLROOT}/lib
#        ${MKLROOT}/lib/intel64
#        ${INTEL}/mkl/lib
#        ${INTEL}/mkl/lib/intel64
#        NO_DEFAULT_PATH)

#cmake_print_variables(MKL_INTERFACE_LIBRARY)

#find_library(MKL_SEQUENTIAL_LAYER_LIBRARY
#        NAMES ${SEQ_LIB}
#        PATHS $ENV{MKLROOT}/lib
#        $ENV{MKLROOT}/lib/intel64
#        $ENV{INTEL}/mkl/lib
#        $ENV{INTEL}/mkl/lib/intel64
#        ${MKLROOT}/lib
#        ${MKLROOT}/lib/intel64
#        ${INTEL}/mkl/lib
#        ${INTEL}/mkl/lib/intel64
#        NO_DEFAULT_PATH)

#find_library(MKL_CORE_LIBRARY
#        NAMES ${COR_LIB}
#        PATHS $ENV{MKLROOT}/lib
#        $ENV{MKLROOT}/lib/intel64
#        $ENV{INTEL}/mkl/lib
#        $ENV{INTEL}/mkl/lib/intel64
#        ${MKLROOT}/lib
#        ${MKLROOT}/lib/intel64
#        ${INTEL}/mkl/lib
#        ${INTEL}/mkl/lib/intel64
#        NO_DEFAULT_PATH)

cmake_print_variables(MKLROOT)

function(check_lib validator_result_var item)
    message(STATUS "Check: ${item}")
    if(NOT item MATCHES ".*mkl_rt\.so.*")
        set(${validator_result_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

find_library(MKL_RT_LIBRARY
    NAMES mkl_rt
    PATHS ${MKLROOT}/lib
#    $ENV{MKLROOT}/lib/intel64
#    $ENV{INTEL}/mkl/lib
#    $ENV{INTEL}/mkl/lib/intel64
#    ${MKLROOT}/lib
#    ${MKLROOT}/lib/intel64
#    ${INTEL}/mkl/lib
#    ${INTEL}/mkl/lib/intel64
    VALIDATOR check_lib
    REQUIRED
)
# This works if the link libmkl_rt.so -> libmkl_rt.so.2 is created
# Doing this automatically is possible but too cumbersome
# Intel guys require to use full names of the libraries in
# link statement.
# See implementation in our building/extension_geometry.py.
# TODO fix for absent link and for darwin and windows libs

cmake_print_variables(MKL_RT_LIBRARY)

set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
#set(MKL_LIBRARIES ${MKL_INTERFACE_LIBRARY} ${MKL_SEQUENTIAL_LAYER_LIBRARY} ${MKL_CORE_LIBRARY})
set(MKL_LIBRARIES ${MKL_RT_LIBRARY})

#if (NOT (MKL_INCLUDE_DIR AND MKL_RT_LIBRARY) )
#
#    set(MKL_INCLUDE_DIRS "")
#    set(MKL_LIBRARIES "")
##    set(MKL_INTERFACE_LIBRARY "")
##    set(MKL_SEQUENTIAL_LAYER_LIBRARY "")
##    set(MKL_CORE_LIBRARY "")
#
#endif()

# Handle the QUIETLY and REQUIRED arguments and set MKL_FOUND to TRUE if
# all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
#FIND_PACKAGE_HANDLE_STANDARD_ARGS(MKL DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_DIRS MKL_INTERFACE_LIBRARY MKL_SEQUENTIAL_LAYER_LIBRARY MKL_CORE_LIBRARY)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_DIRS)

#MARK_AS_ADVANCED(MKL_INCLUDE_DIRS MKL_LIBRARIES MKL_INTERFACE_LIBRARY MKL_SEQUENTIAL_LAYER_LIBRARY MKL_CORE_LIBRARY)
mark_as_advanced(MKL_INCLUDE_DIRS MKL_LIBRARIES)
