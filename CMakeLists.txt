cmake_minimum_required(VERSION 3.19...3.27)

# mckit_version=0.9.0 # TODO: retrieve from poetry
# project(mckit VERSION ${mckit_version} LANGUAGES C CXX)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION}
        DESCRIPTION "mckit geometry extension"
        LANGUAGES C)  # TODO: upgrade to CXX

#https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
#include(CheckLanguage)
#check_language(CUDA)
#if(CMAKE_CUDA_COMPILER)
#    enable_language(CUDA)
#endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(CMakePrintHelpers)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)


  # Set CMAKE_BUILD_TYPE to Release by default.
  # Must be done before calling project()
  set(default_build_type "Release")
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
  endif()

  if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Building in-source is not supported! Create a build dir and remove ${CMAKE_SOURCE_DIR}/CMakeCache.txt")
  endif()

  find_package(Git QUIET)

  if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
      execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                      RESULT_VARIABLE GIT_SUBMOD_RESULT)
      if(NOT GIT_SUBMOD_RESULT EQUAL "0")
          message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
      endif()
  endif()
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/nlopt/CMakeLists.txt")
      message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
  endif()
endif()

execute_process(
    COMMAND python -c "import sys; print(sys.prefix)"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE Python_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

cmake_print_variables(Python_ROOT)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
cmake_print_variables(
    Python_SOABI
    Python_SOSABI
)

add_subdirectory(extern)

list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/cmake)
#list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/cmake "${Python_ROOT}/lib/cmake" "${Python_ROOT}/Library/lib/cmake")
#cmake_print_variables(CMAKE_PREFIX_PATH)

set(MKLROOT ${Python_ROOT})
set($ENV{MKLROOT} ${MKLROOT})
set($ENV{MKL_ROOT} ${MKLROOT})
#set(MKL_ARCH intel64)
#set(MKL_THREADING intel_thread)
#set(MKL_INTERFACE lp64)
#find_package(MKL CONFIG REQUIRED)
include(FindMKL)
# ========================
# MKL
# https://gitlab.inria.fr/solverstack/test_fembem/-/blob/master/CMakeLists.txt
# ========================
# find_package(MKL REQUIRED)
# if(MKL_FOUND)
#   set(CMAKE_C_FLAGS "${MKL_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
#   set(CMAKE_CXX_FLAGS "${MKL_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
# endif()


set(geometry_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/mckit/src/)

set(geometry_sources
   ${geometry_dir}box.c
   ${geometry_dir}box.h
   ${geometry_dir}box_doc.h
   ${geometry_dir}geometrymodule.c
   ${geometry_dir}rbtree.c
   ${geometry_dir}rbtree.h
   ${geometry_dir}shape.c
   ${geometry_dir}shape.h
   ${geometry_dir}surface.c
   ${geometry_dir}surface.h
   ${geometry_dir}surf_doc.h
   ${geometry_dir}shape.h
)


Python_add_library(geometry MODULE ${geometry_sources})
target_include_directories(geometry
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mckit/src
        ${MKL_INCLUDE_DIR}
        ${Python_NumPy_INCLUDE_DIRS}
)
target_link_libraries(geometry PUBLIC nlopt ${MKL_LIBRARIES})
#cmake_print_properties(
#    TARGETS geometry
#    PROPERTIES SOURCES
#)
install(TARGETS geometry DESTINATION src/mckit)

include(FeatureSummary)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    feature_summary(WHAT ENABLED_FEATURES DISABLED_FEATURES PACKAGES_FOUND)
    feature_summary(FILENAME ${CMAKE_CURRENT_BINARY_DIR}/features.log WHAT ALL)
endif()
