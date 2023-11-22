cmake_minimum_required(VERSION 3.23)
project(mckit_geometry C)

set(CMAKE_C_STANDARD 11)

find_program(python python)
message(STATUS "python: ${python}")

execute_process(
    COMMAND ${python} -c "import sys; print(sys.prefix)"
    OUTPUT_VARIABLE sys_prefix
)
message(STATUS "sp: ${sys_prefix}")
SET(ENV{VIRTUAL_ENV} "${sys_prefix}")

find_package(Python 3.11 COMPONENTS Interpreter Development NumPy)

if( !PythonFound)
    message(FATAL_ERROR "Cannot find python environment")
else ()
    message(STATUS "pyinc: ${Python_INCLUDE_DIRS}")
    message(STATUS "numpy: ${Python_NumPy_INCLUDE_DIRS}")
endif()

# include_directories(. "${Python_INCLUDE_DIRS}" "${Python_NumPy_INCLUDE_DIRS}")

add_library(mckit_geometry
    box.c
    box.h
    box_doc.h
    common.h
    geometrymodule.c
    rbtree.c
    rbtree.h
    shape.c
    shape.h
    surf_doc.h
    surface.c
    surface.h
)
