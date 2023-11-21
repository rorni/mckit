cmake_minimum_required(VERSION 3.15...3.26)

# mckit_version=0.9.0 # TODO: retrieve from poetry
# project(mckit VERSION ${mckit_version} LANGUAGES C CXX)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C)


add_subdirectory(3rd-party/nlopt)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

FILE(GLOB geometry_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/mckit/src/*.c)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/mckit/src)

Python_add_library(geometry MODULE ${geometry_sources})

install(TARGETS geometry DESTINATION src/mckit)
