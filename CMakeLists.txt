cmake_minimum_required(VERSION 3.19...3.27)

# mckit_version=0.9.0 # TODO: retrieve from poetry
# project(mckit VERSION ${mckit_version} LANGUAGES C CXX)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION}
        DESCRIPTION "mckit geometry extension"
        LANGUAGES C)


add_subdirectory(3rd-party/nlopt)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

FILE(GLOB geometry_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/mckit/src/*.c)

Python_add_library(geometry MODULE ${geometry_sources})
target_include_directories(geometry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/mckit/src)
target_link_libaries(geometry PUBLIC nlopt)

install(TARGETS geometry DESTINATION src/mckit)
