cmake_minimum_required (VERSION 2.6)
project (PerformancePortability)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set (CMAKE_CXX_FLAGS "-Wall")

ADD_DEFINITIONS(-DOMP_ENABLED)

if (${MSVC})
    ADD_DEFINITIONS(/arch:SSE2)
    ADD_DEFINITIONS(/W4)
    ADD_DEFINITIONS(/openmp)
elseif(${CMAKE_COMPILER_IS_GNUCXX})
    ADD_DEFINITIONS(-msse2)
    ADD_DEFINITIONS(-Werror)
    ADD_DEFINITIONS(-Wextra)
    ADD_DEFINITIONS(-pedantic)
    ADD_DEFINITIONS(-std=c++11)
endif()

file (GLOB MATRIX_SOURCES "*.h" "*.cc")
file (GLOB AgnerFog "vectorclass/*.h" "vectorclass/*.cpp")
source_group("" FILES MATRIX_SOURCES)
source_group("" FILES AgnerFog)

include_directories(..)

add_executable(matrix run_benchmark.cc)

set_property(TARGET matrix PROPERTY FOLDER "executables")

install(TARGETS matrix RUNTIME DESTINATION ${PROJECT_BINARY_DIR}/bin)
