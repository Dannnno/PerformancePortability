cmake_minimum_required (VERSION 2.6)
project (PerformancePortability)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

ADD_DEFINITIONS(-DOMP_ENABLED)
ADD_DEFINITIONS(-Wall)

if (${MSVC})
    ADD_DEFINITIONS(/arch:SSE4)
    ADD_DEFINITIONS(/W4)
    ADD_DEFINITIONS(/openmp)
elseif(${CMAKE_COMPILER_IS_GNUCXX})
    ADD_DEFINITIONS(-msse4)
    ADD_DEFINITIONS(-mavx)
    ADD_DEFINITIONS(-funsafe-math-optimizations)
    ADD_DEFINITIONS(-fabi-version=6)
    ADD_DEFINITIONS(-fopenmp)
    ADD_DEFINITIONS(-Werror)
    ADD_DEFINITIONS(-Wextra)
    ADD_DEFINITIONS(-pedantic)
    ADD_DEFINITIONS(-std=c++11)
endif()

file (GLOB MATRIX_SOURCES "*.h" "*.cc")
file (GLOB AgnerFog "vectorclass/*.h" "vectorclass/*.cpp")
source_group("matrix" FILES MATRIX_SOURCES)
source_group("vectorclass" FILES AgnerFog)

include_directories(. ./vectorclass)

add_executable(matrix run_benchmark.cc matrix.h aligned_allocator.h)

