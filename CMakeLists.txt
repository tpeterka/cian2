project                     (cian)
cmake_minimum_required      (VERSION 2.8)

option                      (optimize       "Build cian with optimization"          ON)
option                      (debug          "Build cian with debugging on"          OFF)

# Debugging
if                          (debug)
    if                      (optimize)
            set             (cxx_flags                  ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    else                    (optimize)
            set             (cxx_flags                  ${CMAKE_CXX_FLAGS_DEBUG})
    endif                   (optimize)
    add_definitions         (-DDEBUG)
else                        (debug)
    if                      (optimize)
            set             (cxx_flags                  ${CMAKE_CXX_FLAGS_RELEASE})
    else                    (optimize)
            set             (cxx_flags                  ${CMAKE_CXX_FLAGS})
    endif                   (optimize)
endif                       (debug)
add_definitions             (${cxx_flags})

# MPI
find_package                (MPI REQUIRED)
set                         (mpi_libraries              ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

# Threads
find_package                (Threads)
find_package                (OpenMP)

# DIY
find_path                   (DIY_INCLUDE_DIRS       diy/types.h)

set                         (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
include_directories         (${DIY_INCLUDE_DIRS}
                             SYSTEM ${MPI_INCLUDE_PATH}
                            )

add_executable              (neighbor neighbor.cpp) # TODO: later add reduce.cpp
target_link_libraries       (neighbor   ${mpi_libraries})
