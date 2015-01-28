project                     (cian)
cmake_minimum_required      (VERSION 2.8)

option                      (optimize       "Build cian with optimization"          ON)
option                      (debug          "Build cian with debugging on"          OFF)
option                      (threads        "Build cian with threading on"          OFF)
option                      (bgq            "Build on BG/Q"                         OFF)

if                          (bgq)
    add_definitions         (-DBGQ)
endif                       (bgq)

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
if                          (NOT bgq)
    set                     (mpi_libraries              ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif                       (NOT bgq)

# Threads
if                          (threads)
    find_package            (Threads)
    find_package            (OpenMP)
else                        (threads)
    add_definitions         (-DDIY_NO_THREADS)
endif                       (threads)


# DIY
find_path                   (DIY_INCLUDE_DIRS           diy/types.h)

set                         (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
include_directories         (${DIY_INCLUDE_DIRS}
                             SYSTEM ${MPI_INCLUDE_PATH}
                            )

add_executable              (neighbor neighbor.cpp)
add_executable              (merge    merge.cpp)
add_executable              (swap     swap.cpp)

if                          (NOT bgq)
    target_link_libraries   (neighbor                   ${mpi_libraries})
    target_link_libraries   (merge                      ${mpi_libraries})
    target_link_libraries   (swap                       ${mpi_libraries})
endif                       (NOT bgq)

if                          (threads)
    target_link_libraries   (neighbor                   ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries   (merge                      ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries   (swap                       ${CMAKE_THREAD_LIBS_INIT})
endif                       (threads)
