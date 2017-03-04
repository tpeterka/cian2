project                     (cian)
cmake_minimum_required      (VERSION 2.8)

option                      (optimize       "Build cian with optimization"          ON)
option                      (debug          "Build cian with debugging on"          OFF)
option                      (bgq            "Build on BG/Q"                         OFF)
option                      (diy_thread     "Enable diy threading"                  OFF)

if                          (bgq)
    add_definitions         (-DBGQ)
endif                       (bgq)

# C++11
set                         (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# OSX flags
if                          (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions         (-DMAC_OSX)
    # --- following RPATH settings are for Sierra w/ Clang, hopefully they don't hurt other versions
    # ref: https://cmake.org/Wiki/CMake_RPATH_handling
    # use, i.e. don't skip, the full RPATH for the build tree
    set                     (CMAKE_SKIP_BUILD_RPATH            false)
    # when building, don't use the install RPATH already (but later on when installing)
    set                     (CMAKE_BUILD_WITH_INSTALL_RPATH    false)
    # set RPATH to install path
    set                     (CMAKE_INSTALL_RPATH               "${CMAKE_INSTALL_PREFIX}/lib")
    # add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    set                     (CMAKE_INSTALL_RPATH_USE_LINK_PATH true)
    # the RPATH to be used when installing, but only if it's not a system directory
    list                    (FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
                                                               "${CMAKE_INSTALL_PREFIX}/lib"
                                                               isSystemDir)
    if                      ("${isSystemDir}" STREQUAL         "-1")
        set                 (CMAKE_INSTALL_RPATH               "${CMAKE_INSTALL_PREFIX}/lib")
    endif()
endif                       (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

 # MPI
find_package                (MPI REQUIRED)
if                          (NOT bgq)
    set                     (libraries ${libraries} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif                       (NOT bgq)

# Threads
find_package                (Threads)
find_package                (OpenMP)
if                          (OPENMP_FOUND)
    set                     (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set                     (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else                        ()
    message                 ("OpenMP not found")
    add_definitions         (-DTESS_NO_OPENMP)
endif                       ()
if                          (NOT diy_thread)
    message                 ("Diy threading is disabled; setting diy threads will have no effect")
    add_definitions         (-DDIY_NO_THREADS)
endif                       (NOT diy_thread)

# DIY
find_path                   (DIY_INCLUDE_DIRS diy/types.h)

# Zlib
find_package                (ZLIB)

# Include dirs
set                         (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
include_directories         (${DIY_INCLUDE_DIRS}
                             ${HDF5_INCLUDE_DIRS}
                             ${MOAB_INCLUDE_DIRS}
                             ${ZLIB_INCLUDE_DIRS}
                             SYSTEM ${MPI_INCLUDE_PATH})

# Libraries
set                         (libraries
                             ${libraries}
                             ${HDF5_LIBRARY}
                             ${MOAB_LIBRARY}
                             ${MB_COUPLER_LIBRARY}
                             ${CMAKE_THREAD_LIBS_INIT})
if                          (HDF5_LIBRARY AND MOAB_LIBRARY AND MB_COUPLER_LIBRARY)
set                         (libraries
                             ${libraries}
                             ${ZLIB_LIBRARIES})
endif                       ()

# Subdirectories
if                          (HDF5_INCLUDE_DIRS AND MOAB_INCLUDE_DIRS AND DIY_INCLUDE_DIRS AND
                             HDF5_LIBRARY AND MOAB_LIBRARY AND MB_COUPLER_LIBRARY AND
                             ZLIB_LIBRARIES)
    message                 (STATUS "Building coupling proxy app")
    add_subdirectory        (coupling)
endif                       ()
if                          (DIY_INCLUDE_DIRS)
    message                 (STATUS "Building communication proxy apps")
    add_subdirectory        (communication)
else                        ()
    message                 (ERROR "Need diy2 in order to build any proxy apps")
endif                       ()
if                          (DIY_INCLUDE_DIRS)
    message                 (STATUS "Building I/O proxy app")
    add_subdirectory        (io)
else                        ()
    message                 (ERROR "Need diy2 in order to build any proxy apps")
endif                       ()
