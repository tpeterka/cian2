project                     (cian)
cmake_minimum_required      (VERSION 2.8)

option                      (optimize       "Build cian with optimization"          ON)
option                      (debug          "Build cian with debugging on"          OFF)
option                      (bgq            "Build on BG/Q"                         OFF)
option                      (diy_thread     "Enable diy threading"                  OFF)

if                          (bgq)
    add_definitions         (-DBGQ)
endif                       (bgq)

# Debugging
if                          (debug)
    if                      (optimize)
            set             (cxx_flags      ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    else                    (optimize)
            set             (cxx_flags      ${CMAKE_CXX_FLAGS_DEBUG})
    endif                   (optimize)
    add_definitions         (-DDEBUG)
else                        (debug)
    if                      (optimize)
            set             (cxx_flags      ${CMAKE_CXX_FLAGS_RELEASE})
    else                    (optimize)
            set             (cxx_flags      ${CMAKE_CXX_FLAGS})
    endif                   (optimize)
endif                       (debug)
add_definitions             (${cxx_flags})

# OSX flags
if                          (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions	    (-DMAC_OSX)
endif                       (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

 # MPI
find_package                (MPI REQUIRED)
if                          (NOT bgq)
    set                     (libraries ${libraries} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif                       (NOT bgq)

# Threads
find_package		    (Threads)
find_package		    (OpenMP)
if 			    (OPENMP_FOUND)
  set                       (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set                       (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else			    ()
  message		    ("OpenMP not found")
  add_definitions	    (-DTESS_NO_OPENMP)
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
set			    (libraries
                             ${libraries}
                             ${HDF5_LIBRARY}
                             ${MOAB_LIBRARY}
                             ${MB_COUPLER_LIBRARY}
                             ${ZLIB_LIBRARIES}
                             ${CMAKE_THREAD_LIBS_INIT})

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
