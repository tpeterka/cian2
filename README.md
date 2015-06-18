# CESAR Integrated Analytics Proxy Applications

# Licensing

Cian is released in the [public domain](./COPYING).

# Installation

1. Install Dependencies

a. DIY

```
#!bash
git clone https://github.com/diatomic/diy2
```

b. HDF5 (only if using the coupling proxy app)

```
#!bash
wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.15-patch1.tar.gz
tar -xvf hdf5-1.8.15-patch1.tar.gz
cd hdf5-1.8.15-patch1
CC=mpicc FC=mpif90 ./configure –enable-parallel –enable-fortran –prefix=‘pwd‘
make
make install
```

c. Moab (only if using the coupling proxy app)

```
#!bash
git clone https://bitbucket.org/fathomteam/moab
autoreconf -fi

./configure \
LFLAGS=-static \
–with-mpi \
–with-hdf5=/path/to/hdf5/base/directory \
–prefix=‘pwd‘ \
–without-netcdf \
CXX=mpicxx \
CC=mpicc \
–without-pnetcdf \
–enable-mbcoupler

make -j 4
make install
```

2. Install cian

```
#!bash
git clone https://github.com/tpeterka/cian2
cd cian2
```

Configure using cmake:

```
#!bash
cmake /path/to/cian \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_C_COMPILER=mpicc \
-DDIY_INCLUDE_DIRS=/path/to/diy2/include \
-DHDF5_INCLUDE_DIRS=/path/to/hdf5/include \
-DHDF5_LIBRARY=/path/to/hdf5/lib/libhdf5.a \
-DMOAB_INCLUDE_DIRS=/path/to/moab/include \
-DMOAB_LIBRARY=/path/to/moab/lib/libMOAB.a \
-Ddebug=true \

make
```

# Execution

## Coupling

```
#!bash
cd coupling
```

Edit the run script COUPLING_TEST for the desired parameters as follows:

- min procs, max procs = minimum and maximum number of MPI processes
- mf = n (do not read mesh files from disk, create on the fly instead)
- st = source mesh type (h or t for hexahedral or tetrahedral)
- min ss, max ss = minimum and maximum source mesh size (regular grid vertices per side, eg. ss = 100)
- tt = target mesh type (h or t for hexahedral or tetrahedral)
- min ts = minimum target mesh size (regular grid vertices per side, eg. ts = 100)
- ni = number of iterations to (simulate) the convergence to a solution
- slab = whether or not source and target domain decomposition are homogeneous cubes (slab = 0) or heterogeneous slabs (slab = 1).

Notes: If min ss = max ss, the MPI process count will increase by a factor of 2X from min procs to max procs. This will be a strong scaling test. Otherwise, the source and target mesh sizes will double in each dimension (a total factor of 8X) from min to max, and the MPI process count will also increase by a factor of 8X. This will be a weak scaling test.

```
#!bash
./COUPLING_TEST
```

## Communication

### Neighbor exchange

```
#!bash
cd communication/neighbor
```

Edit the run script NEIGHBOR_TEST for the desired parameters.

```
#!bash
./NEIGHBOR_TEST
```

### Merge-reduction

```
#!bash
cd communication/merge
```

Edit the run script MERGE_TEST for the desired parameters.

```
#!bash
./MERGE_TEST
```

### Swap-reduction

```
#!bash
cd communication/swap
```

Edit the run script SWAP_TEST for the desired parameters.

```
#!bash
./SWAP_TEST
```

### All-to-all

TBD

### Sort

```
#!bash
cd communication/sort
```

Edit the run script SORT_TEST for the desired parameters.

```
#!bash
./SORT_TEST
```
