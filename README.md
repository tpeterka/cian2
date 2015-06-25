## CESAR Integrated Analytics Proxy Applications

## Licensing

Cian is [public domain](./COPYING) software.

## Installation

Build dependencies

a. DIY

```
git clone https://github.com/diatomic/diy2
```

b. HDF5 (only if using the coupling proxy app)

```
wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.15-patch1.tar.gz
tar -xvf hdf5-1.8.15-patch1.tar.gz
cd hdf5-1.8.15-patch1
CC=mpicc FC=mpif90 ./configure –enable-parallel –enable-fortran –prefix=‘pwd‘
make
make install
```

c. Moab (only if using the coupling proxy app)

```
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

Build cian

```
git clone https://github.com/tpeterka/cian2

cmake /path/to/cian2 \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_INSTALL_PREFIX=/path/to/cian2/install \
-DDIY_INCLUDE_DIRS=/path/to/diy2/include \
-DHDF5_INCLUDE_DIRS=/path/to/hdf5/include \
-DHDF5_LIBRARY=/path/to/hdf5/lib/libhdf5.a \
-DMOAB_INCLUDE_DIRS=/path/to/moab/include \
-DMOAB_LIBRARY=/path/to/moab/lib/libMOAB.a \
-DMB_COUPLER_LIBRARY=/path/to/moab/lib/libmbcoupler.a \
-Ddebug=true \

make
make install
```

## Execution

### Coupling

```
cd path/to/cian2/install/coupling
```

Edit the run script COUPLING_TEST for the desired parameters:

- min procs, max procs = minimum and maximum number of MPI processes
- st = source mesh type (h or t for hexahedral or tetrahedral)
- min ss, max ss = minimum and maximum source mesh size (regular grid vertices per side, eg. ss = 100)
- tt = target mesh type (h or t for hexahedral or tetrahedral)
- min ts = minimum target mesh size (regular grid vertices per side, eg. ts = 100)
- ni = number of iterations to (simulate) the convergence to a solution
- slab = whether or not source and target domain decomposition are homogeneous cubes (slab = 0) or heterogeneous slabs (slab = 1).

Notes: If min ss = max ss, the MPI process count will increase by a factor of 2X from min procs to max procs. This will be a strong scaling test. Otherwise, the source and target mesh sizes will double in each dimension (a total factor of 8X) from min to max, and the MPI process count will also increase by a factor of 8X. This will be a weak scaling test.

```
./COUPLING_TEST
```

### Communication

- Neighbor exchange

```
cd path/to/cian2/install/communication/neighbor
```

Edit the run script NEIGHBOR_TEST for the desired parameters:

- min procs, max procs = minimum and maximum number of MPI processes
- min items, max items = minimum and maximum number of items to exchange
- item size = number of integers in one item (* 4 bytes per int)
- nb = number of blocks per MPI process

```
./NEIGHBOR_TEST
```

- Merge-reduction

```
cd path/to/cian2/install/communication/merge
```

The reduction operator used in this merge reduction is the noncommutative ``over'' operator used in image composition. For every pair of four elements (e.g., the RGBA channels of a pixel), the first three elements are modulated by the value of the fourth element and added in a predetermined order.

Edit the run script MERGE_TEST for the desired parameters:

- min procs, max procs = minimum and maximum number of MPI processes
- min elems, max elems = minimum and maximum number of elements to reduce. Each element is one floating point value (* 4 bytes per float)
- nb = number of blocks per MPI process
- k = target k value (radix for k-ary reduction)
- op = the reduction operator can be 0 or 1 for no-op or image composition, respectively

```
./MERGE_TEST
```

- Swap-reduction

```
cd path/to/cian2/install/communication/swap
```

The reduction operator used in this swap reduction is the noncommutative ``over'' operator used in image composition. For every pair of four elements (e.g., the RGBA channels of a pixel), the first three elements are modulated by the value of the fourth element and added in a predetermined order.

Edit the run script SWAP_TEST for the desired parameters:

- min procs, max procs = minimum and maximum number of MPI processes
- min elems, max elems = minimum and maximum number of elements to reduce. Each element is one floating point value (* 4 bytes per float)
- nb = number of blocks per MPI process
- k = target k value (radix for k-ary reduction)
- op = the reduction operator can be 0 or 1 for no-op or image composition, respectively

```
./SWAP_TEST
```

- All-to-all

TBD

- Sort

```
cd path/to/cian2/install/communication/sort
```

Edit the run script SORT_TEST for the desired parameters:

- min procs, max procs = minimum and maximum number of MPI processes
- min elems, max elems = minimum and maximum number of elements to sort. Each element is one integer (* 4 bytes per int) randomly assigned in the range minimum to maximum values of a signed integer
- nb = number of blocks per MPI process
- k = target k value (radix for k-ary reduction)
- ns = number of samples per block for the sample sort
- h = number of histogram bins per block for the histogram sort

```
./SORT_TEST
```
