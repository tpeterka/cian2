# Codesign Integrated Analytics Proxy Applications

A set of DIY and MOAB benchmarks for coupling, communication, and I/O

# Licensing

Cian is [public domain](./COPYING) software.

# Installation

Build dependencies

a. DIY

```
git clone https://github.com/diatomic/diy
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

A C++11 compiler is required.

```
git clone https://github.com/tpeterka/cian2

cmake /path/to/cian2 \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_INSTALL_PREFIX=/path/to/cian2/install \
-DDIY_INCLUDE_DIRS=/path/to/diy/include \
-DHDF5_INCLUDE_DIRS=/path/to/hdf5/include \
-DHDF5_LIBRARY=/path/to/hdf5/lib/libhdf5.a \
-DMOAB_INCLUDE_DIRS=/path/to/moab/include \
-DMOAB_LIBRARY=/path/to/moab/lib/libMOAB.a \
-DMB_COUPLER_LIBRARY=/path/to/moab/lib/libmbcoupler.a \
-Ddebug=true \

make
make install
```

# Execution

## Coupling

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

## Communication

### Neighbor exchange

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

### Merge-reduction

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

### Swap-reduction

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

### Partial composite swap-reduction (DIY swap-reduction vs. MPI Alltoallv)

```
cd path/to/cian2/install/communication/swap_vs_alltoallv
```

The reduction operator used in this swap reduction is a no-op copy of the
received data into the block in each round.

Edit the run script SWAP_TEST for the desired parameters:

- min procs, max procs = minimum and maximum number of MPI processes
- min rays, max rays = minimum and maximum number of rays in each block
  (process). Each ray has a given average number of elements along it to be
  reduced.
- nb = number of blocks per MPI process
- k = target k value (radix for k-ary reduction)
- avg_elems = average number of elements per ray. Each element is one
  floating-point value in this test. Even though the number of elements per ray is given as
  a constant, the proxy app is constructed to not assume a constant
  value. In other words, the number of elements in each ray is communicated before the
  actual elements are communicated.

```
./SWAP_TEST
```

### All-to-all

```
cd path/to/cian2/install/communication/alltoall
```

This alltoall test uses an exchange operator that matches MPI's all to all behavior. Data are exchanged between processes and transposed in units of the current number of elements in the test. (See any MPI textbook for the alltoall data exchange pattern.) Internally, the pattern is implemented with a k-ary swap-reduce, hence the need to specify the target k value. In this test, all the blocks are the same size, but DIY does not require this to be the case. Hence, the DIY perfomance is compared to both MPI all_to_all and all_to_allv. In the future, we may actually make different block sizes to confirm the all_to_allv comparison.

Edit the run script ALLTOALL_TEST for the desired parameters:

- min procs, max procs = minimum and maximum number of MPI processes
- min elems, max elems = minimum and maximum number of elements to reduce. Each element is one floating point value (* 4 bytes per float)
- nb = number of blocks per MPI process
- k = target k value (radix for k-ary reduction)

### Sort

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

## I/O

```
cd path/to/cian2/install/io
```
Tests DIY's block-based parallel I/O (writing or reading) implemented using MPI-IO.

Edit the run script IO_TEST for the desired parameters:

- min procs, max procs = minimum and maximum number of MPI processes
- min elems, max elems = minimum and maximum number of elements per block. Each element is one floating point value (4 bytes per float). In addition to the data values, each block also contains one integer (4 bytes) containing the block global ID and one long integer (size_t, 8 bytes) containing the number of elements.
- nb = number of blocks per MPI process
- op = w or r for writing or reading

Note: Use the same parameters (min_procs, max_procs, min_elems, max_elems, nb) for reading as for writing, and run the reading test (```op=r```) after the writing (```op=w```). The reason is that file names for each combination of parameters are automatically generated, and you want the reader to find the same files, named the same way, that the writer created. Files are not deleted automatically, meaning you can exceed disk storage quotas if you are not careful. Files are named *.out, where * is the run number corresponding to a combination of parameters (num_procs, num_elems).
