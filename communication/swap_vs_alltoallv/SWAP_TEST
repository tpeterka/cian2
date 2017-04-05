#!/bin/bash

##PBS -A CSC033
#PBS -A CI-CCR000086
#PBS -N t
#PBS -j oe
#PBS -l walltime=0:10:00,size=12

#----------------------------------------------------------------------------
#
# mpi run script
#
# Tom Peterka
# Argonne National Laboratory
# 9700 S. Cass Ave.
# Argonne, IL 60439
# tpeterka@mcs.anl.gov
#
#----------------------------------------------------------------------------
ARCH=MAC_OSX
#ARCH=LINUX
#ARCH=BGQ
#ARCH=XC

# number of procs
min_procs=2
max_procs=8

# procs per node
ppn=4 # adjustable for BG/Q, allowed 1, 2, 4, 8, 16, 32, 64

# number of nodes
num_nodes=$[$max_procs / $ppn]
if [ $num_nodes -lt 1 ]; then
    num_nodes=1
fi

# executable
exe=./swap_vs_alltoallv

# number of rays to reduce
K=1024
M=1048576
min_rays=$[1 * 1 * $K]
max_rays=$[8 * 1 * $K]

# average number of elements per ray to reduce
avg_elems=24

# number of blocks per process
nb=1

# target k-value
k=2

# reduction factor (mimics compositing rf elements into one in each round)
# rf <= 1 signals do not reduce and do not copy until last round, the closest comparison to an
# alltoallv, with no reductions (memory copies) in the intermediate rounds
rf=2

#------
#
# program arguments
#
args="$min_procs $min_rays $max_rays $nb $k $avg_elems $rf"

#------
#
# run commands
#

if [ "$ARCH" = "MAC_OSX" ]; then

mpiexec -l -n $max_procs $exe $args

#dsymutil $exe ; mpiexec -l -n $max_procs xterm -e gdb -x gdb.run --args $exe $args

#dsymutil $exe ; mpiexec -n $max_procs valgrind -q $exe $args

#dsymutil $exe ; mpiexec -n $max_procs valgrind -q --leak-check=yes $exe $args

fi

if [ "$ARCH" = "LINUX" ]; then

#mpiexec -n $max_procs $exe $args

#mpiexec -n $max_procs xterm -e gdb -x gdb.run --args $exe $args

#mpiexec -n $max_procs valgrind -q $exe $args

mpiexec -n $max_procs valgrind -q --leak-check=yes $exe $args

fi

if [ "$ARCH" = "BGQ" ]; then

qsub -n $num_nodes --mode c$ppn -A SDAV -t 30 $exe $args

fi

if [ "$ARCH" = "XC" ]; then

aprun -n $max_procs $exe $args

fi
