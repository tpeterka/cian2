//
// testing DIY's reduction performance and comparing to MPI
//
// Tom Peterka
// Argonne National Laboratory
// 9700 S. Cass Ave.
// Argonne, IL 60439
// tpeterka@mcs.anl.gov
//
//--------------------------------------------------------------------------
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
#include <vector>
#include <algorithm>
#include <assert.h>

#include <diy/master.hpp>
#include <diy/reduce.hpp>
#include <diy/partners/swap.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>

#include "opts.h"

using namespace std;

typedef  diy::ContinuousBounds       Bounds;
typedef  diy::RegularContinuousLink  RCLink;

// function prototypes
void GetArgs(int argc, char **argv, int &min_procs, int &min_elems,
	     int &max_elems, int &nb, int &target_k, bool &op);
void MpiReduceScatter(double *reduce_scatter_time, int run, float *in_data, MPI_Comm comm,
                      int num_elems, bool op);
void DiySwap(double *swap_time, int run, int k, MPI_Comm comm, int dim, int totblocks,
             bool contiguous, diy::Master& master, diy::ContiguousAssigner& assigner, bool op);
void PrintResults(double *reduce_scatter_time, double *swap_time, int min_procs,
		  int max_procs, int min_elems, int max_elems);
void ComputeSwap(void* b_, const diy::ReduceProxy& rp, const diy::RegularSwapPartners&);
void NoopSwap(void* b_, const diy::ReduceProxy& rp, const diy::RegularSwapPartners&);
void Over(void *in, void *inout, int *len, MPI_Datatype*);
void Noop(void*, void*, int*, MPI_Datatype*) {}
void ResetBlock(void* b_, const diy::Master::ProxyWithLink& cp, void*);
bool compare(pair<int, int> u, pair<int, int> t);

// block
struct Block
{
  Block()                                                     {}
  static void*    create()                                    { return new Block; }
  static void     destroy(void* b)                            { delete static_cast<Block*>(b); }
  static void     save(const void* b, diy::BinaryBuffer& bb)
    { diy::save(bb, *static_cast<const Block*>(b)); }
  static void     load(void* b, diy::BinaryBuffer& bb)
    { diy::load(bb, *static_cast<Block*>(b)); }
  void generate_data(size_t n, int tot_b)
  {
    data.resize(n);
    for (size_t i = 0; i < n / 4; ++i)
    {
      data[4 * i    ] = gid * n / 4 + i;
      data[4 * i + 1] = gid * n / 4 + i;
      data[4 * i + 2] = gid * n / 4 + i;
      data[4 * i + 3] = gid / (tot_b - 1);
    }
  }

  vector<float> data;
  int gid;
  int sub_start; // starting index of subset of the total data that this block owns
  int sub_size;  // number of elements in the subset of the total data that this block owns
};
//
// add blocks to a master
//
struct AddBlock
{
  AddBlock(diy::Master& master_, size_t num_elems_, int tot_blocks_):
    master(master_), num_elems(num_elems_), tot_blocks(tot_blocks_) {}

  void operator()(int gid, const Bounds& core, const Bounds& bounds, const Bounds& domain,
                   const RCLink& link) const
  {
    Block*        b = new Block();
    RCLink*       l = new RCLink(link);
    diy::Master&  m = const_cast<diy::Master&>(master);
    m.add(gid, b, l);
    b->gid = gid;
    b->generate_data(num_elems, tot_blocks);
    b->sub_start = 0;
    b->sub_size = num_elems;
  }

  diy::Master&  master;
  size_t num_elems;
  int tot_blocks;
};
//
// reset the size and data values in a block
// args[0]: num_elems
// args[1]: tot_blocks
//
void ResetBlock(void* b_, const diy::Master::ProxyWithLink& cp, void* args)
{
    Block* b   = static_cast<Block*>(b_);
    int num_elems = *(int*)args;
    int tot_blocks = *((int*)args + 1);
    b->generate_data(num_elems, tot_blocks);
}
//
// prints data values in a block (debugging)
//
void PrintBlock(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
  Block* b   = static_cast<Block*>(b_);
  for (int i = b->sub_start; i < b->sub_start + b->sub_size; i++)
    fprintf(stderr, "diy reduced data[%d] = %.1f\n", i, b->data[i]);
}
//
// main
//
int main(int argc, char **argv)
{
  int dim = 3;              // number of dimensions in the problem
  int nblocks;              // local number of blocks
  int tot_blocks;           // total number of blocks
  int target_k;             // target k-value
  int min_elems, max_elems; // min, max number of elements per block
  int num_elems;            // current number of data elements per block
  int rank, groupsize;      // MPI usual
  int min_procs;            // minimum number of processes
  int max_procs;            // maximum number of processes (groupsize of MPI_COMM_WORLD)
  bool op;                  // actual operator or no-op

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &max_procs);

  GetArgs(argc, argv, min_procs, min_elems, max_elems, nblocks, target_k, op);

  // data extents, unused
  Bounds domain;
  for(int i = 0; i < dim; i++)
  {
    domain.min[i] = 0.0;
    domain.max[i] = 1.0;
  }

  int num_runs = (int)((log2(max_procs / min_procs) + 1) *
    (log2(max_elems / min_elems) + 1));

  // timing
  double reduce_scatter_time[num_runs];
  double swap_time[num_runs];

  // data for MPI reduce, only for one local block
  float *in_data = new float[max_elems];

  // iterate over processes
  int run = 0; // run number
  groupsize = min_procs;
  while (groupsize <= max_procs)
  {
    // form a new communicator
    MPI_Comm comm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_split(MPI_COMM_WORLD, (rank < groupsize), rank, &comm);
    if (rank >= groupsize)
    {
      MPI_Comm_free(&comm);
      groupsize *= 2;
      continue;
    }

    // initialize DIY
    tot_blocks = nblocks * groupsize;
    int mem_blocks = -1; // everything in core for now
    int num_threads = 1; // needed in order to do timing
    diy::mpi::communicator    world(comm);
    diy::FileStorage          storage("./DIY.XXXXXX");
    diy::Communicator         diy_comm(world);
    diy::Master               master(diy_comm,
                                     &Block::create,
                                     &Block::destroy,
                                     mem_blocks,
                                     num_threads,
                                     &storage,
                                     &Block::save,
                                     &Block::load);
    diy::ContiguousAssigner   assigner(world.size(), tot_blocks);
    AddBlock                  create(master, min_elems, tot_blocks);
    diy::decompose(dim, world.rank(), domain, assigner, create);

    // iterate over number of elements
    num_elems = min_elems;
    while (num_elems <= max_elems)
    {
      // MPI reduce-scatter, only for one block per process
      if (tot_blocks == groupsize)
	MpiReduceScatter(reduce_scatter_time, run, in_data, comm, num_elems, op);

      // DIY swap
      DiySwap(swap_time, run, target_k, comm, dim, tot_blocks, true, master, assigner, op);

      // debug
//       master.foreach(PrintBlock, &tot_blocks);

      // re-initialize input data (in-place DIY swap disturbed it)
      int args[2];
      args[0] = num_elems;
      args[1] = tot_blocks;
      master.foreach(ResetBlock, args);

      num_elems *= 2; // double the number of elements every time
      run++;

    } // elem iteration

    groupsize *= 2; // double the number of processes every time
    MPI_Comm_free(&comm);

  } // proc iteration

  // print results
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  fflush(stderr);
  if (rank == 0)
    PrintResults(reduce_scatter_time, swap_time, min_procs, max_procs, min_elems, max_elems);

  // cleanup
  delete[] in_data;
  MPI_Finalize();

  return 0;
}
//
// MPI reduce scatter
//
// reduce_scatter_time: time (output)
// run: run number
// in_data: input data
// comm: current communicator
// num_elems: current number of elements
// op: run actual op or noop
//
void MpiReduceScatter(double *reduce_scatter_time, int run, float *in_data, MPI_Comm comm,
                      int num_elems, bool op)
{
  // init
  MPI_Op op_fun;                       // custom operator
  if (op)
    MPI_Op_create(&Over, 0, &op_fun);  // noncommutative
  else
    MPI_Op_create(&Noop, 0, &op_fun);  // noncommutative, even if it doesn't do anything
  float *reduce_scatter_data = new float[num_elems];
  int rank;
  int groupsize;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &groupsize);
  int counts[groupsize];
  int size = num_elems  / groupsize;
  for (int i = 0; i < groupsize; i++)
    counts[i] = size;
  counts[groupsize - 1] = num_elems - (groupsize - 1) * size; // remainder
  for (int i = 0; i < num_elems / 4; i++) // init input data
  {
    in_data[4 * i    ] = rank * num_elems / 4 + i;
    in_data[4 * i + 1] = rank * num_elems / 4 + i;
    in_data[4 * i + 2] = rank * num_elems / 4 + i;
    in_data[4 * i + 3] = rank / (groupsize - 1);
  }

  // reduce
  MPI_Barrier(comm);
  double t0 = MPI_Wtime();
  MPI_Reduce_scatter((void *)in_data, (void *)reduce_scatter_data, counts, MPI_FLOAT, op_fun, comm);
  MPI_Barrier(comm);
  reduce_scatter_time[run] = MPI_Wtime() - t0;

  // debug: print the reduce-scattered data
//   for (int i = 0; i < counts[rank]; i++)
//     fprintf(stderr, "reduce_scatter_data[%d] = %.1f\n", i, reduce_scatter_data[i]);

  // cleanup
  delete[] reduce_scatter_data;
  MPI_Op_free(&op_fun);
}
//
// DIY swap
//
// swap_time: time (output)
// run: run number
// k: desired k value
// comm: MPI communicator
// dim: dimensionality of decompostion
// totblocks: total number of blocks
// contiguous: use contiguous partners
// master, assigner: diy usual
// op: run actual op or noop
//
void DiySwap(double *swap_time, int run, int k, MPI_Comm comm, int dim, int totblocks,
             bool contiguous, diy::Master& master, diy::ContiguousAssigner& assigner, bool op)
{
  MPI_Barrier(comm);
  double t0 = MPI_Wtime();

  diy::RegularSwapPartners  partners(dim, totblocks, k, contiguous);
  if (op)
    diy::reduce(master, assigner, partners, ComputeSwap);
  else
    diy::reduce(master, assigner, partners, NoopSwap);

  MPI_Barrier(comm);
  swap_time[run] = MPI_Wtime() - t0;
}
//
// print results
//
// reduce_scatter_time, swap_time: times
// min_procs, max_procs: process range
// min_elems, max_elems: data range
//
void PrintResults(double *reduce_scatter_time, double *swap_time, int min_procs,
		  int max_procs, int min_elems, int max_elems)
{
  int elem_iter = 0;                                            // element iteration number
  int num_elem_iters = (int)(log2(max_elems / min_elems) + 1);  // number of element iterations
  int proc_iter = 0;                                            // process iteration number

  fprintf(stderr, "----- Timing Results -----\n");

  // iterate over number of elements
  int num_elems = min_elems;
  while (num_elems <= max_elems)
  {
    fprintf(stderr, "\n# num_elemnts = %d   size @ 4 bytes / element = %d KB\n",
	    num_elems, num_elems * 4 / 1024);
    fprintf(stderr, "# procs \t red_scat_time \t swap_time\n");

    // iterate over processes
    int groupsize = min_procs;
    proc_iter = 0;
    while (groupsize <= max_procs)
    {
      int i = proc_iter * num_elem_iters + elem_iter; // index into times
      fprintf(stderr, "%d \t\t %.3lf \t\t %.3lf\n",
	      groupsize, reduce_scatter_time[i], swap_time[i]);

      groupsize *= 2; // double the number of processes every time
      proc_iter++;
    } // proc iteration

    num_elems *= 2; // double the number of elements every time
    elem_iter++;
  } // elem iteration

  fprintf(stderr, "\n--------------------------\n\n");
}
//
// Swap operator for DIY swap
// performs the "over" operator for image compositing
// ordering of the over operator is by gid
//
void ComputeSwap(void* b_, const diy::ReduceProxy& rp, const diy::RegularSwapPartners& partners)
{
  Block* b = static_cast<Block*>(b_);
  int sub_start = 0;                           // subset starting index
  int sub_size = b->data.size();               // subset size

  // find my position in the link
  int mypos;
  for (unsigned i = 0; i < rp.in_link().size(); ++i)
  {
    if (rp.in_link().target(i).gid == rp.gid())
      mypos = i;
  }

  // dequeue and reduce
  int k = rp.in_link().size();
  for (unsigned i = 0; i < k; ++i)
  {
    if (rp.in_link().target(i).gid == rp.gid())
      continue;

    // allocate receive buffer to correct subsize and then dequeue
    if (i == k - 1) // last subset may be different size
      sub_size = b->sub_size - (i * b->sub_size / k);
    else
      sub_size = b->sub_size / k;
    std::vector< float > in(sub_size);
    rp.dequeue(rp.in_link().target(i).gid, &in[0], sub_size);
//     fprintf(stderr, "[%d:%d] Received %d values from [%d]\n",
//             rp.gid(), rp.round(), (int)in_vals[i].size(), rp.in_link().target(i).gid);

    // compute my subset indices for the result of the merge
    b->sub_start += (mypos * b->sub_size / k);
    if (mypos == k - 1) // last subset may be different size
      b->sub_size = b->sub_size - (mypos * b->sub_size / k);
    else
      b->sub_size = b->sub_size / k;

    // NB: assumes that all items are same size, b->sub_size
    // TODO: figure out what to do when they are not, eg, when last item has extra values
    for (int j = 0; j < b->sub_size / 4; j++)
    {
      int s = b->sub_start;
      b->data[s + j * 4    ] =
        (1.0f - in[j * 4 + 3]) * b->data[s + j * 4    ] + in[j * 4    ];
      b->data[s + j * 4 + 1] =
        (1.0f - in[j * 4 + 3]) * b->data[s + j * 4 + 1] + in[j * 4 + 1];
      b->data[s + j * 4 + 2] =
        (1.0f - in[j * 4 + 3]) * b->data[s + j * 4 + 2] + in[j * 4 + 2];
      b->data[s + j * 4 + 3] =
        (1.0f - in[j * 4 + 3]) * b->data[s + j * 4 + 3] + in[j * 4 + 3];
    }
  }

  if (!rp.out_link().size())
    return;

  // enqueue
  k = rp.out_link().size();
  for (unsigned i = 0; i < k; i++)
  {
    // temp versions of sub_start and sub_size are for sending
    // final versions stored in the block are updated upon receiving (above)
    sub_start = b->sub_start + (i * b->sub_size / k);
    if (i == k - 1) // last subset may be different size
      sub_size = b->sub_size - (i * b->sub_size / k);
    else
      sub_size = b->sub_size / k;
    rp.enqueue(rp.out_link().target(i), &b->data[sub_start], sub_size);
//     fprintf(stderr, "[%d:%d] Sent %lu values starting at %d to [%d]\n",
//             rp.gid(), rp.round(), send_buf.size(), sub_start, rp.out_link().target(i).gid);
  }
}
//
// Noop for DIY swap
//
void NoopSwap(void* b_, const diy::ReduceProxy& rp, const diy::RegularSwapPartners& partners)
{
  Block*    b        = static_cast<Block*>(b_);

  // dequeue all incoming neighbors
  // need all incoming data up front so that it can be composited in a particular order
  // in this test, the order is increasing gid
  std::vector< vector< float > > in_vals;
  in_vals.resize(rp.in_link().size());
  int mypos; // my position in the link
  for (unsigned i = 0; i < rp.in_link().size(); ++i)
    rp.dequeue(rp.in_link().target(i).gid, in_vals[i]);

  if (!rp.out_link().size())
    return;

  // enqueue
  int k = rp.out_link().size();
  vector <float> send_buf; // subset of b's data to be sent, TODO: avoidable?
  for (unsigned i = 0; i < k; i++)
  {
    // temp versions of sub_start and sub_size are for sending
    // final versions stored in the block are updated upon receiving (above)
    int sub_start = b->sub_start + (i * b->sub_size / k);
    int sub_size;
    if (i == k - 1) // last subset may be different size
      sub_size = b->sub_size - (i * b->sub_size / k);
    else
      sub_size = b->sub_size / k;
    send_buf.resize(sub_size);
    for (int j = 0; j < sub_size; j++)    // TODO: any way to avoid this copy?
      send_buf[j] = b->data[b->sub_start + j];
    rp.enqueue(rp.out_link().target(i), send_buf);
  }
}
//
// comparison function for searching vector of (gid, pos) pairs
//
bool compare(pair<int, int> u, pair <int, int> t)
{
  return(u.first < t.first);
}
//
// performs in over inout
// inout is the result
// both in and inout have same size in pixels
//
void Over(void *in, void *inout, int *len, MPI_Datatype *type)
{
  for (int i = 0; i < *len / 4; i++)
  {
    ((float *)inout)[i * 4] =
      (1.0f - ((float *)in)[i * 4 + 3]) * ((float *)inout)[i * 4] +
      ((float *)in)[i * 4];

    ((float *)inout)[i * 4 + 1] =
      (1.0f - ((float *)in)[i * 4 + 3]) * ((float *)inout)[i * 4 + 1] +
      ((float *)in)[i * 4 + 1];

    ((float *)inout)[i * 4 + 2] =
      (1.0f - ((float *)in)[i * 4 + 3]) * ((float *)inout)[i * 4 + 2] +
      ((float *)in)[i * 4 + 2];

    ((float *)inout)[i * 4 + 3] =
      (1.0f - ((float *)in)[i * 4 + 3]) * ((float *)inout)[i * 4 + 3] +
      ((float *)in)[i * 4 + 3];
  }
}
//
// gets command line args
//
// argc, argv: usual
// min_procs: minimum number of processes (output)
// min_elems: minimum number of elements to reduce(output)
// max_elems: maximum number of elements to reduce (output)
// nb: number of blocks per process (output)
// target_k: target k-value (output)
// op: whether to run to operator or no op
//
void GetArgs(int argc, char **argv, int &min_procs,
	     int &min_elems, int &max_elems, int &nb, int &target_k, bool &op)
{
  using namespace opts;
  Options ops(argc, argv);
  int max_procs;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &max_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (ops >> Present('h', "help", "show help") ||
      !(ops >> PosOption(min_procs)
        >> PosOption(min_elems)
        >> PosOption(max_elems)
        >> PosOption(nb)
        >> PosOption(target_k)
        >> PosOption(op)))
  {
    if (rank == 0)
      fprintf(stderr, "Usage: %s min_procs min_elems max_elems nb target_k op\n", argv[0]);
    exit(1);
  }

  // check there is at least four elements (eg., one pixel) per block
  assert(min_elems >= 4 *nb * max_procs); // at least one element per block

  if (rank == 0)
    fprintf(stderr, "min_procs = %d min_elems = %d max_elems = %d nb = %d "
	    "target_k = %d\n", min_procs, min_elems, max_elems, nb, target_k);
}

