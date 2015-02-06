//
// tests to run:
//
// ranks per node
// different overall message sizes
// wraparound neighbors
//
//---------------------------------------------------------------------------
//
// testing DIY's neighborhood exchange performance
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

#include <diy/mpi.hpp>
#include <diy/master.hpp>
#include <diy/assigner.hpp>
#include <diy/serialization.hpp>
#include <diy/decomposition.hpp>

#include "opts.h"

using namespace std;

typedef  diy::ContinuousBounds       Bounds;
typedef  diy::RegularContinuousLink  RCLink;

// block
struct block_t
{
  vector< vector <int> > items; // received items
};

// function prototypes
void GetArgs(int argc, char **argv, int &min_procs, int &min_items,
	     int &max_items, int &num_ints, int &nb);
void PrintResults(double *enqueue_time, double *exchange_time, int min_procs, int max_procs,
		  int min_items, int max_items, int item_size, int num_item_iters,
                  int proc_factor, int item_factor);
void* create_block();
void destroy_block(void* b_);
void save_block(const void* b, diy::BinaryBuffer& bb);
void load_block(void* b, diy::BinaryBuffer& bb);
void enqueue(void* b_, const diy::Master::ProxyWithLink& cp, void*);
void parse(void* b_, const diy::Master::ProxyWithLink& cp, void*);

// add blocks to a master
struct AddBlock
{
  AddBlock(diy::Master& master_):
    master(master_)           {}

  void  operator()(int gid, const Bounds& core, const Bounds& bounds, const Bounds& domain,
                   const RCLink& link) const
  {
    block_t*      b = static_cast<block_t*>(create_block());
    RCLink*       l = new RCLink(link);
    diy::Master&  m = const_cast<diy::Master&>(master);
    m.add(gid, b, l);
  }

  diy::Master&  master;
};

// serialize a block
namespace diy
{
  template<>
  struct Serialization<block_t>
  {
    static void save(BinaryBuffer& bb, const block_t& d)
    {
      diy::save(bb, d.items);
    }

    static void load(BinaryBuffer& bb, block_t& d)
    {
      diy::load(bb, d.items);
    }
  };
}

// globals
int num_items; // current number of items in a round
int num_ints; // number of ints in one item
int proc_factor = 4; // factor for iterating over procs, eg 4X more each time
int item_factor = 4; // factor for iterating over items, eg 4X more each time

//----------------------------------------------------------------------------
//
// main
//
int main(int argc, char **argv)
{
  int dim = 3; // number of dimensions in the problem
  int tot_blocks; // total number of blocks
  int rank, groupsize; // MPI usual
  int min_procs; // minimum number of processes
  int max_procs; // maximum number of processes (groupsize of MPI_COMM_WORLD)
  int min_items, max_items; // min, max numbe of elements
  double t0; // temp time
  int nblocks; // my local number of blocks
  int num_item_iters; // number of item iterations per process

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &max_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  GetArgs(argc, argv, min_procs, min_items, max_items, num_ints, nblocks);
  int item_size = num_ints * sizeof(int);

  // data extents, unused
  Bounds domain;
  for(int i = 0; i < dim; i++)
  {
    domain.min[i] = 0.0;
    domain.max[i] = 1.0;
  }

  // max number of runs, based on proc_factor and item_factor = 2
  int num_runs = (int)((log2(max_procs / min_procs) + 1) *
    (log2(max_items / min_items) + 1));
  double enqueue_time[num_runs]; // enqueue time for each run
  double exchange_time[num_runs]; // exchange time for each run

  // iterate over processes
  int run = 0; // run number
  groupsize = min_procs;
  while (groupsize <= max_procs)
  {
    // form a new communicator
    MPI_Comm mpi_comm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_split(MPI_COMM_WORLD, (rank < groupsize), rank, &mpi_comm);
    if (rank >= groupsize)
    {
      MPI_Comm_free(&mpi_comm);
      groupsize *= proc_factor;
      continue;
    }
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &groupsize);

    // initialize DIY
    tot_blocks = nblocks * groupsize;
    int mem_blocks = -1; // everything in core for now
    int num_threads = 1; // because of timing
    diy::mpi::communicator    world(mpi_comm);
    diy::FileStorage          storage("./DIY.XXXXXX");
    diy::Master               master(world,
                                     &create_block,
                                     &destroy_block,
                                     mem_blocks,
                                     num_threads,
                                     &storage,
                                     &save_block,
                                     &load_block);
    diy::RoundRobinAssigner   assigner(world.size(), tot_blocks);
    AddBlock create(master);
    rank = world.rank();

    // decompose
    std::vector<int> my_gids;
    assigner.local_gids(diy_comm.rank(), my_gids);
    nblocks = my_gids.size();
    diy::decompose(dim, rank, domain, assigner, create);

    // iterate over number of items
    num_items = min_items;
    num_item_iters = 0; // number of item iterations per process
    while (num_items <= max_items)
    {
      MPI_Barrier(mpi_comm);
      t0 = MPI_Wtime();

      // enqueue the items
      master.foreach(&enqueue);

      MPI_Barrier(mpi_comm);
      enqueue_time[run] = MPI_Wtime() - t0;
      t0 = MPI_Wtime();

      // exchange neighbors
      master.exchange();

      //  parse received items
      master.foreach(&parse);

      MPI_Barrier(mpi_comm);
      exchange_time[run] = MPI_Wtime() - t0;

      num_items *= item_factor;
      run++;
      num_item_iters++;

    } // item iteration

    // cleanup
    groupsize *= proc_factor;
    MPI_Comm_free(&mpi_comm);

  } // proc iteration

  // print results
  MPI_Barrier(MPI_COMM_WORLD);
  fflush(stderr);
  if (rank == 0)
    PrintResults(enqueue_time, exchange_time, min_procs, max_procs, min_items,
                 max_items, item_size, num_item_iters, proc_factor, item_factor);

  MPI_Finalize();

  return 0;
}
//
// foreach block functions
//
void enqueue(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
  vector <int> vals(num_ints, 0);

  for (int i = 0; i < num_items; i++)
  {
    for (int j = 0; j < cp.link()->size(); j++)
      cp.enqueue(cp.link()->target(j), vals);
  }
}

void parse(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
  block_t* b = (block_t*)b_;
  std::vector<int> in; // gids of sources
  cp.incoming(in);

  b->items.resize(in.size() * num_items);

  // copy received items
  for (int i = 0; i < (int)in.size(); i++)
  {
    for(int j = 0; j < num_items; j++)
      cp.dequeue(in[i], b->items[i * num_items + j]);
  }
}
//----------------------------------------------------------------------------
//
// diy::Master callback functions
//
void* create_block()
{
  block_t* b = new block_t;
  return b;
}

void destroy_block(void* b_)
{
  block_t* b = (block_t*)b_;
  for (int i = 0; i < b->items.size(); i++)
    b->items[i].clear();
  b->items.clear();
  delete b;
}

void save_block(const void* b, diy::BinaryBuffer& bb)
{
}

void load_block(void* b, diy::BinaryBuffer& bb)
{
}
//----------------------------------------------------------------------------
//
// print results
//
// enqueue_time: enqueue time per run
// exchange_time: exchange time per run
// min_procs, max_procs: process range
// min_items, max_items: data range
// item_size: in bytes
// num_item_iters: number of item iterations per process
// proc_factor: factor change for process iteration
// item_factor: factor change for item iteration
//
void PrintResults(double *enqueue_time, double *exchange_time, int min_procs, int max_procs,
		  int min_items, int max_items, int item_size, int num_item_iters,
                  int proc_factor, int item_factor) {

  int item_iter = 0; // item iteration number
  int proc_iter = 0; // process iteration number

  fprintf(stderr, "----- Timing Results -----\n");

  // iterate over number of elements
  int num_items = min_items;
  while (num_items <= max_items) {

    fprintf(stderr, "\n# %d items * %d bytes / item = %d KB\n",
	    num_items, item_size, num_items * item_size / 1024);
    fprintf(stderr, "# procs \t enqueue_time (s) \t exchange_time (s)\n");

    // iterate over processes
    int groupsize = min_procs;
    proc_iter = 0;
    while (groupsize <= max_procs) {

      int i = proc_iter * num_item_iters + item_iter; // index into times
      fprintf(stderr, "%d \t\t %.3lf \t\t\t %.3lf\n",
	      groupsize, enqueue_time[i], exchange_time[i]);

      groupsize *= proc_factor;
      proc_iter++;

    } // proc iteration

    num_items *= item_factor;
    item_iter++;

  } // elem iteration

  fprintf(stderr, "\n--------------------------\n\n");

}
//----------------------------------------------------------------------------
//
// gets command line args
//
// argc, argv: usual
// min_procs: minimum number of processes (output)
// min_items: minimum number of items to exchange (output)
// max_items: maximum number of items to exchange (output)
// num_ints: number of ints per item (output)
// nb: number of local blocks
//
void GetArgs(int argc, char **argv, int &min_procs,
	     int &min_items, int &max_items, int &num_ints, int &nb) {

  using namespace opts;
  Options ops(argc, argv);
  int max_procs;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &max_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (ops >> Present('h', "help", "show help") ||
      !(ops >> PosOption(min_procs)
        >> PosOption(min_items)
        >> PosOption(max_items)
        >> PosOption(num_ints)
        >> PosOption(nb)))
  {
    if (rank == 0)
      fprintf(stderr, "Usage: %s min_procs min_items max_items num_ints nb\n", argv[0]);
    exit(1);
  }

  if (rank == 0) {
    fprintf(stderr, "min_procs = %d max_procs = %d "
	    "min_items = %d max_items = %d num_num_ints = %d nb = %d\n",
	    min_procs, max_procs, min_items, max_items, num_ints, nb);
  }

}
//----------------------------------------------------------------------------
