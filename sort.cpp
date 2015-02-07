//
// testing DIY2's sort performance and comparing to DIY1
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
#include <limits>
#include <assert.h>
#include <limits>

#include <diy/master.hpp>
#include <diy/reduce.hpp>
#include <diy/partners/all-reduce.hpp>
#include <diy/partners/swap.hpp>
#include <diy/assigner.hpp>

#include "opts.h"
#include "psort.h"

using namespace std;

typedef  diy::ContinuousBounds       Bounds;
typedef  diy::RegularContinuousLink  RCLink;

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
  void generate_data(size_t n)
  {
    std::numeric_limits<int> lims;
    min = lims.min();
    max = lims.max();
    values.resize(n);
    srand(gid);
    for (size_t i = 0; i < n; ++i)
      values[i] = rand();
  }

  int                   min, max;   // min, max of values
  std::vector<int>      values;     // data values
  int                   gid;        // block gid
  int                   bins;       // number of bins in the histogram
  std::vector<size_t>   histogram;  // histogram used to sort values into child blocks
};

// add blocks to a master
struct AddBlock
{
  AddBlock(diy::Master& master_): master(master_)               {}

  void operator()(int gid, const Bounds& core, const Bounds& bounds, const Bounds& domain,
                   const RCLink& link) const
  {
    Block*        b = new Block();
    RCLink*       l = new RCLink(link);
    diy::Master&  m = const_cast<diy::Master&>(master);
    m.add(gid, b, l);
    b->gid = gid;
  }

  diy::Master&  master;
};

// reset the size and data values in a block
// args[0]: num_elems
// args[1]: bins
void ResetBlock(void* b_, const diy::Master::ProxyWithLink& cp, void* args)
{
  Block* b      = static_cast<Block*>(b_);
  int *a        = (int*)args;
  int num_elems = a[0];
  b->bins       = a[1];
  b->generate_data(num_elems);
}

void VerifyBlock(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
  Block* b   = static_cast<Block*>(b_);
  int act_min, act_max; // actual min and max of the values
  for (size_t i = 0; i < b->values.size(); ++i)
  {
    if (i == 0 || b->values[i] < act_min)
      act_min = b->values[i];
    if (i == 0 || b->values[i] > act_max)
      act_max = b->values[i];
    if (b->values[i] < b->min || b->values[i] > b->max)
      fprintf(stderr, "Warning: in gid %d %d outside of %d, %d\n",
              b->gid, b->values[i], b->min, b->max);
    if (i > 0 && b->values[i] < b->values[i - 1])
      fprintf(stderr, "Warning: gid %d is not sorted\n", b->gid);
//     fprintf(stderr, "diy2: gid %d sorted data[%lu] = %d\n", b->gid, i, b->values[i]);
  }
  fprintf(stderr, "diy2 sort: gid %d num_elems %lu (min, max) = (%d %d)\n",
          b->gid, b->values.size(), act_min, act_max);
}

// 1D sort partners:
//   these allow for k-ary reductions (as opposed to kd-trees,
//   which are fixed at k=2)
struct SortPartners
{
  // bool = are we in a swap (vs histogram) round
  // int  = round within that partner
  typedef       std::pair<bool, int>            RoundType;

  SortPartners(int nblocks, int k):
    histogram(1, nblocks, k),
    swap(1, nblocks, k, false)
    {
      for (unsigned i = 0; i < swap.rounds(); ++i)
      {
        // fill histogram rounds
        for (unsigned j = 0; j < histogram.rounds(); ++j)
        {
          rounds_.push_back(std::make_pair(false, j));
          if (j == histogram.rounds() / 2 - 1 - i)
            j += 2*i;
        }

        // fill swap round
        rounds_.push_back(std::make_pair(true, i));
      }
    }

  size_t        rounds() const                              { return rounds_.size(); }
  bool          swap_round(int round) const                 { return rounds_[round].first; }
  int           sub_round(int round) const                  { return rounds_[round].second; }

  inline bool   active(int round, int gid) const
    {
      if (round == rounds())
        return true;
      else if (swap_round(round))
        return swap.active(sub_round(round), gid);
      else
        return histogram.active(sub_round(round), gid);
    }

  inline void   incoming(int round, int gid, std::vector<int>& partners) const
    {
      if (round == rounds())
        swap.incoming(sub_round(round-1) + 1, gid, partners);
      else if (swap_round(round))
        histogram.incoming(histogram.rounds(), gid, partners);
      else
      {
        if (round > 0 && sub_round(round) == 0)
          swap.incoming(sub_round(round - 1) + 1, gid, partners);
        // jump through histogram rounds
        else if (round > 0 && sub_round(round - 1) != sub_round(round) - 1)
          histogram.incoming(sub_round(round - 1) + 1, gid, partners);
        else
          histogram.incoming(sub_round(round), gid, partners);
      }
    }

  inline void   outgoing(int round, int gid, std::vector<int>& partners) const
    {
      if (round == rounds())
        swap.outgoing(sub_round(round-1) + 1, gid, partners);
      else if (swap_round(round))
        swap.outgoing(sub_round(round), gid, partners);
      else
        histogram.outgoing(sub_round(round), gid, partners);
    }

  diy::RegularAllReducePartners     histogram;
  diy::RegularSwapPartners          swap;

  std::vector<RoundType>            rounds_;
};

// helper functions for the sort operator
void compute_local_histogram(void* b_, const diy::ReduceProxy& srp)
{
  Block* b = static_cast<Block*>(b_);

  // compute and enqueue local histogram
  b->histogram.clear();
  b->histogram.resize(b->bins);
  float width = ((float)b->max - (float)b->min) / b->bins;
  for (size_t i = 0; i < b->values.size(); ++i)
  {
    int x = b->values[i];
    int loc = ((float)x - b->min) / width;
    if (loc >= b->bins)
      loc = b->bins - 1;
    ++(b->histogram[loc]);
  }
  if (srp.out_link().target(0).gid != srp.gid())
    srp.enqueue(srp.out_link().target(0), b->histogram);
}

void add_histogram(void* b_, const diy::ReduceProxy& srp)
{
  Block* b = static_cast<Block*>(b_);

  // dequeue and add up the histograms
  for (unsigned i = 0; i < srp.in_link().size(); ++i)
  {
    int nbr_gid = srp.in_link().target(i).gid;
    if (nbr_gid != srp.gid())
    {
      std::vector<size_t> hist;
      srp.dequeue(nbr_gid, hist);
      for (size_t i = 0; i < hist.size(); ++i)
        b->histogram[i] += hist[i];
    }
  }
  if (srp.out_link().target(0).gid != srp.gid())
    srp.enqueue(srp.out_link().target(0), b->histogram);
}

void receive_histogram(void* b_, const diy::ReduceProxy& srp)
{
  Block* b = static_cast<Block*>(b_);

  if (srp.in_link().target(0).gid != srp.gid())
    srp.dequeue(srp.in_link().target(0).gid, b->histogram);
}

void forward_histogram(void* b_, const diy::ReduceProxy& srp)
{
  Block* b = static_cast<Block*>(b_);

  for (unsigned i = 0; i < srp.out_link().size(); ++i)
    if (srp.out_link().target(i).gid != srp.gid())
      srp.enqueue(srp.out_link().target(i), b->histogram);
}

void enqueue_exchange(void* b_, const diy::ReduceProxy& srp)
{
  Block* b = static_cast<Block*>(b_);

  int k = srp.out_link().size();

  // pick split points
  size_t total = 0;
  for (size_t i = 0; i < b->histogram.size(); ++i)
    total += b->histogram[i];

  std::vector<int> splits;
  splits.push_back(b->min);
  size_t cur = 0;
  float width = ((float)b->max - (float)b->min) / b->bins;
  for (size_t i = 0; i < b->histogram.size(); ++i)
  {
    if (cur + b->histogram[i] > total / k * splits.size())
      splits.push_back(b->min + width * i + width / 2);   // mid-point of the bin

    cur += b->histogram[i];

    if (splits.size() == k)
      break;
  }

  // subset and enqueue
  if (srp.out_link().size() == 0)        // final round; nothing needs to be sent
    return;

  std::vector< std::vector<int> > out_values(srp.out_link().size());
  for (size_t i = 0; i < b->values.size(); ++i)
  {
    int loc = std::upper_bound(splits.begin(), splits.end(), b->values[i]) - splits.begin() - 1;
    out_values[loc].push_back(b->values[i]);
  }
  int pos = -1;
  for (int i = 0; i < k; ++i)
  {
    if (srp.out_link().target(i).gid == srp.gid())
    {
      b->values.swap(out_values[i]);
      pos = i;
    }
    else
      srp.enqueue(srp.out_link().target(i), out_values[i]);
  }
  splits.push_back(b->max);
  int new_min = splits[pos];
  int new_max = splits[pos+1];
  b->min = new_min;
  b->max = new_max;
}

void dequeue_exchange(void* b_, const diy::ReduceProxy& srp)
{
  Block* b = static_cast<Block*>(b_);

  for (unsigned i = 0; i < srp.in_link().size(); ++i)
  {
    int nbr_gid = srp.in_link().target(i).gid;
    if (nbr_gid == srp.gid())
      continue;

    std::vector<int>    in_values;
    srp.dequeue(nbr_gid, in_values);
    for (size_t j = 0; j < in_values.size(); ++j)
    {
      if (in_values[j] < b->min)
      {
        std::cerr << "Warning: " << in_values[j] << " < min = " << b->min << std::endl;
        std::abort();
      }
      b->values.push_back(in_values[j]);
    }
  }
}

void sort_local(void* b_, const diy::ReduceProxy&)
{
  Block* b = static_cast<Block*>(b_);
  std::sort(b->values.begin(), b->values.end());
}

void sort_all(void* b_, const diy::ReduceProxy& srp, const SortPartners& partners)
{
  if (srp.round() == partners.rounds())
  {
    dequeue_exchange(b_, srp);
    sort_local(b_, srp);
  }
  else if (partners.swap_round(srp.round()))
  {
    receive_histogram(b_, srp);
    enqueue_exchange(b_, srp);
  } else if (partners.sub_round(srp.round()) == 0)
  {
    if (srp.round() > 0)
      dequeue_exchange(b_, srp);

    compute_local_histogram(b_, srp);
  } else if (partners.sub_round(srp.round()) < partners.histogram.rounds()/2)
    add_histogram(b_, srp);
  else
  {
    receive_histogram(b_, srp);
    forward_histogram(b_, srp);
  }
}

// DIY2 sort
//
// time: time (output)
// run: run number
// k: desired k value
// comm: MPI communicator
// totblocks: total number of blocks
// master, assigner: diy usual
void Diy2Sort(double *time, int run, int k, MPI_Comm comm, int totblocks,
              diy::Master& master, diy::ContiguousAssigner& assigner)
{
  MPI_Barrier(comm);
  double t0 = MPI_Wtime();


  SortPartners partners(totblocks, k);
  diy::reduce(master, assigner, partners, sort_all);

  MPI_Barrier(comm);
  time[run] = MPI_Wtime() - t0;
}

// DIY1 sort
int compare_int(const void* a, const void* b)
{
  if (*(int*)a < *(int*)b)
    return -1;
  else if (*(int*)a > *(int*)b)
    return 1;
  else
    return 0;
}
void Diy1Sort(double *time, int run, MPI_Comm comm, int num_elems)
{
  // init in_data
  // using malloc/free instead of new/delete because Diy1Sort is implemented in C
  int *in_data = (int*)malloc(num_elems * sizeof(int));
  std::numeric_limits<int> lims;
  int min = lims.min();
  int max = lims.max();
  int rank;
  MPI_Comm_rank(comm, &rank);
  srand(rank);
  for (size_t i = 0; i < num_elems; ++i)
    in_data[i] = rand();

  // sort
  MPI_Barrier(comm);
  double t0 = MPI_Wtime();

  // parallel sample sort
  pssort((void**)&in_data, &num_elems, sizeof(int), compare_int, comm);

  // parallel merge sort
//   pmsort((void**)&in_data, &num_elems, sizeof(int), compare_int, comm);

  MPI_Barrier(comm);
  time[run] = MPI_Wtime() - t0;

  // verify
//   for (size_t i = 0; i < num_elems; ++i)
//   {
//     if (i == 0 || in_data[i] < min)
//       min = in_data[i];
//     if (i == 0 || in_data[i] > max)
//       max = in_data[i];
//     if (i > 0 && in_data[i] < in_data[i - 1])
//       fprintf(stderr, "Warning: diy1 sort: rank %d is not sorted\n", rank);
// //     fprintf(stderr, "diy1: rank %d sorted data[%lu] = %d\n", rank, i, in_data[i]);
//   }
//   fprintf(stderr, "diy1 sort: rank %d num_elems %d (min, max) = (%d %d)\n",
//           rank, num_elems, min, max);

  // cleanup
  free(in_data);
}

// print results
//
// ssort_time, dsort_time: times
// min_procs, max_procs, proc_x: process range
// min_elems, max_elems, elem_x: data range
void PrintResults(double *ssort_time, double *dsort_time, int min_procs,
		  int max_procs, int proc_x, int min_elems, int max_elems, int elem_x)
{
  int elem_iter = 0;                                            // element iteration number
  int num_elem_iters = 0;                                       // number of element iterations
  int proc_iter;                                                // process iteration number

  for (int i = min_elems; i <= max_elems; i *= elem_x)
    num_elem_iters++;

  fprintf(stderr, "----- Timing Results -----\n");

  // iterate over number of elements
  int num_elems = min_elems;
  while (num_elems <= max_elems)
  {
    fprintf(stderr, "\n# num_elements = %d\n", num_elems);
    fprintf(stderr, "# procs \t diy1 time \t diy2 time\n");

    // iterate over processes
    int groupsize = min_procs;
    proc_iter = 0;
    while (groupsize <= max_procs)
    {
      int i = proc_iter * num_elem_iters + elem_iter; // index into times
      fprintf(stderr, "%d \t\t %.3lf \t\t %.3lf\n",
	      groupsize, ssort_time[i], dsort_time[i]);

      groupsize *= proc_x;
      proc_iter++;
    } // proc iteration

    num_elems *= elem_x;
    elem_iter++;
  } // elem iteration

  fprintf(stderr, "\n--------------------------\n\n");
}

// gets command line args
void GetArgs(int argc, char **argv, int &min_procs,
	     int &min_elems, int &max_elems, int &nb, int &target_k,int &hbins)
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
        >> PosOption(hbins)))
  {
    if (rank == 0)
      fprintf(stderr, "Usage: %s min_procs min_elems max_elems nb target_k hbins\n", argv[0]);
    exit(1);
  }

  if (rank == 0)
    fprintf(stderr, "min_procs = %d min_elems = %d max_elems = %d nb = %d target_k = %d hbins = %d\n",
	    min_procs, min_elems, max_elems, nb, target_k, hbins);
}

int main(int argc, char **argv)
{
  int nblocks;              // local number of blocks
  int tot_blocks;           // total number of blocks
  int target_k;             // target k-value
  int min_elems, max_elems; // min, max number of elements per block
  int num_elems;            // current number of data elements per block
  int rank, groupsize;      // MPI usual
  int min_procs;            // minimum number of processes
  int max_procs;            // maximum number of processes (groupsize of MPI_COMM_WORLD)
  int hbins;                // number of histogram bins
  int proc_x, elem_x;       // factors for procs and elems

  proc_x = 4;
  elem_x = 4;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &max_procs);

  GetArgs(argc, argv, min_procs, min_elems, max_elems, nblocks, target_k, hbins);

  // timing
  int num_runs = 0;
  for (int i = min_procs; i <= max_procs; i *= proc_x)
  {
    for (int j = min_elems; j <= max_elems; j *= elem_x)
      num_runs++;
  }
  double *ssort_time = new double[num_runs]; // sample sort from diy1
  double *dsort_time = new double[num_runs]; // sort from diy2

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
      groupsize *= proc_x;
      continue;
    }

    // initialize DIY
    tot_blocks = nblocks * groupsize;
    int dim = 1;
    int mem_blocks = -1;      // everything in core for now
    int num_threads = 1;      // needed in order to do timing
    diy::mpi::communicator    world(comm);
    diy::FileStorage          storage("./DIY.XXXXXX");
    diy::Master               master(world,
                                     &Block::create,
                                     &Block::destroy,
                                     mem_blocks,
                                     num_threads,
                                     &storage,
                                     &Block::save,
                                     &Block::load);
    diy::ContiguousAssigner   assigner(world.size(), tot_blocks);
    AddBlock                  create(master);
    Bounds domain;
    diy::decompose(dim, world.rank(), domain, assigner, create);

    // iterate over number of elements
    num_elems = min_elems;
    while (num_elems <= max_elems)
    {
      // DIY1 sort, only for one block per process
      if (tot_blocks == groupsize)
	Diy1Sort(ssort_time, run, comm, num_elems);

      // DIY2 sort
      int args[2];
      args[0] = num_elems;
      args[1] = target_k * hbins;
      master.foreach(ResetBlock, args);
      Diy2Sort(dsort_time, run, target_k, comm, tot_blocks, master, assigner);

      // debug
//       master.foreach(VerifyBlock);

      num_elems *= elem_x;
      run++;
    } // elem iteration

    groupsize *= proc_x;
    MPI_Comm_free(&comm);

  } // proc iteration

  // print results
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  fflush(stderr);
  if (rank == 0)
    PrintResults(ssort_time, dsort_time, min_procs, max_procs, proc_x, min_elems, max_elems,
                 elem_x);

  // cleanup
  delete[] ssort_time;
  delete[] dsort_time;
  MPI_Finalize();
  return 0;
}
