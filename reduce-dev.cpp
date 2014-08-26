//--------------------------------------------------------------------------
//
// development of diy2 swap reduction as an application example before building into diy core
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
#include <diy/communicator.hpp>
#include <diy/master.hpp>
#include <diy/assigner.hpp>
#include <diy/serialization.hpp>
#include <diy/decomposition.hpp>

using namespace std;

typedef  diy::ContinuousBounds       Bounds;
typedef  diy::RegularContinuousLink  RCLink;

// block
struct block_t
{
  int gid;
  vector<int> buf;
};

// auxiliaary arguments to SwapReduce
struct aux_t
{
  int num_ints; // number of ints in one item
  vector<int> kv; // k-values
  int round; // current round
  diy::RoundRobinAssigner* assigner;
};

// function prototypes
void GetArgs(int argc, char **argv, int& num_ints, int& nb, diy::Communicator diy_comm);
void PrintResults(double *enqueue_time, double *exchange_time,
		  double *flush_time, int min_procs, int max_procs,
		  int min_items, int max_items, int item_size,
		  int num_item_iters, int proc_factor, int item_factor);
void* create_block();
void destroy_block(void* b_);
void save_block(const void* b, diy::BinaryBuffer& bb);
void load_block(void* b, diy::BinaryBuffer& bb);
void SwapEnqueue(void* b_, const diy::Master::ProxyWithLink& cp, void*);
void SwapDequeue(void* b_, const diy::Master::ProxyWithLink& cp, void*);
void GetPartners(const vector<int>& kv, int cur_r, int gid, vector<int>& partners);
void GetGrpPos(int cur_r, const vector<int>& kv, int gid, int& g, int& p);
void subset(block_t* b, vector<int>& send_buf, int cur_round);
void reduce(block_t* b, vector< vector <int > >& recv_bufs, vector<int>& gids);

// add blocks to a master
struct AddBlock
{
  AddBlock(diy::Master& master_, int num_ints_):
    master(master_), num_ints(num_ints_)           {}

  void  operator()(int gid, const Bounds& core, const Bounds& bounds, const Bounds& domain,
                   const RCLink& link) const
  {
    block_t*      b = static_cast<block_t*>(create_block());
    RCLink*       l = new RCLink(link);
    diy::Master&  m = const_cast<diy::Master&>(master);
    m.add(gid, b, l);

    b->gid = gid;
    b->buf.resize(num_ints, 0);
  }

  diy::Master&  master;
  int num_ints;
};

// serialize a block
namespace diy
{
  template<>
  struct Serialization<block_t>
  {
    static void save(BinaryBuffer& bb, const block_t& d)
    {
      diy::save(bb, d.buf);
    }

    static void load(BinaryBuffer& bb, block_t& d)
    {
      diy::load(bb, d.buf);
    }
  };
}
//
// user-defined callbacks
//
void subset(block_t* b, vector<int>& send_buf, int cur_round)
{
  // simple full copy
  for (int i = 0; i < (int)b->buf.size(); i++)
    send_buf.push_back(b->buf[i]);
}
void reduce(block_t* b, vector< vector <int > >& recv_bufs, vector<int>& gids)
{
  // simple sum of full size
  for (int i = 0; i < (int)b->buf.size(); i++)
  {
    for (int j = 0; j < (int)gids.size(); j++)
      b->buf[i] += recv_bufs[j][i];
  }
}
//----------------------------------------------------------------------------
//
// main
//
int main(int argc, char **argv)
{
  int dim = 3; // number of dimensions in the problem
  int rank, groupsize; // MPI usual
  int nblocks; // my local number of blocks
  int num_ints; // number of ints in the buffer to be reduced

  // init MPI and diy
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &groupsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  diy::mpi::communicator world(mpi_comm);
  diy::Communicator diy_comm(world);
  diy::FileStorage          storage("./DIY.XXXXXX");

  GetArgs(argc, argv, num_ints, nblocks, diy_comm);

  // data extents, unused
  Bounds domain;
  for(int i = 0; i < dim; i++)
  {
    domain.min[i] = 0.0;
    domain.max[i] = 1.0;
  }

  // initialize DIY
  int tot_blocks = nblocks * groupsize; // total global number of blocks
  int mem_blocks = -1; // everything in core for now


  diy::Master               master(diy_comm,
                                   &create_block,
                                   &destroy_block,
                                   mem_blocks,
                                   &storage,
                                   &save_block,
                                   &load_block);
  diy::RoundRobinAssigner   assigner(world.size(), tot_blocks);
  AddBlock create(master, num_ints);
  rank = world.rank();

  // decompose
  std::vector<int> my_gids;
  assigner.local_gids(diy_comm.rank(), my_gids);
  nblocks = my_gids.size();
  diy::decompose(dim, rank, domain, assigner, create);

  // auxiliary arguments for the swap reduction
  struct aux_t aux;
  aux.num_ints; // number of ints in one item
  aux.assigner = &assigner;
  aux.kv.push_back(2); // TODO: hard-coded for now

  // for all rounds
  for (aux.round = 0; aux.round < (int)(aux.kv.size()); aux.round++)
  {
    master.foreach(&SwapEnqueue, &aux);
    master.exchange();
    master.foreach(&SwapDequeue, &aux);
  }

  MPI_Finalize();

  return 0;
}
//----------------------------------------------------------------------------
//
// foreach block functions
//
void SwapEnqueue(void* b_, const diy::Master::ProxyWithLink& cp, void* a_)
{
  block_t* b = (block_t*)b_;
  aux_t* a = (aux_t*)a_;

  // get the partners for my group in this round
  vector<int> partners; // gids in my group, excluding myself
  GetPartners(a->kv, a->round, b->gid, partners);

  // setup the link for this goup
  diy::Link link;
  for (int i = 0; i < a->kv[a->round]; ++i)
  {
    diy::BlockID  neighbor;
    // TODO: have partners exclude myself?
    // if so, following test would not be needed
    if (partners[i] != b->gid) // skip myself
    {
      neighbor.gid  = partners[i];
      neighbor.proc = a->assigner->rank(neighbor.gid);
      link.add_neighbor(neighbor);
    }
  }

  // TODO: faking the type of buffer and leaving its contents uninitialized
  vector<int> send_buf; // TODO: is there ever a need to have separate buffers for each partner?

  // subset the send buffer
  subset(b, send_buf, a->round);

  // enqueue items within the link
  for (int j = 0; j < link.count(); j++)
    cp.enqueue(link.target(j), send_buf);
}

void SwapDequeue(void* b_, const diy::Master::ProxyWithLink& cp, void* a_)
{
  block_t* b = (block_t*)b_;
  aux_t* a = (aux_t*)a_;

  // get gids of partners for my group in this round
  vector<int> partners; // gids in my group, excluding myself
  GetPartners(a->kv, a->round, b->gid, partners);

  // TODO: faking the type of buffer and leaving its contents uninitialized
  vector< vector <int> > recv_bufs(a->kv[a->round] - 1);
  for (int i = 0; i < a->kv[a->round] - 1; i++)
    recv_bufs[i].resize(a->num_ints);

  std::vector<int> in; // gids of sources
  cp.incoming(in);

  for (int i = 0; i < (int)in.size(); i++)
    cp.dequeue(in[i], recv_bufs[i]);

  // do the reduction
  reduce(b, recv_bufs, partners);
}
//----------------------------------------------------------------------------
//
// helper functions
//
// gets the global ids of the blocks in my group
//
// kv: vector of k values
// cur_r: current round number (0 to r - 1)
// gid: global id of the block
// partners: global ids of the partners (blocks) in my group, excluding myself (output)
//
void GetPartners(const vector<int>& kv, int cur_r, int gid, vector<int>& partners)
{
  int step = 1; // gids jump by this much in the current round
  int pos; // position of the block in the group
  int unused;
  partners.resize(kv[cur_r] - 1); // TODO: reserve or resize?

  GetGrpPos(cur_r, kv, gid, unused, pos);

  for (int r = 0; r < cur_r; r++)
    step *= kv[r];

  int partner = gid - pos * step;
  if (partner != gid)
    partners.push_back(partner);
  for (int k = 1; k < kv[cur_r]; k++)
  {
    partner += step;
    if (partner != gid)
      partners.push_back(partner);
  }
}
//
// computes group number and position within that group for my block
// to participate in the swap communication
//
// group number is 0 to the global number of groups in the current round - 1
// position number is 0 to k value of the current round - 1
//
// cur_r: current round
// kv: vector of k values
// gid: global id of the block
// g: group number (output)
// p: position number within the group (output)
//
void GetGrpPos(int cur_r, const vector<int>& kv, int gid, int& g, int& p)
{
  int step = 1;

  for (int i = 0; i < cur_r; i++)
    step *= kv[i];

  // the second term in the following expression does not simplify to
  // (gid - start_b) / kv[r]
  // because the division gid / (step * kv[r]) is integer and truncates
  // this is exactly what we want
  g = gid % step + gid / (step * kv[cur_r]) * step;

  p = gid / step % kv[cur_r];

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
  b->buf.clear();
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
// gets command line args
//
// argc, argv: usual
// num_ints: number of ints per item (output)
// nb: number of local blocks per process (output)
//
void GetArgs(int argc, char **argv, int& num_ints, int& nb, diy::Communicator diy_comm)
{
  assert(argc >= 3);

  num_ints = atoi(argv[1]);
  nb = atoi(argv[2]);

  if (diy_comm.rank() == 0)
    fprintf(stderr, "num_procs = %d num_ints = %d nb = %d\n", diy_comm.size(), num_ints, nb);
}
//----------------------------------------------------------------------------
