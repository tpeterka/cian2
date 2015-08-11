// TODO:
// empty file cache each time?
// compare read and written blocks? (I checked the first tiny run manually)

//--------------------------------------------------------------------------
//
// testing DIY's i/o performance
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
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/io/block.hpp>

#include "../include/opts.h"

using namespace std;

typedef  diy::ContinuousBounds       Bounds;
typedef  diy::RegularContinuousLink  RCLink;

//
// block
//
struct Block
{
    Block()                                                     {}
    static void*    create()                                    { return new Block; }
    static void     destroy(void* b)                            { delete static_cast<Block*>(b); }
    static void     save(const void* b, diy::BinaryBuffer& bb)
        { diy::save(bb, *static_cast<const Block*>(b)); }
    static void     load(void* b, diy::BinaryBuffer& bb)
        { diy::load(bb, *static_cast<Block*>(b)); }
    void generate_data(int n_)
        {
            size = n_;
            data.resize(size);
            for (int i = 0; i < size; ++i)
                data[i] = gid * size + i;
        }

    std::vector<float> data;                 // block values
    int gid;                                 // block global id
    size_t size;                             // number of values
};

namespace diy
{
    template<>
        struct Serialization<Block>
    {
        static void save(BinaryBuffer& bb, const Block& b)
        {
            diy::save(bb, b.data);
            diy::save(bb, b.gid);
            diy::save(bb, b.size);
        }

        static void load(BinaryBuffer& bb, Block& b)
        {
            diy::load(bb, b.data);
            diy::load(bb, b.gid);
            diy::load(bb, b.size);
        }
    };
}

//
// add blocks to a master
//
struct AddBlock
{
    AddBlock(diy::Master& master_):
        master(master_)     {}

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

//
// reset the size and data values in a block
// args[0]: num_elems
// args[1]: tot_blocks
//
void ResetBlock(void* b_, const diy::Master::ProxyWithLink& cp, void* args)
{
    Block* b   = static_cast<Block*>(b_);
    int num_elems = *(int*)args;
    b->generate_data(num_elems);
}

//
// prints data values in a block (debugging)
//
void PrintBlock(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
    Block* b   = static_cast<Block*>(b_);
    for (int i = 0; i < b->size; i++)
        fprintf(stderr, "gid %d size %lu data[%d] = %.1f\n", b->gid, b->size, i, b->data[i]);
}

//
// print results
//
// time: time
// tot_b: total number of blocks
// min_procs, max_procs: process range
// min_elems, max_elems: data range
//
void PrintResults(double *time,
                  int tot_b,
                  int min_procs,
                  int max_procs,
                  int min_elems,
                  int max_elems)
{
    int elem_iter = 0;                                            // element iteration number
    int num_elem_iters = (int)(log2(max_elems / min_elems) + 1);  // number of element iterations
    int proc_iter = 0;                                            // process iteration number
    float gb = 1073741824.0f;                                     // 1 GB

    fprintf(stderr, "----- Timing Results -----\n");

    // iterate over number of elements
    int num_elems = min_elems;
    while (num_elems <= max_elems)
    {
        fprintf(stderr, "\n# num_elemnts = %d   size @ 4 bytes / element = %d KB\n",
                num_elems, num_elems * 4 / 1024);
        fprintf(stderr, "# procs \t time(s) \t bw(GB/s)\n");

        // iterate over processes
        int groupsize = min_procs;
        proc_iter = 0;
        while (groupsize <= max_procs)
        {
            int i = proc_iter * num_elem_iters + elem_iter; // index into times
            fprintf(stderr, "%d \t\t %.3lf \t\t %.3lf\n",
                    groupsize, time[i], (float)num_elems * 4.0f * (float)tot_b / gb / time[i]);

            groupsize *= 2; // double the number of processes every time
            proc_iter++;
        } // proc iteration

        num_elems *= 2; // double the number of elements every time
        elem_iter++;
    } // elem iteration

    fprintf(stderr, "\n--------------------------\n\n");
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
// write: write (true) or read (false)
//
void GetArgs(int argc,
             char **argv,
             int &min_procs,
             int &min_elems,
             int &max_elems,
             int &nb,
             bool &write)
{
    using namespace opts;
    Options ops(argc, argv);
    int max_procs;
    int rank;
    char op;                                 // w or r (write or read)
    MPI_Comm_size(MPI_COMM_WORLD, &max_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (ops >> Present('h', "help", "show help") ||
        !(ops >> PosOption(min_procs)
          >> PosOption(min_elems)
          >> PosOption(max_elems)
          >> PosOption(nb)
          >> PosOption(op)))
    {
        if (rank == 0)
            fprintf(stderr, "Usage: %s min_procs min_elems max_elems nb\n", argv[0]);
        exit(1);
    }

    // check there is at least one element per block
    if (min_elems < nb * max_procs && rank == 0)
    {
        fprintf(stderr, "Error: minimum number of elements must be >= maximum number of blocks "
                " so that there is at least one element per block\n");
        exit(1);
    }

    write = (op == 'w' || op == 'W') ? true : false;

    if (rank == 0)
        fprintf(stderr, "min_procs = %d min_elems = %d max_elems = %d nb = %d write = %d\n",
                min_procs, min_elems, max_elems, nb, write);
}

//
// main
//
int main(int argc, char **argv)
{
    int dim = 1;              // number of dimensions in the problem
    int nblocks;              // local number of blocks
    int tot_blocks;           // total number of blocks
    int min_elems, max_elems; // min, max number of elements per block
    int num_elems;            // current number of data elements per block
    int rank, groupsize;      // MPI usual
    int min_procs;            // minimum number of processes
    int max_procs;            // maximum number of processes (groupsize of MPI_COMM_WORLD)
    double t0;                // start time
    char buf[256];            // filename
    bool write;               // write or read

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &max_procs);

    GetArgs(argc, argv, min_procs, min_elems, max_elems, nblocks, write);

    // data extents, unused
    Bounds domain;
    for(int i = 0; i < dim; i++)
    {
        domain.min[i] = 0.0;
        domain.max[i] = 1.0;
    }

    int num_runs = (int)((log2(max_procs / min_procs) + 1) *
                         (log2(max_elems / min_elems) + 1));

    double io_time[num_runs];                // timing

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
        diy::Master               master(world,
                                         num_threads,
                                         mem_blocks,
                                         &Block::create,
                                         &Block::destroy,
                                         &storage,
                                         &Block::save,
                                         &Block::load);
        diy::ContiguousAssigner   *assigner;
        if (write)
        {
            assigner = new diy::ContiguousAssigner(world.size(), tot_blocks);
            AddBlock                  create(master);
            diy::decompose(dim, world.rank(), domain, *assigner, create);
        }
        else // number of blocks set by read_blocks()
            assigner = new diy::ContiguousAssigner(world.size(), -1);

        // iterate over number of elements
        num_elems = min_elems;
        while (num_elems <= max_elems)
        {
            // initialize input data
            master.foreach(&ResetBlock, &num_elems);

            // debug
//             master.foreach(&PrintBlock);

            // name the file by the run number
            // rank 0 has the correct run numbering because it participated in all groups, so
            // rank 0 broadcasts the run number (easier, safer than trying to compute it)
            MPI_Bcast(&run, 1, MPI_INT, 0, comm);
            sprintf(buf, "%d.out", run);

            // write the data
            if (write)
            {
                MPI_Barrier(comm);
                t0 = MPI_Wtime();
                diy::io::write_blocks(buf, world, master);
                MPI_Barrier(comm);
                io_time[run] = MPI_Wtime() - t0;
            }

            // read the data
            else
            {
                MPI_Barrier(comm);
                t0 = MPI_Wtime();
                diy::io::read_blocks(buf, world, *assigner, master);
                MPI_Barrier(comm);
                io_time[run] = MPI_Wtime() - t0;
            }

            // debug
//             master.foreach(&PrintBlock);

            num_elems *= 2; // double the number of elements every time
            run++;

        } // elem iteration

        groupsize *= 2; // double the number of processes every time
        delete assigner;
        MPI_Comm_free(&comm);

    } // proc iteration

    // print results
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fflush(stderr);
    if (rank == 0)
        PrintResults(io_time, tot_blocks, min_procs, max_procs, min_elems, max_elems);

    // cleanup
    MPI_Finalize();

    return 0;
}
