//--------------------------------------------------------------------------
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
#include <diy/reduce-operations.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>

#include "../../include/opts.h"

using namespace std;

typedef  diy::ContinuousBounds          Bounds;
typedef  diy::RegularContinuousLink     RCLink;
typedef  diy::RegularDecomposer<Bounds> Decomposer;

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
    void generate_data(int n_, int tot_b_)
        {
            size = n_;
            tot_b = tot_b_;
            data.resize(size);
            for (int i = 0; i < size; ++i)
            {
                data[i] = gid * size + i;
                // debug
//                 fprintf(stderr, "diy2 gid %d indata[%d] = %.1f\n", gid, i, data[i]);
            }
        }

    std::vector<float> data;
    int    gid;
    size_t size;   // total number of elements per block
    int tot_b;     // total number of blocks
};

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
    int tot_blocks = *((int*)args + 1);
    b->generate_data(num_elems, tot_blocks);
}

//
// prints data values in a block (debugging)
//
void PrintBlock(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
    Block* b   = static_cast<Block*>(b_);
    for (int i = 0; i < b->size; i++)
        fprintf(stderr, "diy2 gid %d reduced data[%d] = %.1f\n", b->gid, i, b->data[i]);
}

//
// checks diy2 block data against mpi data
//
void CheckBlock(void* b_, const diy::Master::ProxyWithLink& cp, void* rs_)
{
    Block* b   = static_cast<Block*>(b_);
    float* rs = static_cast<float*>(rs_);

    for (int i = 0; i < b->size; i++)
    {
        if (b->data[i] != rs[i])
            fprintf(stderr, "i = %d gid = %d size = %lu: "
                    "diy2 value %.1f does not match mpi reduced value %.2f\n",
                    i, b->gid, b->size, b->data[i], rs[i]);
    }
}

//
// MPI all_to_all and all_to_allv
//
void MpiAlltoAll(float    *alltoall_data,    // data values
                 double   *mpi_allall_time,  // time (output) for all_to_all
                 double   *mpi_allallv_time, // time (output) for all_to_allv
                 int      run,               // run number
                 float    *in_data,          // input data
                 MPI_Comm comm,              // current communicator
                 int      num_elems)         // current number of elements per block
{
    // init
    int rank;
    int groupsize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &groupsize);
    for (int i = 0; i < num_elems; i++) // init input data
    {
        in_data[i] = rank * num_elems + i;
        // debug
//         fprintf(stderr, "mpi rank %d indata[%d] = %.1f\n", rank, i, in_data[i]);
    }

    // reduce using alltoall
    // count is same for all processes
    // just num_elems / groupsize, dropping any remainder
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    MPI_Alltoall((void *)in_data, num_elems / groupsize, MPI_FLOAT,
                 (void *)alltoall_data, num_elems / groupsize, MPI_FLOAT, comm);
    MPI_Barrier(comm);
    mpi_allall_time[run] = MPI_Wtime() - t0;

    // reduce using alltoallv
    // even though count is actually same for all processes, this is how long it would take
    // if the count were different
    int* send_counts = new int[groupsize];
    int* send_displs = new int[groupsize];
    int* recv_counts = new int[groupsize];
    int* recv_displs = new int[groupsize];
    for (int i = 0; i < groupsize; i++)
    {
        send_counts[i] = num_elems / groupsize;
        send_displs[i] = (i == 0) ? 0 : send_displs[i - 1] + send_counts[i - 1];
        recv_counts[i] = num_elems / groupsize;
        recv_displs[i] = (i == 0) ? 0 : recv_displs[i - 1] + recv_counts[i - 1];
    }
    MPI_Barrier(comm);
    t0 = MPI_Wtime();
    MPI_Alltoallv((void *)in_data, send_counts, send_displs, MPI_FLOAT,
                  (void *)alltoall_data, recv_counts, recv_displs, MPI_FLOAT, comm);
    MPI_Barrier(comm);
    mpi_allallv_time[run] = MPI_Wtime() - t0;
    delete[] send_counts;
    delete[] send_displs;
    delete[] recv_counts;
    delete[] recv_displs;

    // debug: print the mpi data
//       for (int i = 0; i < num_elems; i++)
//         fprintf(stderr, "mpi rank %d reduced data[%d] = %.1f\n", rank, i, alltoall_data[i]);
}

//
// Exchange for DIY
// receives enqueued data and stores it in the same transposed locations as mpi
//
struct Exchange
{
    Exchange(const Decomposer& decomposer_):
        decomposer(decomposer_)              {}
    void operator()(void* b_, const diy::ReduceProxy& rp) const
        {
            Block* b = static_cast<Block*>(b_);

            // enqueue
            int sz = 0;                       // current location in b-> from which to read
            for (unsigned i = 0; i < rp.out_link().size(); i++)
            {
                rp.enqueue(rp.out_link().target(i), &b->data[sz], b->size / b->tot_b);
                sz += (b->size / b->tot_b);
            }

            // dequeue
            sz = 0;                          // current location in b->data into which to write
            for (unsigned i = 0; i < rp.in_link().size(); ++i)
            {
                int gid = rp.in_link().target(i).gid;
                diy::MemoryBuffer& incoming = rp.incoming(gid);
                size_t incoming_sz  = incoming.size() / sizeof(float);
                std::copy((float*) &incoming.buffer[0],
                          (float*) &incoming.buffer[0] + incoming_sz,
                          &b->data[sz]);
                sz += incoming_sz;
            }
        }

    const Decomposer& decomposer;
};

//
// DIY all to all
//
// diy_time: time (output)
// run: run number
// k: desired k value
// comm: MPI communicator
// master, assigner, decomposer: diy objects
//
void DiyAlltoAll(double *diy_time, int run, int k, MPI_Comm comm, diy::Master& master,
                 diy::ContiguousAssigner& assigner, Decomposer& decomposer)
{
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    //printf("---- %d ----\n", totblocks);
    diy::all_to_all(master, assigner, Exchange(decomposer), k);

    //printf("------------\n");
    MPI_Barrier(comm);
    diy_time[run] = MPI_Wtime() - t0;
}
//
// print results
//
void PrintResults(double *mpi_allall_time,   // mpi all_to_all time
                  double *mpi_allallv_time,  // mpi all_to_allv time
                  double *diy_time,          // diy time
                  int    min_procs,          // minimum number of processes
		  int    max_procs,          // maximum number of processes
                  int    min_elems,          // minimum number of elements
                  int    max_elems)          // maximum number of elements
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
        fprintf(stderr, "# procs \t mpi_allall_time \t mpi_allallv_time \t diy_time\n");

        // iterate over processes
        int groupsize = min_procs;
        proc_iter = 0;
        while (groupsize <= max_procs)
        {
            int i = proc_iter * num_elem_iters + elem_iter; // index into times
            fprintf(stderr, "%d \t\t %.3lf \t\t\t %.3lf \t\t\t %.3lf\n",
                    groupsize, mpi_allall_time[i], mpi_allallv_time[i], diy_time[i]);

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
//
void GetArgs(int argc, char **argv, int &min_procs,
	     int &min_elems, int &max_elems, int &nb, int &target_k)
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
          >> PosOption(target_k)))
    {
        if (rank == 0)
            fprintf(stderr, "Usage: %s min_procs min_elems max_elems nb target_k\n", argv[0]);
        exit(1);
    }

    // check there is at least one element per block
    if (min_elems < nb * max_procs && rank == 0)
    {
        fprintf(stderr, "Error: minimum number of elements must be >= maximum number of blocks "
                " so that there is at least one element per block\n");
        exit(1);
    }

    if (rank == 0)
        fprintf(stderr, "min_procs = %d min_elems = %d max_elems = %d nb = %d "
                "target_k = %d\n", min_procs, min_elems, max_elems, nb, target_k);
}

//
// main
//
int main(int argc, char **argv)
{
    int dim = 1;              // number of dimensions in the problem
    int nblocks;              // local number of blocks
    int tot_blocks;           // total number of blocks
    int target_k;             // target k-value
    int min_elems, max_elems; // min, max number of elements per block
    int num_elems;            // current number of data elements per block
    int rank, groupsize;      // MPI usual
    int min_procs;            // minimum number of processes
    int max_procs;            // maximum number of processes (groupsize of MPI_COMM_WORLD)

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &max_procs);

    GetArgs(argc, argv, min_procs, min_elems, max_elems, nblocks, target_k);

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
    double mpi_allall_time[num_runs];
    double mpi_allallv_time[num_runs];
    double diy_time[num_runs];

    // data for MPI reduce, only for one local block
    float *in_data = new float[max_elems];
    float *alltoall_data = new float[max_elems];

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
        diy::ContiguousAssigner   assigner(world.size(), tot_blocks);
        AddBlock                  create(master);
        Decomposer    decomposer(dim, domain, assigner);
        decomposer.decompose(world.rank(), create);

        // iterate over number of elements
        num_elems = min_elems;
        while (num_elems <= max_elems)
        {
            // MPI alltoall, only for one block per process
            if (tot_blocks == groupsize)
                MpiAlltoAll(alltoall_data,
                            mpi_allall_time,
                            mpi_allallv_time,
                            run,
                            in_data,
                            comm,
                            num_elems);

            // initialize DIY input data
            int args[2];
            args[0] = num_elems;
            args[1] = tot_blocks;
            master.foreach(&ResetBlock, args);

            // DIY alltoall
            DiyAlltoAll(diy_time,
                        run,
                        target_k,
                        comm,
                        master,
                        assigner,
                        decomposer);

            // debug
//             master.foreach(&PrintBlock);
            master.foreach(&CheckBlock, alltoall_data);

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
        PrintResults(mpi_allall_time,
                     mpi_allallv_time,
                     diy_time,
                     min_procs,
                     max_procs,
                     min_elems,
                     max_elems);

    // cleanup
    delete[] in_data;
    delete[] alltoall_data;
    MPI_Finalize();

    return 0;
}
