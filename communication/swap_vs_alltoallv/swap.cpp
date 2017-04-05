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
#include <diy/reduce.hpp>
#include <diy/partners/swap.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>

#include "../../include/opts.h"

using namespace std;

typedef  diy::ContinuousBounds          Bounds;
typedef  diy::RegularContinuousLink     RCLink;
typedef  diy::RegularDecomposer<Bounds> Decomposer;

void NoopSwap(void* b_, const diy::ReduceProxy& rp, const diy::RegularSwapPartners&);

// block
struct Block
{
    Block()                                                     {}
    static void*    create()                                    { return new Block; }
    static void     destroy(void* b)                            { delete static_cast<Block*>(b); }
    static void     save(const void* b_, diy::BinaryBuffer& bb)
    {
        const Block& b = *static_cast<const Block*>(b_);
        diy::save(bb, b.data);
        diy::save(bb, b.gid);
        diy::save(bb, b.sub_start);
        diy::save(bb, b.sub_size);
        diy::save(bb, b.nrays);
        diy::save(bb, b.nray_elems);
    }
    static void     load(void* b_, diy::BinaryBuffer& bb)
    {
        Block& b = *static_cast<Block*>(b_);
        diy::load(bb, b.data);
        diy::load(bb, b.gid);
        diy::load(bb, b.sub_start);
        diy::load(bb, b.sub_size);
        diy::load(bb, b.nrays);
        diy::load(bb, b.nray_elems);
    }
    void generate_data(int tot_nrays,       // total global number of rays
            int avg_elems,                  // average number of elements per ray
            int reduce_factor_,             // number of elements composed into one in each round
            diy::Master& master)
    {
        // each block starts with the entire image of all rays
        nrays = tot_nrays;

        // even though in this example the number of elements per ray is given as a constant
        // (avg_elems), the block data model is written as if the number of elements per ray
        // varies
        data.resize(nrays * avg_elems);

        for (int i = 0; i < data.size(); ++i)
            data[i] = i / avg_elems;    // for now just give all the elements in a ray the same value
        nray_elems.resize(nrays);
        for (int i = 0; i < nrays; i++)
            nray_elems[i] = avg_elems;

        sub_start = 0;
        sub_size  = nrays * avg_elems;
        start_ray = 0;
        reduce_factor = reduce_factor_;
    }

    // assume there are a variable number of elements per ray, each element is one float
    std::vector<float> data;        // element data for all rays
    int gid;                        // block global id
    size_t sub_start;               // starting index of subset of the total data that this block owns
    size_t sub_size;                // number of elements in the subset of the total data that this block owns
    size_t start_ray;               // index of first ray in data range [sub_start, sub_start + sub_size - 1]
    size_t nrays;                   // number of rays
    std::vector<int> nray_elems;    // number of elements in each ray
    int reduce_factor;              // number of elements composed into one in each round
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
//
void ResetBlock(
        Block*                              b,
        const  diy::Master::ProxyWithLink&  cp,
        int                                 num_rays,
        int                                 avg_elems,
        int                                 reduce_factor,
        diy::Master&                        master)
{
    b->generate_data(num_rays, avg_elems, reduce_factor, master);
}
//
// prints data values in a block (debugging)
//
void PrintBlock(Block* b,
        const diy::Master::ProxyWithLink& cp,
        int avg_elems)
{
    fprintf(stderr, "gid = %d sub_start = %ld sub_size = %ld start_ray=%ld nrays = %ld\n",
            b->gid, b->sub_start, b->sub_size, b->start_ray, b->nrays);
    int n = b->sub_start;
    for (int i = 0; i < b->nrays; i++)
    {
        fprintf(stderr, "ray %d = ", i);
        for (int j = 0; j < b->nray_elems[b->start_ray + i]; j++)
            fprintf(stderr, "%.1f ", b->data[n++]);
        fprintf(stderr, "\n");
    }
}
//
// MPI all_to_all_v
//
// all_all_v_data: data values (output) allocated by caller
// all_all_v_time: time (output) allocated by caller
// run: run number
// in_data: input data allocated by caller
// comm: current communicator
// nrays: total global number of rays (all procs start with entire image of all rays)
// avg_elems: average number of elements per ray
//
void MpiAlltoAllv(float* all_all_v_data, double *all_all_v_time, int run,
        float *in_data, MPI_Comm comm, int nrays, int avg_elems)
{
    // init
    int rank;
    int groupsize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &groupsize);

    // initialize data
    // even though in this example the number of elements per ray is given as a constant
    // (avg_elems), the data model is written as if the number of elements per ray varies
    for (int i = 0; i < nrays * avg_elems; ++i)
        in_data[i] = i / avg_elems;         // for now just give all the elements in a ray the same value
    int* nray_elems = new int[nrays];       // number of elements in each ray
    for (int i = 0; i < nrays; i++)
        nray_elems[i] = avg_elems;

    // start the timer
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    // distribute rays in some arbitrary way across the ranks
    // in this example, use a contiguous partitioning
    // in a real example, would need to sort the rays into ranges by destination rank

    // exchange number of elements per ray
    int* ray_counts  = new int[groupsize];
    int* displs      = new int[groupsize];
    int* recv_counts = new int[groupsize];
    int* recv_displs = new int[groupsize];
    int recv_nrays = nrays / groupsize;                         // number of rays in the chunk I am receiving
    double f = ((double)groupsize - 1) / (double)groupsize;     // fraction of ranks excluding the last one
    if (rank == groupsize - 1)
        recv_nrays = (1.0 - f) * nrays;
    int* recv_nray_elems = new int[nrays];
    for (int i = 0; i < groupsize; i++)
    {
        ray_counts[i]  = nrays / groupsize;
        displs[i]      = (i == 0 ? 0 : displs[i - 1] + ray_counts[i]);
        recv_counts[i] = recv_nrays;
        recv_displs[i] = (i == 0 ? 0 : displs[i - 1] + recv_counts[i - 1]);
    }
    ray_counts[groupsize - 1] = (1.0 - f) * nrays; // remainder
    //     if (rank == 0)
    //     {
    //         fprintf(stderr, "1: f=%lf 1.0-f=%lf rem_n_rays=%d\n", f, 1.0 - f, (int)((1.0 - f) * nrays));
    //         fprintf(stderr, "2: ray_counts[%d %d ... %d] displs[%d %d ... %d] recv_counts[%d %d ... %d] recv_displs[%d %d ... %d]\n",
    //                 ray_counts[0], ray_counts[1], ray_counts[groupsize - 1], displs[0], displs[1], displs[groupsize - 1],
    //                 recv_counts[0], recv_counts[1], recv_counts[groupsize - 1], recv_displs[0], recv_displs[1], recv_displs[groupsize - 1]);
    //     }
    MPI_Alltoallv((void *)nray_elems, ray_counts, displs, MPI_INT,
            (void *)recv_nray_elems, recv_counts, recv_displs, MPI_INT, comm);

    // the number of elements to be sent (in general could vary per ray, need to add them up)
    int* elem_counts = new int[groupsize];
    for (int j = 0; j < groupsize; j++)
    {
        elem_counts[j] = 0;
        for (int i = 0; i < ray_counts[j]; i++)
            elem_counts[j] += avg_elems;
    }

    //     if (rank == 0)
    //         fprintf(stderr, "2: elem_counts[%d %d ... %d]\n", elem_counts[0], elem_counts[1], elem_counts[groupsize - 1]);

    // the number of elements to be received, total and from each process
    int tot_recv_nelems = 0;
    int n = 0;
    for (int j = 0; j < groupsize; j++)
    {
        recv_counts[j] = 0;
        for (int i = 0; i < recv_nrays; i++)
        {
            tot_recv_nelems += recv_nray_elems[n];
            recv_counts[j]  += recv_nray_elems[n];
            n++;
        }
    }

    //     fprintf(stderr, "3: recv_nelems=%d\n", tot_recv_nelems);

    // the elements in the rays
    float* recv_data = new float[tot_recv_nelems];
    for (int i = 0; i < groupsize; i++)
    {
        displs[i]      = (i == 0 ? 0 : displs[i - 1] + elem_counts[i - 1]);
        recv_displs[i] = (i == 0 ? 0 : displs[i - 1] + recv_counts[i - 1]);
    }
    //     fprintf(stderr, "4: counts[%d %d] displs[%d %d] recv_counts[%d %d] recv_displs[%d %d]\n",
    //             counts[0], counts[1], displs[0], displs[1],
    //             recv_counts[0], recv_counts[1], recv_displs[0], recv_displs[1]);
    MPI_Alltoallv(in_data, elem_counts, displs, MPI_FLOAT,
            recv_data, recv_counts, recv_displs, MPI_FLOAT, comm);

    // "reduce" the received elements, which in this no-op test is just a copy from recv_data
    // into all_all_v_data, with the received data from all procs overwriting the portion of nrays
    // for which this process is reponsible
    int elem_idx = 0;                           // index in output elementsj
    int ray_idx  = 0;                           // index in rays
    for (int i = 0; i < tot_recv_nelems; i++)
    {
        all_all_v_data[elem_idx++] = recv_data[i];
        if (elem_idx == recv_counts[ray_idx])
        {
            ray_idx++;
            elem_idx = 0;
        }
    }

    // debug: print the output data
    //     fprintf(stderr, "all_all_v_data:\n");
    //     for (int i = 0; i < recv_counts[rank]; i++)
    //         fprintf(stderr, "%.1f ", all_all_v_data[i]);
    //     fprintf(stderr, "\n");

    // cleanup
    delete[] nray_elems;
    delete[] ray_counts;
    delete[] displs;
    delete[] recv_counts;
    delete[] recv_displs;
    delete[] recv_nray_elems;
    delete[] elem_counts;
    delete[] recv_data;

    // stop the timer
    MPI_Barrier(comm);
    all_all_v_time[run] = MPI_Wtime() - t0;
}

// DIY swap
//
void DiySwap(double *swap_time,                          // time (output)
        int run,                                    // run number
        int k,                                      // desired k value (reduction tree radix)
        MPI_Comm comm,                              // MPI communicator
        Decomposer& decomposer,                     // diy RegularDecomposer object
        diy::Master& master,                        // diy Master object
        diy::ContiguousAssigner& assigner)          // diy Assigner object
{
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    diy::RegularSwapPartners  partners(decomposer, k, true);
    diy::reduce(master, assigner, partners, &NoopSwap);

    MPI_Barrier(comm);
    swap_time[run] = MPI_Wtime() - t0;
}
//
// Reduction (noop) for DIY swap
//
void NoopSwap(void* b_,
        const diy::ReduceProxy& rp,
        const diy::RegularSwapPartners& partners)
{
    Block* b = static_cast<Block*>(b_);
    int sub_start;              // subset starting index
    int sub_size;               // subset size
    int* nray_elems;            // pointer to just received or current number of elements in each ray
    float* data;                // pointer to just received data

    // find my position in the link
    int k = rp.in_link().size();
    int mypos;
    if (k > 0)
    {
        for (int i = 0; i < k; i++)
            if (rp.in_link().target(i).gid == rp.gid())
                mypos = i;

        // sub_start, starting ray index, and number of rays to be received
        int ray_idx = b->start_ray;         // index of current ray
        int nrays   = b->nrays / k;         // number of rays in any section except last one
        for (int i = 0; i < mypos; i++)
            for (int j = 0; j < nrays; j++)
                b->sub_start += b->nray_elems[ray_idx++];
        b->start_ray = ray_idx;
        if (mypos == k - 1)                 // last subset may be different size
            nrays = b->nrays - mypos * b->nrays / k;
        b->nrays = nrays;

        for (int i = 0; i < k; i++)
        {

            if (rp.in_link().target(i).gid == rp.gid())
                continue;

            // "shallow" dequeue number of elements in each ray (set pointer to start)
            nray_elems = (int*)&rp.incoming(rp.in_link().target(i).gid).buffer[0];

            // sub_size = total number of ray elements received
            b->sub_size = 0;
            for (int j = 0; j < b->nrays; j++)
                b->sub_size += nray_elems[j];

                // debug: print number of elements per ray
            //             for (int j = 0; j < b->nrays; j++)
            //                 fprintf(stderr, "ray %d has %d elements\n", j, nray_elems[j]);

            // "shallow" dequeue ray elements (set pointer to their start)
            // first (b->nrays * sizeof(int)) bytes of the buffer contain the previous number of
            // elements per ray; element data starts after that
            data = (float*)&rp.incoming(rp.in_link().target(i).gid).buffer[b->nrays * sizeof(int)];

            // reduce the contents of data with the contents of b->data starting at sub_start and
            // for size sub_size; in this proxy app, just replace b->data with data and reduce
            // number of elements by reduce_factor in each round to mimic the compositing
            // together of the elements by some factor
            // if reduce_factor == 0, don't copy anything (mimics a no-op, not even a copy, in each round)
            if (b->reduce_factor > 1)
            {
                memcpy(&b->data[b->sub_start], data, b->sub_size * sizeof(float));
                for (int i = 0; i < b->nrays; i++)
                    nray_elems[i] /= b->reduce_factor;
                // adjust sub_size for the reduced number of elements that were composed
                // not exactly correct because this leaves holes in the data (each ray has fewer
                // elements), but we're not repacking the data because the alltoallv version is just
                // a simple memcopy too, not a data reshuffling either
                b->sub_size /= b->reduce_factor;
            }

            // debug: print the received data
            //             fprintf(stderr, "received data sub_start=%ld sub_size=%ld:\n", b->sub_start, b->sub_size);
            //             for (int i = 0; i < b->sub_size; i++)
            //                 fprintf(stderr, "%.1f ", b->data[b->sub_start + i]);
            //             fprintf(stderr, "\n");
        }
    }

    // last round: save nray_elems in the block and data if it wasn't copied before
    if (!rp.out_link().size())
    {
        memcpy(&b->nray_elems[b->start_ray], nray_elems, b->nrays * sizeof(int));
        if (b->reduce_factor <= 1)
            memcpy(&b->data[b->sub_start], data, b->sub_size * sizeof(float));
    }

    // first round: get nray_elems from the block
    if (!rp.in_link().size())
        nray_elems = &b->nray_elems[0];

    // enqueue
    k = rp.out_link().size();
    sub_start = b->sub_start;                   // start of the section to send
    for (int i = 0; i < k; i++)                 // for all neighbors
    {
        // number of rays is divided evenly among neighbors, except for the last neighbor remainder
        // however, each ray can have a different number of elements
        int send_nrays;
        if (i < k - 1)
            send_nrays = b->nrays / k;
        else
            send_nrays = b->nrays - i * b->nrays / k;   // remainder

        // send the number of elements in each ray that will follow
        int ofst = i * b->nrays / k;
        rp.enqueue(rp.out_link().target(i), &nray_elems[ofst], send_nrays);

        // sub_start and sub_size mark the section of actual ray elements to send to this neighbor
        sub_size = 0;
        for (int j = 0; j < send_nrays; j++)
            sub_size += nray_elems[j];

        // send the actual ray elements
        rp.enqueue(rp.out_link().target(i), &b->data[sub_start], sub_size);

        // debug: print the sent data
        //         fprintf(stderr, "sent data sub_start=%d sub_size=%d:\n", sub_start, sub_size);
        //         for (int i = 0; i < sub_size; i++)
        //             fprintf(stderr, "%.1f ", b->data[sub_start + i]);
        //         fprintf(stderr, "\n");

        sub_start += sub_size;
    }
}
//
// print results
//
// all_all_v_time, swap_time: times
// min_procs, max_procs: process range
// min_rays, max_rays: data range
//
void PrintResults(double *all_all_v_time,
        double *swap_time,
        int min_procs,
        int max_procs,
        int min_rays,
        int max_rays,
        int avg_elems)
{
    int ray_iter = 0;                                            // ray iteration number
    int num_ray_iters = (int)(log2(max_rays / min_rays) + 1);    // number of ray iterations
    int proc_iter = 0;                                           // process iteration number

    fprintf(stderr, "----- Timing Results -----\n");

    // iterate over number of rays
    int num_rays = min_rays;
    while (num_rays <= max_rays)
    {
        fprintf(stderr, "\n# num_rays = %d   size @ %d elements * 4 bytes / ray = %d KB\n",
                num_rays, avg_elems, num_rays * avg_elems * 4 / 1024);
        fprintf(stderr, "# procs \t all_all_v_time \t swap_time\n");

        // iterate over processes
        int groupsize = min_procs;
        proc_iter = 0;
        while (groupsize <= max_procs)
        {
            int i = proc_iter * num_ray_iters + ray_iter; // index into times
            fprintf(stderr, "%d \t\t %.3lf \t\t\t %.3lf\n", groupsize, all_all_v_time[i], swap_time[i]);

            groupsize *= 2; // double the number of processes every time
            proc_iter++;
        } // proc iteration

        num_rays *= 2; // double the number of rays every time
        ray_iter++;
    } // ray iteration

    fprintf(stderr, "\n--------------------------\n\n");
}
//
// gets command line args
//
// argc, argv: usual
// min_procs: minimum number of processes (output)
// min_rays: minimum number of rays (output)
// max_rays: maximum number of rays (output)
// nb: number of blocks per process (output)
// target_k: target k-value (output)
// avg_elems: average number of elements per ray
// reduce_factor: reduce number of elements in each round by this factor (compose them together)
//
void GetArgs(int argc,
        char **argv,
        int &min_procs,
        int &min_rays,
        int &max_rays,
        int &nb,
        int &target_k,
        int &avg_elems,
        int &reduce_factor)
{
    using namespace opts;
    Options ops(argc, argv);
    int max_procs;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &max_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (ops >> Present('h', "help", "show help") ||
            !(ops >> PosOption(min_procs)
                >> PosOption(min_rays)
                >> PosOption(max_rays)
                >> PosOption(nb)
                >> PosOption(target_k)
                >> PosOption(avg_elems)
                >> PosOption(reduce_factor)))
    {
        if (rank == 0)
            fprintf(stderr, "Usage: %s min_procs min_rays max_rays nb target_k avg_elems reduce_factor\n", argv[0]);
        exit(1);
    }

    // check there is at least one element per block
    assert(min_rays >= nb * max_procs);

    if (rank == 0)
        fprintf(stderr, "min_procs = %d min_rays = %d max_rays = %d nb = %d target_k = %d avg_elems = %d reduce_factor = %d\n",
                min_procs, min_rays, max_rays, nb, target_k, avg_elems, reduce_factor);
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
    int min_rays, max_rays;   // min, max number of rays per block
    int num_rays;             // current total, global number of rays per block
    int avg_elems;            // average number of elements per ray
    int reduce_factor;        // number of elements to compose into one in each round
    int rank, groupsize;      // MPI usual
    int min_procs;            // minimum number of processes
    int max_procs;            // maximum number of processes (groupsize of MPI_COMM_WORLD)

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &max_procs);

    GetArgs(argc, argv, min_procs, min_rays, max_rays, nblocks, target_k, avg_elems, reduce_factor);

    // data extents, unused
    Bounds domain;
    for(int i = 0; i < dim; i++)
    {
        domain.min[i] = 0.0;
        domain.max[i] = 1.0;
    }

    int num_runs = (int)((log2(max_procs / min_procs) + 1) *
            (log2(max_rays / min_rays) + 1));

    // timing
    double all_all_v_time[num_runs];
    double swap_time[num_runs];

    // data for MPI reduce
    // allocate once for largest run (max_rays and min_procs) and reuse for all runs
    int out_size = max_rays * avg_elems / min_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == max_procs - 1)
        out_size = max_rays * avg_elems - (min_procs - 1) * max_rays * avg_elems / min_procs;
    float *all_all_v_data = new float[out_size];
    float *in_data        = new float[max_rays * avg_elems];

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
        Decomposer                decomposer(dim, domain, assigner.nblocks());
        decomposer.decompose(world.rank(), assigner, create);

        // iterate over (total, global) number of rays
        num_rays = min_rays;
        while (num_rays <= max_rays)
        {
            // MPI alltoallv, only for one block per process
            if (tot_blocks == groupsize)
                MpiAlltoAllv(all_all_v_data, all_all_v_time, run, in_data, comm, num_rays, avg_elems);

            // DIY swap
            // initialize input data
            master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
                    { ResetBlock(b, cp, num_rays, avg_elems, reduce_factor, master); });

            DiySwap(swap_time, run, target_k, comm, decomposer, master, assigner);

            // debug: print the block
//             master.foreach([&](Block* b, const diy::Master::ProxyWithLink& cp)
//                     { PrintBlock(b, cp, avg_elems); });

            num_rays *= 2; // double the number of rays every time
            run++;

        } // ray iteration

        groupsize *= 2; // double the number of processes every time
        MPI_Comm_free(&comm);

    } // proc iteration

    // print results
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fflush(stderr);
    if (rank == 0)
        PrintResults(all_all_v_time, swap_time, min_procs, max_procs, min_rays, max_rays, avg_elems);

    // cleanup
    delete[] in_data;
    delete[] all_all_v_data;
    MPI_Finalize();

    return 0;
}

