//---------------------------------------------------------------------------
//
// creates two moab meshes, computes an analytical field on the first mesh
// and projects the solution onto the second mesh
// evaluates the error between the projected solution and the original field
// applied directly to the second mesh
//
// Tom Peterka
// Argonne National Laboratory
// 9700 S. Cass Ave.
// Argonne, IL 60439
// tpeterka@mcs.anl.gov
//
//--------------------------------------------------------------------------

#include <iostream>
#include <iomanip>
#include <sstream>
#include "mpi.h"
#include <stddef.h>
#include <sys/resource.h>

#include "MBCore.hpp"
#include "MBRange.hpp"
#include "MBTagConventions.hpp"
#include "moab/ParallelComm.hpp"
#include "Coupler.hpp"

#include <diy/master.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>

#ifdef BGQ
#include <spi/include/kernel/memory.h>
#endif

#include "include/mesh_gen.hpp"

// memory profiling
#define MEMORY

using namespace std;
using namespace moab;

// aggregate statistics
#define NUM_STATS 10                         // number of statistics in message
#define NUM_VALS 0                           // number of values in sum
#define SUM_SQ_ERR 1                         // sum of squared errors
#define WORST_SQ 2                           // worst case squared error
#define WORST_VAL 3                          // field value at worst error
#define WORST_REF 4                          // reference value at worst error
#define WORST_X 5                            // x coordinate of worst error
#define WORST_Y 6                            // y coordinate of worst error
#define WORST_Z 7                            // z coordinate of worst error
#define SUM_SQ_REF 8                         // sum of reference values squared
#define MAX_REF 9                            // max ref value (value range assuming min = 0.0)

// field type
enum field_type
{
    VERTEX_FIELD,
    ELEMENT_FIELD,
};

// timing
enum
{
    TOT_TIME,
    INSTANT_TIME,
    POINTLOC_TIME,
    INTERP_TIME,
    GNORM_TIME,
    SNORM_TIME,
    EXCH_TIME,
    COUPLE_TIME,
    MESHGEN_TIME,
    PROJECT_TIME,                            // either vertex or element projection
    TOT_PROJECT_TIME,                        // sum of vertex plus element projection
    MAX_TIMES,
};

//
// starts / stops timing
// (does a barrier)
//
void GetTiming(int start,                    // index of timer to start (-1 if not used)
               int stop,                     // index of timer to stop (-1 if not used)
               double* times,                // time values
               MPI_Comm comm,                // MPI communicator
               bool add = false)             // add the time to the previous value
{
    static double temp_times[MAX_TIMES];

    if (start < 0 && stop < 0)
    {
        for (int i = 0; i < MAX_TIMES; i++)
            times[i] = 0.0;
    }

    MPI_Barrier(comm);

    if (start >= 0)
    {
        if (add)
            temp_times[start] = MPI_Wtime();
        else
            times[start] = MPI_Wtime();
    }
    if (stop >= 0)
    {
        if (add)
            times[stop] += (MPI_Wtime() - temp_times[stop]);
        else
            times[stop] = MPI_Wtime() - times[stop];
    }
}

//
// memory profile, prints max reseident usage
//
void GetMem(int breakpoint,                  // breakpoint number
            MPI_Comm comm)                   // communicator
{
    // quite compiler warnings in case MEMORY is not defined
    breakpoint = breakpoint;

#ifdef MEMORY

    int rank;
    MPI_Comm_rank(comm, &rank);

#ifdef BGQ

    uint64_t shared, persist, heapavail, stackavail,
        stack, heap, heapmax, guard, mmap;

    // we're only interested in max heap size
    // (same as max resident size, high water mark)
    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPMAX, &heapmax);

    // some examples of other memory usage we could get if we wanted it
    // note that stack and heap both count the total of both, use one or the other
    Kernel_GetMemorySize(KERNEL_MEMSIZE_SHARED, &shared);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_PERSIST, &persist);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_STACKAVAIL, &stackavail);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_STACK, &stack);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &heap);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_GUARD, &guard);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_MMAP, &mmap);

    int to_mb = 1024 * 1024;
    double heap_mem = double(heapmax) / to_mb;
    double max_heap_mem;
    MPI_Reduce(&heap_mem, &max_heap_mem, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0)
        fprintf(stderr, "%d: BGQ max memory = %.0lf MB\n", breakpoint, max_heap_mem);

#else

    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);

#ifdef __APPLE__
    const int to_mb = 1048576;
#else
    const int to_mb = 1024;
#endif

    float res = r_usage.ru_maxrss;
    float mem = res / (float)to_mb;
    float max_mem;
    MPI_Reduce(&mem, &max_mem, 1, MPI_FLOAT, MPI_MAX, 0, comm);
    if (rank == 0)
        fprintf(stderr, "%d: max memory = %0.1f MB\n", breakpoint, max_mem);

#endif // BGQ

#endif // MEMORY
}

// prints total and per iteration times
// resets point location, interpolation, and tag exchange times for reuse
//
void PrintTimes(double* times,               // all times
                int num_iter)                // number of iterations
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        fprintf(stderr, "------------------------------------------------------\n");
        fprintf(stderr, "total time for (%d) iterations: %.3lf s = \n", num_iter, times[PROJECT_TIME]);
        fprintf(stderr, "%.3lf s pointloc + (%d) * %.3lf s interp + (%d) * %.3lf s tag_exch\n",
                times[POINTLOC_TIME], num_iter, times[INTERP_TIME] / num_iter,
                num_iter, times[EXCH_TIME] / num_iter);
    }
    times[POINTLOC_TIME] = 0.0;
    times[INTERP_TIME] = 0.0;
    times[EXCH_TIME] = 0.0;
}

//
// custom reduction for aggregate statistics
//
void AggregateCouplingStats(void *in,
                            void *inout,
                            int *len,
                            MPI_Datatype *type)
{
    // quiet complier warnings aobut unused variables
    type = type;
    len = len;

    ((double *)inout)[NUM_VALS] += ((double *)in)[NUM_VALS];
    ((double *)inout)[SUM_SQ_ERR] += ((double *)in)[SUM_SQ_ERR];
    ((double *)inout)[SUM_SQ_REF] += ((double *)in)[SUM_SQ_REF];
    if (((double *)in)[WORST_SQ] > ((double *)inout)[WORST_SQ])
    {
        ((double *)inout)[WORST_SQ]  = ((double *)in)[WORST_SQ];
        ((double *)inout)[WORST_VAL] = ((double *)in)[WORST_VAL];
        ((double *)inout)[WORST_REF] = ((double *)in)[WORST_REF];
        ((double *)inout)[WORST_X]   = ((double *)in)[WORST_X];
        ((double *)inout)[WORST_Y]   = ((double *)in)[WORST_Y];
        ((double *)inout)[WORST_Z]   = ((double *)in)[WORST_Z];
    }
    if (((double *)in)[MAX_REF] > ((double *)inout)[MAX_REF])
        ((double *)inout)[MAX_REF] = ((double *)in)[MAX_REF];
}

//
// prints mesh statistics (for debugging)
//
void PrintMeshStats(Interface *mbint,        // moab interface
                    EntityHandle *mesh_set,  // moab mesh set
                    ParallelComm *mbpc)      // moab parallel communicator
{
    Range range;
    ErrorCode rval;
    static int mesh_num = 0;                 // counts how many time this function is called
    float loc_verts = 0.0;                   // num local verts (fractional for shared verts)
    float glo_verts;                         // global number of verts
    float loc_share_verts = 0.0;             // num local shared verts (fractional)
    float glo_share_verts;                   // global number of shared verts
    int loc_cells, glo_cells;                // local and global number of cells (no sharing)
    int rank;

    MPI_Comm_rank(mbpc->comm(), &rank);

    // get local quantities
    range.clear();
    rval = mbint->get_entities_by_dimension(*mesh_set, 0, range); ERR;

    // compute fractional contribution of shared vertices attributed to this proc
    int ps[MAX_SHARING_PROCS];               // sharing procs for a vert
    EntityHandle hs[MAX_SHARING_PROCS];      // handles of shared vert on sharing procs
    int num_ps = 0;                          // number of sharing procs, returned by moab
    unsigned char pstat;                     // pstatus, returned by moab
    for (Range::iterator verts_it = range.begin(); verts_it != range.end();
         verts_it++)
    {
        rval = mbpc->get_sharing_data(*verts_it, ps, hs, pstat, num_ps); ERR;
        if (num_ps == 0)
            loc_verts++;
        else if (num_ps == 1)                // when 2 procs, moab lists only one (the other)
        {
            loc_verts += 0.5;
            loc_share_verts += 0.5;
        }
        else
        {
            loc_verts += (1.0 / (float)num_ps);
            loc_share_verts += (1.0 / (float)num_ps);
        }
        // debug
        // if (rank == 0) {
        // fprintf(stderr, "num_ps = %d: ", num_ps);
        // for (int i = 0; i < num_ps; i++)
        // 	fprintf(stderr, "%d ", ps[i]);
        // fprintf(stderr, "\n");
        // fprintf(stderr, "loc_verts = %.3f\n", loc_verts);
        // }
    }

    range.clear();
    rval = mbint->get_entities_by_dimension(*mesh_set, 3, range); ERR;
    loc_cells = (int)range.size();

    // add totals for global quantities
    MPI_Reduce(&loc_verts, &glo_verts, 1, MPI_FLOAT, MPI_SUM, 0,
               mbpc->comm());
    MPI_Reduce(&loc_share_verts, &glo_share_verts, 1, MPI_FLOAT, MPI_SUM, 0,
               mbpc->comm());
    MPI_Reduce(&loc_cells, &glo_cells, 1, MPI_INT, MPI_SUM, 0, mbpc->comm());

    // report results
    if (rank == 0)
    {
        fprintf(stderr, "----------------- Mesh %d statistics -----------------\n",
                mesh_num);
        fprintf(stderr, "Total number of verts = %.0f of which %.0f "
                "are shared\n", glo_verts, glo_share_verts);
        fprintf(stderr, "Total number of cells = %d\n", glo_cells);
        fprintf(stderr, "------------------------------------------------------\n");
    }

    mesh_num = (mesh_num + 1) % 2;
}

//
// return a value for the field position (simple magnitude)
//
double PhysField(double x,
                 double y,
                 double z,
                 double factor)
{
    return factor * sqrt(x * x + y * y + z * z);
}

//
// add a value to each element in the field
//
void PutElementField(Interface *mbi,
                     EntityHandle eh,
                     const char *tagname,
		     double factor)
{
    Range elems;
    ErrorCode rval;
    const double defVal = 0.;
    Tag fieldTag;

    rval = mbi->get_entities_by_dimension(eh, 3, elems); ERR;
    rval = mbi->tag_get_handle(tagname, 1, MB_TYPE_DOUBLE, fieldTag,
                               MB_TAG_DENSE|MB_TAG_CREAT, &defVal); ERR;

    for(int i = 0; i< (int)elems.size(); i++)
    {
        EntityHandle elem = elems[i];
        double pos[3];
        rval = mbi->get_coords(&elem, 1, pos); ERR;
        double field_value =  PhysField(pos[0], pos[1], pos[2], factor);
        rval = mbi->tag_set_data(fieldTag, &elem, 1, &field_value); ERR;

        // debug
        //     fprintf(stderr, "setting element %d: f(%.3lf, %.3lf, %.3lf) = %.3lf\n",
        // 	    i, x, y, z, fieldValue);
    }
}

//
// gets the element field
//
void GetElementField(Interface *mbi,
                     EntityHandle eh,
                     const char *tagname,
		     double factor,
                     MPI_Comm comm,
                     double *glo_stats,
		     bool debug)
{
    Range elems;
    double *field_values;

    mbi->get_entities_by_dimension(eh, 3, elems);

    Tag field_tag;
    int count;
    mbi->tag_get_handle(tagname, 1, MB_TYPE_DOUBLE, field_tag);
    mbi->tag_iterate(field_tag, elems.begin(), elems.end(), count,
                     (void *&)field_values);
    assert(count == (int)elems.size());

    // init stats
    double sq_err;                           // squared error between field value and ref value
    double stats[NUM_STATS];                 // local statistics
    stats[SUM_SQ_ERR] = 0.0;
    stats[SUM_SQ_REF] = 0.0;
    stats[WORST_SQ] = 0.0;
    stats[NUM_VALS] = (int)elems.size();
    stats[MAX_REF] = 0.0;

    for(int i = 0; i < (int)elems.size(); i++)
    {
        EntityHandle elem = elems[i];
        double pos[3];
        mbi->get_coords(&elem, 1, pos);
        double ref_value = PhysField(pos[0], pos[1], pos[2], factor);

        // update stats
        double sq_err;                       // squared error between field value and ref value
        sq_err = (field_values[i] - ref_value) * (field_values[i] - ref_value);
        stats[SUM_SQ_ERR] += sq_err;
        stats[SUM_SQ_REF] += (ref_value * ref_value);
        if (sq_err > stats[WORST_SQ])
        {
            stats[WORST_SQ] = sq_err;
            stats[WORST_VAL] = field_values[i];
            stats[WORST_REF] = ref_value;
            stats[WORST_X] = pos[0];
            stats[WORST_Y] = pos[1];
            stats[WORST_Z] = pos[2];
        }
        if (ref_value > stats[MAX_REF])
            stats[MAX_REF] = ref_value;

        // debug
        if (debug)
            fprintf(stderr, "element %d: f(%.3lf, %.3lf, %.3lf) = %.3lf "
                    "ref_value = %.3lf sq_error = %.3lf\n",
                    i, pos[0], pos[1], pos[2], field_values[i], ref_value, sq_err);
    }

    // aggregate stats
    MPI_Op op;                               // custom reduction operator for stats
    MPI_Op_create(&AggregateCouplingStats, 1, &op);
    MPI_Reduce(stats, glo_stats, NUM_STATS, MPI_DOUBLE, op, 0, comm);
    MPI_Op_free(&op);
}

//
// add a value to each vertex in the field
//
void PutVertexField(Interface *mbi,
                    EntityHandle eh,
                    const char *tagname,
		    double factor)
{
    Range verts;

    mbi->get_entities_by_type(eh, MBVERTEX, verts);

    const double defVal = 0.;
    Tag fieldTag;
    mbi->tag_get_handle(tagname, 1, MB_TYPE_DOUBLE, fieldTag,
                        MB_TAG_DENSE|MB_TAG_CREAT, &defVal);

    int numVerts = verts.size();

    for(int i = 0; i < numVerts; i++)
    {
        EntityHandle vert = verts[i];
        double pos[3];
        mbi->get_coords(&vert, 1, pos);
        double fieldValue =  PhysField(pos[0], pos[1], pos[2], factor);
        mbi->tag_set_data(fieldTag, &vert, 1, &fieldValue);

        // debug
        //     fprintf(stderr, "setting element %d: f(%.3lf, %.3lf, %.3lf) = %.3lf\n",
        // 	    i, pos[0], pos[1], pos[2], fieldValue);
    }
}

//
// gets the vertex field
//
void GetVertexField(Interface *mbi,
                    EntityHandle eh,
                    const char *tagname,
                    double factor,
                    MPI_Comm comm,
                    double *glo_stats,
                    bool debug)
{
    Range verts;
    double *field_values;

    mbi->get_entities_by_type(eh, MBVERTEX, verts);

    Tag field_tag;
    int count;
    mbi->tag_get_handle(tagname, 1, MB_TYPE_DOUBLE, field_tag);
    mbi->tag_iterate(field_tag, verts.begin(), verts.end(), count, (void *&)field_values);
    assert(count == (int)verts.size());

    // init stats
    double sq_err;                           // squared error between field value and ref value
    double stats[NUM_STATS];                 // local statistics
    stats[SUM_SQ_ERR] = 0.0;
    stats[SUM_SQ_REF] = 0.0;
    stats[WORST_SQ] = 0.0;
    stats[NUM_VALS] = (int)verts.size();
    stats[MAX_REF] = 0.0;

    for(int i = 0; i < (int)verts.size(); i++)
    {
        EntityHandle vert = verts[i];
        double pos[3];
        mbi->get_coords(&vert, 1, pos);
        double ref_value = PhysField(pos[0], pos[1], pos[2], factor);

        // update stats
        sq_err = (field_values[i] - ref_value) * (field_values[i] - ref_value);
        stats[SUM_SQ_ERR] += sq_err;
        stats[SUM_SQ_REF] += (ref_value * ref_value);
        if (sq_err > stats[WORST_SQ])
        {
            stats[WORST_SQ] = sq_err;
            stats[WORST_VAL] = field_values[i];
            stats[WORST_REF] = ref_value;
            stats[WORST_X] = pos[0];
            stats[WORST_Y] = pos[1];
            stats[WORST_Z] = pos[2];
        }
        if (ref_value > stats[MAX_REF])
            stats[MAX_REF] = ref_value;

        // debug
        if (debug)
            fprintf(stderr, "vertex %d: f(%.3lf, %.3lf, %.3lf) = %.3lf "
                    "ref_value = %.3lf sq_error = %.3lf\n",
                    i, pos[0], pos[1], pos[2], field_values[i], ref_value, sq_err);
    }

    // aggregate stats
    MPI_Op op;                               // custom reduction operator for stats
    MPI_Op_create(&AggregateCouplingStats, 1, &op);
    MPI_Reduce(stats, glo_stats, NUM_STATS, MPI_DOUBLE, op, 0, comm);
    MPI_Op_free(&op);
}

//
// prepares the meshes for coupling by decomposing source and target domains
// and creating meshes in situ
//
void PrepMeshes(int src_type,
                int trgt_type,
                int src_size,
                int trgt_size,
                int slab,
                Interface* mbi,
                vector<ParallelComm*> &pcs,
                EntityHandle *roots,
                double factor,
                double* times,
                vector< diy::RegularDecomposer<Bounds>* >& decomps,
                vector<diy::RoundRobinAssigner*>& assigners,
                bool debug = false)
{
    decomps.reserve(2);
    diy::RegularDecomposer<Bounds>::CoordinateVector ghost(3, 0);
    diy::RegularDecomposer<Bounds>::DivisionsVector  given0(3, 0);
    diy::RegularDecomposer<Bounds>::DivisionsVector  given1(3, 0);
    vector<bool>                                     share_face(3, true);
    vector<bool>                                     wrap(3, false);

    vector<diy::mpi::communicator> comms(2);
    comms[0] = pcs[0]->comm();
    comms[1] = pcs[1]->comm();

    int src_mesh_size[3]  = {src_size, src_size, src_size};      // source size
    int trgt_mesh_size[3] = {trgt_size, trgt_size, trgt_size};   // target size
    Bounds domain0, domain1;
    domain0.min[0] = domain0.min[1] = domain0.min[2] = 0;
    domain1.min[0] = domain1.min[1] = domain1.min[2] = 0;
    domain0.max[0] = domain0.max[1] = domain0.max[2] = src_size - 1;
    domain1.max[0] = domain1.max[1] = domain1.max[2] = trgt_size - 1;

    // decompose domains
    if (slab == 1)
    {
        // source blocks are cubes in all 3 directions
        // target blocks are bricks in y direction
        given1[1] = 1;
    }
    if (slab == 2)
    {
        // source blocks are cubes in all 3 directions
        // target blocks are slabs in z direction
        given1[0] = 1;
        given1[1] = 1;
    }
    if (slab == 3)
    {
        // source blocks are slabs in the z direction
        given0[0] = 1;
        given0[1] = 1;
        // target blocks are slabs in the x direction
        given1[1] = 1;
        given1[2] = 1;
    }

    // the following calls to asssigners are for 1 block per process
    assigners[0] = new diy::RoundRobinAssigner(comms[0].size(), comms[0].size());
    assigners[1] = new diy::RoundRobinAssigner(comms[1].size(), comms[1].size());
    decomps[0]   = new diy::RegularDecomposer<Bounds>(3, domain0, *(assigners[0]), share_face,
                                                    wrap, ghost, given0);
    decomps[1]   = new diy::RegularDecomposer<Bounds>(3, domain1, *(assigners[1]), share_face,
                                                    wrap, ghost, given1);

    // report the number of blocks in each dimension of each mesh
    if (comms[0].rank() == 0)
        fprintf(stderr, "Number of blocks in source = [%d %d %d] target = [%d %d %d]\n",
                decomps[0]->divisions[0], decomps[0]->divisions[1], decomps[0]->divisions[2],
                decomps[1]->divisions[0], decomps[1]->divisions[1], decomps[1]->divisions[2]);

    // create meshes in situ
    unsigned long root;
    GetTiming(MESHGEN_TIME, -1, times, pcs[0]->comm());
    for (int i = 0; i < 2; i++)
    {
        int *mesh_size = (i == 0 ? src_mesh_size : trgt_mesh_size);

        if (debug)
            pcs[i]->set_debug_verbosity(5);

        if ((i == 0 && src_type == 0) || (i == 1 && trgt_type == 0))
            hex_mesh_gen(mesh_size, mbi, &(roots[i]), pcs[i], decomps[i], assigners[i]);
        else
            tet_mesh_gen(mesh_size, mbi, &(roots[i]), pcs[i], decomps[i], assigners[i]);

    }
    GetTiming(-1, MESHGEN_TIME, times, pcs[0]->comm());

    // debug: print mesh stats
    for (int i = 0; i < 2; i++)
    {
        PrintMeshStats(mbi, &(roots[i]), pcs[i]);
        if (debug)                           // store the mesh to a file
        {
            char outfile[256];
            sprintf(outfile, "out%d-proc%d.vtk", i, comms[0].rank());
            mbi->write_mesh(outfile, &(roots[i]), 1);
        }
    }

    // add field to input mesh
    PutVertexField(mbi, roots[0], "vertex_field", factor);
    PutElementField(mbi, roots[0], "element_field", factor);
}

//
// initializes the projection from source to target
//
ErrorCode InitProjection(Interface *mbi,
                         vector<ParallelComm *> &pcs,
			 Coupler* &mbc_for,
                         Coupler* &mbc_rev,
                         double* times)
{
    GetTiming(INSTANT_TIME, -1, times, pcs[0]->comm());

    Range src_elems, tgt_elems;
    ErrorCode rval;
    rval = pcs[0]->get_part_entities(src_elems, 3); ERR;
    rval = pcs[1]->get_part_entities(tgt_elems, 3); ERR;

    // instantiate couplers for forward and reverse directions (src->tgt and tgt->src),
    // which also initializes the trees
    mbc_for = new Coupler(mbi, pcs[0], src_elems, 0);
    mbc_rev = new Coupler(mbi, pcs[1], tgt_elems, 1);

    GetTiming(-1, INSTANT_TIME, times, pcs[0]->comm());

    return MB_SUCCESS;
}

//
// step one of the moab coupler: locate points to be interpolated
//
ErrorCode LocatePoints(Interface *mbi,
                       Coupler *mbc,
                       Coupler::Method method,
                       EntityHandle *roots,
                       double* times,
                       double & toler,
                       int& numpts,
                       Range& tgt_elems,
                       Range& tgt_verts,
                       ParallelComm* tgt_pc,
                       int pass)
{
    ErrorCode rval;

    // initialize spectral elements, if they exist. TODO: turned off for now
    bool specSou=false, specTar = false;
    //   rval =  mbc.initialize_spectral_elements((EntityHandle)roots[0],
    //                                            (EntityHandle)roots[1], specSou, specTar);

    GetTiming(POINTLOC_TIME, -1, times, tgt_pc->comm(), pass);

    // get points from the target mesh to interpolate

    std::vector<double> vpos;                // this will have the positions we are interested in
    numpts = 0;

    if (!specTar) // usual case
    {
        Range tmp_verts;

        // first get all vertices adj to partition entities in target mesh
        rval = tgt_pc->get_part_entities(tgt_elems, 3); ERR;
        if (Coupler::CONSTANT == method)
            tgt_verts = tgt_elems;
        else
            rval = mbi->get_adjacencies(tgt_elems, 0, false, tgt_verts, Interface::UNION); ERR;

        // then get non-owned verts and subtract

        // forward direction
        rval = tgt_pc->get_pstatus_entities(0, PSTATUS_NOT_OWNED, tmp_verts); ERR;
        tgt_verts = subtract(tgt_verts, tmp_verts);
        // get position of these entities; these are the target points
        numpts = (int)tgt_verts.size();
        vpos.resize(3 * tgt_verts.size());
        rval = mbi->get_coords(tgt_verts, &vpos[0]); ERR;
    }

    else // spectral case
    {
        // for spectral target, we want values interpolated on the GL positions; for each element,
        // get the GL points, and construct CartVect
        rval = tgt_pc->get_part_entities(tgt_elems, 3); ERR;
        rval = mbc->get_gl_points_on_elements(tgt_elems, vpos, numpts); ERR;
    }

    // locate those points in the source and target mesh
    rval = mbc->locate_points(&vpos[0], numpts, 0, toler); ERR;

    GetTiming(-1, POINTLOC_TIME, times, tgt_pc->comm(), pass);
    return MB_SUCCESS;
}

//
// step two of the moab coupler: interpolate points
//
ErrorCode Interpolate(Interface *mbi,
                      Coupler *mbc,
                      Coupler::Method method,
                      std::string &interpTag,
                      std::string &gNormTag,
                      std::string &ssNormTag,
                      std::vector<const char *> &ssTagNames,
                      std::vector<const char *> &ssTagValues,
                      EntityHandle *roots,
                      ParallelComm* tgt_pc,
                      double* times,
                      int num_iter,
                      int numpts,
                      Range& tgt_elems,
                      Range& tgt_verts,
                      int pass)
{
    ErrorCode rval;
    std::vector<double> field(numpts);

    assert(method >= Coupler::CONSTANT && method <= Coupler::SPECTRAL);

    // initialize spectral elements, if they exist. TODO: turned off for now
    bool specSou = false;
    bool specTar = false;

    //   rval =  mbc.initialize_spectral_elements((EntityHandle)roots[0],
    //                                            (EntityHandle)roots[1], specSou, specTar);

    // emulate some number of iterations to converge to solution
    // in reality, nothing in the solution changes in this example
    // but this makes the timing results better approximate reality
    for (int i = 0; i < num_iter; i++)
    {
        // interpolate
        GetTiming(INTERP_TIME, -1, times, tgt_pc->comm(), true);
        rval = mbc->interpolate(method, interpTag, &field[0]); ERR;
        GetTiming(GNORM_TIME, INTERP_TIME, times, tgt_pc->comm(), true);

        // optional global normalization
        if (!gNormTag.empty())
            rval = (ErrorCode)mbc->normalize_mesh(roots[0], gNormTag.c_str(),
                                                  Coupler::VOLUME, 4); ERR;
        GetTiming(SNORM_TIME, GNORM_TIME, times, tgt_pc->comm(), true);

        // optional subset normalization
        if (!ssNormTag.empty())
            rval = mbc->normalize_subset(roots[0], ssNormTag.c_str(), &ssTagNames[0],
                                         ssTagNames.size(), &ssTagValues[0],
                                         Coupler::VOLUME, 4); ERR;
        GetTiming(-1, SNORM_TIME, times, tgt_pc->comm(), true);

        // set field values as tag on target vertices
        if (specSou)                         // spectral source
        {
            // create a new tag for the values on the target and source
            Tag tag;
            std::string newtag = interpTag +"_TAR";
            rval = mbi->tag_get_handle(newtag.c_str(), 1, MB_TYPE_DOUBLE,
                                       tag, MB_TAG_CREAT|MB_TAG_DENSE); ERR;
            rval = mbi->tag_set_data(tag, tgt_verts, &field[0]); ERR;
        }

        else // nonspectral source
        {
            if (!specTar)                    // nonspectral target
            {
                // use original tag
                Tag tag;
                rval = mbi->tag_get_handle(interpTag.c_str(), 1, MB_TYPE_DOUBLE, tag); ERR;
                rval = mbi->tag_set_data(tag, tgt_verts, &field[0]); ERR;
            }

            else // spectral target
            {
                // we have the field values computed at each GL points, on each element
                // in the target mesh
                // we need to create a new tag, on elements, of size _ntot, to hold
                // all those values.
                // so it turns out we need ntot. maybe we can compute it from the
                // number of values computed, divided by number of elements
                int ntot = numpts / tgt_elems.size();
                Tag tag;
                std::string newtag = interpTag +"_TAR";
                rval = mbi->tag_get_handle(newtag.c_str(), ntot, MB_TYPE_DOUBLE,
                                           tag, MB_TAG_CREAT|MB_TAG_DENSE); ERR;
                rval = mbi->tag_set_data(tag, tgt_elems, &field[0]); ERR;
            }
        }

        // communicate to processor the values found
        GetTiming(EXCH_TIME, -1, times, tgt_pc->comm(), true);
        rval = tgt_pc->exchange_tags(interpTag.c_str(), tgt_verts); ERR;
        GetTiming(-1, EXCH_TIME, times, tgt_pc->comm(), true);
    } // iterations until convergence

    // TODO: how is the mesh returned to the caller? roots?
    // roots is not touched except in normalization, which we are not using

    return MB_SUCCESS;
}

//
// runs the moab coupler to do the projection from source to target
//
ErrorCode Project(Interface *mbi,
                  Coupler *mbc,
                  Coupler::Method method,
                  std::string &interpTag,
                  std::string &gNormTag,
                  std::string &ssNormTag,
		  std::vector<const char *> &ssTagNames,
                  std::vector<const char *> &ssTagValues,
		  EntityHandle *roots,
                  ParallelComm* tgt_pc,
                  double* times,
		  double & toler,
                  int num_iter,
                  int pass)
{
    ErrorCode rval;
    Range tgt_elems, tgt_verts;              // source and target elements and vertices
    int numpts;                              // number of points of interest

    rval = LocatePoints(mbi, mbc, method, roots, times, toler, numpts, tgt_elems,
                        tgt_verts, tgt_pc, pass); ERR;

    rval = Interpolate(mbi, mbc ,method, interpTag, gNormTag, ssNormTag, ssTagNames, ssTagValues,
                       roots, tgt_pc, times, num_iter, numpts, tgt_elems, tgt_verts, pass); ERR;

    // TODO: how is the mesh returned to the caller? roots?

    return MB_SUCCESS;
}

//
// prints aggregate stats and timing info
//
void PrintCouplingStats(char *title,
                        double *glo_stats)
{
    fprintf(stderr, "------------------------------------------------------\n");

    fprintf(stderr, "%s\n", title);
    fprintf(stderr, "Maximum error = %.4e \n", sqrt(glo_stats[WORST_SQ]));
    double rms_err = sqrt(glo_stats[SUM_SQ_ERR] / glo_stats[NUM_VALS]);
    fprintf(stderr, "sum sq error = %.4e\n", glo_stats[SUM_SQ_ERR]);
    double rms_ref = sqrt(glo_stats[SUM_SQ_REF] / glo_stats[NUM_VALS]);
    fprintf(stderr, "For the entire field, RMS error = %.4e\n", rms_err);

    fprintf(stderr, "------------------------------------------------------\n");
}

//
// summarizes resulting field (vertex or element) by comparing to analytical solution and
// computing (and printing) error statistics
//
void SummarizeField(enum field_type ft,
                    EntityHandle *roots,
                    double factor,
                    Interface *mbi,
                    vector<ParallelComm*>& pcs)
{
    double glo_stats[NUM_STATS];             // aggregate statistics, valid only at root
    int rank;                                // MPI usual

    MPI_Comm_rank(pcs[0]->comm(), &rank);

    if (ft == VERTEX_FIELD)                  // vertex field
    {
        // forward
        GetVertexField(mbi, roots[1], "vertex_field", factor, pcs[1]->comm(),
                       glo_stats, false);
        if (rank == 0)
            PrintCouplingStats((char *)"forward coupled vertex field stats", glo_stats);

        // reverse
        GetVertexField(mbi, roots[0], "vertex_field", factor, pcs[0]->comm(),
                       glo_stats, false);
        if (rank == 0)
            PrintCouplingStats((char *)"reverse coupled vertex field stats", glo_stats);
    } else                                   // element field
    {
        // forward
        GetElementField(mbi, roots[1], "element_field", factor, pcs[1]->comm(),
                        glo_stats, false);
        if (rank == 0)
            PrintCouplingStats((char *)"forward coupled element field stats", glo_stats);


        // reverse
        GetElementField(mbi, roots[0], "element_field", factor, pcs[0]->comm(),
                        glo_stats, false);
        if (rank == 0)
            PrintCouplingStats((char *)"reverse coupled element field stats", glo_stats);
    }
}

//
// project vertex field from source to target and target to source
//
void ProjectField(Interface* mbi,
                  Coupler* mbc_for,
                  Coupler* mbc_rev,
                  EntityHandle* roots,
                  std::vector<ParallelComm *>& pcs,
                  double* times,
                  int num_iter,
                  double factor,
                  field_type type)
{
    ErrorCode rval;                          // moab return value
    string interpTag;
    Coupler::Method method;
    string gNormTag = "";
    string ssNormTag = "";
    std::vector<const char *> ssTagNames, ssTagValues;
    double toler = 5.0e-10;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (type == VERTEX_FIELD)
    {
        if (rank == 0)
            fprintf(stderr, "coupling vertex field for %d iterations...\n", num_iter);
        interpTag = "vertex_field";
        method = Coupler::LINEAR_FE;
    }
    else
    {
        if (rank == 0)
            fprintf(stderr, "coupling element field for %d iterations...\n", num_iter);
        interpTag = "element_field";
        // the only method that makes sense for elements is constant
        // because element tags are one per element (not per vertex)
        method = Coupler::CONSTANT;
    }
    GetTiming(PROJECT_TIME, -1, times, pcs[0]->comm());
    // forward direction
    rval = Project(mbi, mbc_for, method, interpTag, gNormTag, ssNormTag,
                   ssTagNames, ssTagValues, roots, pcs[1], times, toler, num_iter, 0); ERR;
    // reverse direction
    rval = Project(mbi, mbc_rev, method, interpTag, gNormTag, ssNormTag,
                   ssTagNames, ssTagValues, roots, pcs[0], times, toler, num_iter, 1); ERR;
    GetTiming(-1, PROJECT_TIME, times, pcs[0]->comm());
    PrintTimes(times, num_iter);
    // TODO: why is reverse direction error 0 for element field
    SummarizeField(type, roots, factor, mbi, pcs);
}

//
// couples two meshes
//
void Coupling(MPI_Comm *mpi_comms,
              int src_size,
              int trgt_size,
              int src_type,
              int trgt_type,
              int num_iter,
              int slab,
              double* times)
{
    int err;                                 // return value
    int rank;                                // MPI usual
    double factor = 1.0;                     // field value scaling factor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // init DIY
    vector< diy::RegularDecomposer<Bounds>* > decomps(2);
    vector<diy::RoundRobinAssigner*> assigners(2);

    // create MOAB instance
    Interface *mbi = new Core();

    std::vector<ParallelComm *> pcs(2);      // communicators for meshes
    EntityHandle roots[2];
    for (int i = 0; i < 2; i++)
    {
        pcs[i] = new ParallelComm(mbi, mpi_comms[i]);
        mbi->create_meshset(MESHSET_SET, roots[i]);
    }

    GetMem(1, mpi_comms[0]);                 // before any real work happens

    // decompose domain, generate meshes
    PrepMeshes(src_type, trgt_type, src_size, trgt_size, slab, mbi, pcs, roots, factor, times,
               decomps, assigners);

    GetMem(2, mpi_comms[0]);                 // after meshes are created

    // init the projection
    Coupler *mbc_for, *mbc_rev; // moab forward and reverse coupler objects
    InitProjection(mbi, pcs, mbc_for, mbc_rev, times);
    MPI_Barrier(mpi_comms[0]);
    if (rank == 0)
    {
        fprintf(stderr, "initialization time = %.3lf s\n", times[INSTANT_TIME]);
        fprintf(stderr, "------------------------------------------------------\n");
    }

    GetMem(3, mpi_comms[0]);                 // after projection initialized

    GetTiming(TOT_PROJECT_TIME, -1, times, pcs[0]->comm());

    // project vertex field from source to target and target to source
    ProjectField(mbi, mbc_for, mbc_rev, roots, pcs, times, num_iter, factor, VERTEX_FIELD);
    GetMem(4, mpi_comms[0]);                 // after projecting vertex field

    // project element field from source to target and target to source
    ProjectField(mbi, mbc_for, mbc_rev, roots, pcs, times, num_iter, factor, ELEMENT_FIELD);
    GetMem(5, mpi_comms[0]);                 // after projecting element field

    GetTiming(-1, TOT_PROJECT_TIME, times, pcs[0]->comm());

    // cleanup
    delete mbc_for;
    delete mbc_rev;
    for (int i = 0; i < 2; i++)
    {
        delete pcs[i];
        delete decomps[i];
        delete assigners[i];
    }
    delete mbi;
}

//
// prints final times
//
void PrintFinalTimes(double* times)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        fprintf(stderr, "-------------total time = %.3lf s --------------------\n",
                times[COUPLE_TIME]);
        fprintf(stderr, "meshgen time %.3lf + initlzn time %.3lf + projctn time %.3lf\n",
                times[MESHGEN_TIME], times[INSTANT_TIME], times[TOT_PROJECT_TIME]);
    }
}

//
// print mesh types and sizes
//
void PrintMeshSizes(int src_type,
                    int src_size,
                    int trgt_type,
                    int trgt_size)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        char src_str[256], trgt_str[256];
        strcpy(src_str, src_type ? "tet" : "hex");
        strcpy(trgt_str, trgt_type ? "tet" : "hex");
        fprintf(stderr, "\n");
        fprintf(stderr, "[%d x %d x %d] %s -> [%d x %d x %d] %s\n\n",
                src_size, src_size, src_size, src_str, trgt_size, trgt_size, trgt_size, trgt_str);
    }
}

//
// parse arguments
//
void ParseArgs(int argc,
               char **argv,
               int *min_procs,               // minimum number of mpi procs
               int *max_procs,               // maximum number of mpi procs
               int *src_type,                // source mesh type (0 = hex 1 = tet)
               int *min_src_size,            // min source mesh size per side (size x size x size)
               int *max_src_size,            // max source mesh size per side (size x size x size)
               int *trgt_type,               // target mesh type (0 = hex 1 = tet)
               int *min_trgt_size,           // min target mesh size per side (size x size x size)
               int *max_trgt_size,           // max target mesh size per side (size x size x size)
               int *num_iter,                // number of iterations (simulating convergence)
               int *slab)                    // "slabbiness" of blocking (0-3; best to worst case)
{
    char src_str[256], trgt_str[256]; // string versions of src and trgt types

    int rank, groupsize; // MPI usual
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &groupsize);

    assert(argc >= 9);

    *min_procs = atoi(argv[1]);
    *max_procs = groupsize;

    if (argv[2][0] == 'h' || argv[2][0] == 'H')
    {
        *src_type = 0;
        strcpy(src_str, "hex");
    }
    else
    {
        *src_type = 1;
        strcpy(src_str, "tet");
    }
    *min_src_size = atoi(argv[3]);
    *max_src_size = atoi(argv[4]);
    if (argv[5][0] == 'h' || argv[5][0] == 'H')
    {
        *trgt_type = 0;
        strcpy(trgt_str, "hex");
    }
    else
    {
        *trgt_type = 1;
        strcpy(trgt_str, "tet");
    }
    *min_trgt_size = atoi(argv[6]);
    *max_trgt_size = *min_trgt_size * (*max_src_size / *min_src_size);
    *num_iter = atoi(argv[7]);
    *slab = atoi(argv[8]);
    assert(*slab >= 0 && *slab <= 3);

    if (rank == 0)
    {
        fprintf(stderr, "min_procs = %d max_procs = %d; "
                "source type = %s; "
                "source size = from [%d x %d x %d] to [%d x %d x %d] "
                "regular grid points (not cells); "
                "target type = %s; "
                "target size = from [%d x %d x %d] to [%d x %d x %d] "
                "regular grid points (not cells); "
                "number of iterations = %d; "
                "slab = %d\n",
                *min_procs, *max_procs,
                src_str, *min_src_size, *min_src_size, *min_src_size,
                *max_src_size, *max_src_size, *max_src_size,
                trgt_str, *min_trgt_size, *min_trgt_size, *min_trgt_size,
                *max_trgt_size, *max_trgt_size, *max_trgt_size, *num_iter, *slab);

        // check min_procs max_procs relationships
        double lp2 = log2(*max_procs / *min_procs);         // log base 2 of procs
        double lp8 = log2(*max_procs / *min_procs) / 3;     // log base 8 of procs
        double lm2 = log2(*max_src_size / *min_src_size);   // log base 2 of mesh
        assert(*max_procs >= *min_procs);
        if (*min_src_size == *max_src_size && lp2 != floor(lp2))
            fprintf(stderr, "Warning: max_procs is not a power of 2 larger than "
                    "min_procs: ending at largest power of 2 factor of min_procs\n");
        if (*min_src_size != *max_src_size && lp8 != floor(lp8))
            fprintf(stderr, "Warning: max_procs is not a power of 8 larger than "
                    "min_procs: ending at largest power of 8 factor of min_procs\n");
        if (*min_src_size != *max_src_size && lm2 != lp8)
            fprintf(stderr, "Warning: log8 of max_procs / min_procs does not equal "
                    "log2 of max_src_size / min_src_size. Weak scaling will not "
                    "have the correct number of runs.\n");
    }
}

int main(int argc,
         char** argv)
{
    MPI_Comm comms[2]; // MPI communicators for two meshes
    int src_type, trgt_type; // source and target mesh types (0 = hex, 1 = tet)
    int min_src_size, max_src_size; // min, max source mesh size per side
    int min_trgt_size, max_trgt_size; // min, max target mesh size per side
    int src_size, trgt_size; // current source and target mesh size per side
    int num_iter; // number of coupling iterations to simulate convergence
    int min_procs, max_procs; // number of MPI processes
    int rank, groupsize; // MPI usual for current communicator
    int slab; // "slabbiness" of blocking (0-3; best to worst case)

    // init
    MPI_Init(&argc, &argv);
    double times[MAX_TIMES];
    GetTiming(-1, -1, times, MPI_COMM_WORLD);

    // parse arguments
    ParseArgs(argc, argv, &min_procs, &max_procs, &src_type, &min_src_size, &max_src_size,
              &trgt_type, &min_trgt_size, &max_trgt_size, &num_iter, &slab);

    // iterate over process counts and mesh size; 4 modes are possible:
    // 0. process count and mesh size both constant   -> single condition test
    // 1. process count constant and mesh size varies -> problem complexity test
    // 2. process count varies and mesh size constant -> strong scaling test
    // 3. process count varies and mesh size varies   -> weak scaling test

    int run = 0; // run number
    groupsize = min_procs;
    // strong or weak scaling depending on whether mesh size is constant
    int groupsize_factor = (min_src_size == max_src_size ? 2 : 8);
    // init mesh sizes
    src_size = min_src_size;
    trgt_size = min_trgt_size;

    // processes
    while (groupsize <= max_procs)
    {
        // form a new communicator
        MPI_Comm comm;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_split(MPI_COMM_WORLD, (rank < groupsize), rank, &comm);
        if (rank >= groupsize)
        {
            MPI_Comm_free(&comm);
            groupsize *= groupsize_factor;
            if (src_size < max_src_size)
            {
                src_size *= 2;
                trgt_size *= 2;
            }
            continue;
        }
        MPI_Comm_rank(comm, &rank);
        comms[0] = comm; // source and target communicators, for now, are same
        comms[1] = comm;

        if (rank == 0)
        {
            fprintf(stderr, "\n");
            if (groupsize == 1)
                fprintf(stderr, "================== 1 process ===================\n\n");
            else
                fprintf(stderr, "================== %d processes ================\n\n",
                        groupsize);
        }

        // modes 0 and 1, process count constant
        if (min_procs == max_procs)
        {
            src_size = min_src_size;
            trgt_size = min_trgt_size;

            while (src_size <= max_src_size)      // iterate over mesh size
            {
                PrintMeshSizes(src_type, src_size, trgt_type, trgt_size);

                // couple the meshes
                GetTiming(COUPLE_TIME, -1, times, comms[0]);
                Coupling(comms, src_size, trgt_size, src_type, trgt_type, num_iter, slab, times);
                GetTiming(-1, COUPLE_TIME, times, comms[0]);
                if (rank == 0)
                    PrintFinalTimes(times);
                src_size *= 2;
                trgt_size *= 2;
            } // mesh size
        } // modes 0 and 1

        // modes 1 and 2, process count varies
        if (min_procs < max_procs)
        {
            PrintMeshSizes(src_type, src_size, trgt_type, trgt_size);

            // couple the meshes
            GetTiming(COUPLE_TIME, -1, times, comms[0]);
            Coupling(comms, src_size, trgt_size, src_type, trgt_type, num_iter, slab, times);
            GetTiming(-1, COUPLE_TIME, times, comms[0]);
            if (rank == 0)
                PrintFinalTimes(times);
            if (src_size < max_src_size)
            {
                src_size *= 2;
                trgt_size *= 2;
            }
        } // modes 1 and 2

        groupsize *= groupsize_factor;
        MPI_Comm_free(&comm);
    } // number of procs

    // cleanup
    MPI_Finalize();
}
