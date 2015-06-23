//---------------------------------------------------------------------------
//
// generates moab meshes on the fly
//
// Tom Peterka
// Argonne National Laboratory
// 9700 S. Cass Ave.
// Argonne, IL 60439
// tpeterka@mcs.anl.gov
//
//--------------------------------------------------------------------------

#include "include/mesh_gen.hpp"

using namespace std;
using namespace moab;

//
// generate a regular structured hex mesh
//
void hex_mesh_gen(int *mesh_size,            // mesh size (i,j,k) number of vertices in each dim
                  Interface *mbint,                               // moab interface instance
                  EntityHandle *mesh_set,                         // moab mesh set
		  ParallelComm *mbpc,                             // moab parallel communicator
                  diy::RegularDecomposer<Bounds>* decomp,         // diy decomposition
                  diy::Assigner* assign)                          // diy assignment
{
    create_hexes_and_verts(mesh_size, mbint, mesh_set, decomp, assign, mbpc);
    resolve_and_exchange(mbint, mesh_set, mbpc);
}
//
// generate a regular structured tet mesh
//
void tet_mesh_gen(int *mesh_size,            // mesh size (i,j,k) number of vertices in each dim
                  Interface *mbint,                                // moab interface instance
                  EntityHandle *mesh_set,                          // moab mesh set
		  ParallelComm *mbpc,                              // moab parallel communicator
                  diy::RegularDecomposer<Bounds>* decomp,          // diy decomposition
                  diy::Assigner* assign)                           // diy assignment
{
    create_tets_and_verts(mesh_size, mbint, mesh_set, decomp, assign, mbpc);
    resolve_and_exchange(mbint, mesh_set, mbpc);
}
//
// create hex cells and vertices
//
void create_hexes_and_verts(int *mesh_size,   // mesh size (i,j,k) number of vertices in each dim
                           Interface *mbint,                       // moab interface instance
                           EntityHandle *mesh_set,                 // moab mesh set
                           diy::RegularDecomposer<Bounds>* decomp, // diy decomposition
                           diy::Assigner* assign,                  // diy assignment
                           ParallelComm *mbpc)                     // moab communicator
{
    Core *mbcore = dynamic_cast<Core*>(mbint);
    EntityHandle vs, cs;
    ErrorCode rval;
    EntityHandle handle;
    vector<int> local_gids;
    int rank;
    MPI_Comm_rank(mbpc->comm(), &rank);
    assign->local_gids(rank, local_gids);
    int nblocks = local_gids.size();
    Bounds bounds[nblocks];
    int b; // local block number

    // get the read interface from moab
    ReadUtilIface *iface;
    rval = mbint->query_interface(iface); ERR;

    // block bounds
    for (int b = 0; b < nblocks; b++)
        decomp->fill_bounds(bounds[b], local_gids[b], false);

    // todo: multiple blocks per process not really supported yet in this example
    b = 0;

    // the following method is based on the example in
    // moab/examples/old/FileRead.cpp, using the ReadUtilIface class

    // allocate a block of vertex handles and store xyz’s into them
    // returns a starting handle for the node sequence
    vector<double*> arrays;
    EntityHandle startv;
    int num_verts =
        (bounds[b].max[0] - bounds[b].min[0] + 1) *
        (bounds[b].max[1] - bounds[b].min[1] + 1) *
        (bounds[b].max[2] - bounds[b].min[2] + 1);
    rval = iface->get_node_coords(3, num_verts, 0, startv, arrays); ERR;

    // populate vertex arrays
    // vertices normalized to be in the range [0.0 - 1.0]
    int n = 0;
    for (int k = bounds[b].min[2]; k <= bounds[b].max[2]; k++)
    {
        for (int j = bounds[b].min[1]; j <= bounds[b].max[1]; j++)
        {
            for (int i = bounds[b].min[0]; i <= bounds[b].max[0]; i++)
            {
                arrays[0][n] = double(i) / (mesh_size[0] - 1);
                arrays[1][n] = double(j) / (mesh_size[1] - 1);
                arrays[2][n] = double(k) / (mesh_size[2] - 1);

                // debug
                // 	fprintf(stderr, "[i,j,k] = [%d %d %d] vert[%d] = [%.2lf %.2lf %.2lf]\n",
                // 		i, j, k, n, arrays[0][n], arrays[1][n], arrays[2][n]);

                n++;
            }
        }
    }

    // allocate connectivity array
    EntityHandle startc; // handle for start of cells
    EntityHandle *starth; // handle for start of connectivity
    int num_hexes =
        (bounds[b].max[0] - bounds[b].min[0]) *
        (bounds[b].max[1] - bounds[b].min[1]) *
        (bounds[b].max[2] - bounds[b].min[2]);
    rval = iface->get_element_connect(num_hexes, 8, MBHEX, 0, startc, starth);

    // populate the connectivity array
    n = 0;
    int m = 0;
    for (int k = bounds[b].min[2]; k <= bounds[b].max[2]; k++)
    {
        for (int j = bounds[b].min[1]; j <= bounds[b].max[1]; j++)
        {
            for (int i = bounds[b].min[0]; i <= bounds[b].max[0]; i++)
            {
                if (i < bounds[b].max[0] && j < bounds[b].max[1] && k < bounds[b].max[2])
                {
                    int A, B, C, D, E, F, G, H; // hex verts according to my diagram
                    D = n;
                    C = D + 1;
                    H = D + bounds[b].max[0] - bounds[b].min[0] + 1;
                    G = H + 1;
                    A = D + (bounds[b].max[0] - bounds[b].min[0] + 1) *
                        (bounds[b].max[1] - bounds[b].min[1] + 1);
                    B = A + 1;
                    E = A + bounds[b].max[0] - bounds[b].min[0] + 1;
                    F = E + 1;

                    // hex ABCDEFGH
                    starth[m++] = startv + A;
                    starth[m++] = startv + B;
                    starth[m++] = startv + C;
                    starth[m++] = startv + D;
                    starth[m++] = startv + E;
                    starth[m++] = startv + F;
                    starth[m++] = startv + G;
                    starth[m++] = startv + H;

                }

                n++;

            }
        }
    }

    // add vertices and cells to the mesh set
    Range vRange(startv, startv + num_verts - 1); // vertex range
    Range cRange(startc, startc + num_hexes - 1); // cell range
    rval = mbint->add_entities(*mesh_set, vRange); ERR;
    rval = mbint->add_entities(*mesh_set, cRange); ERR;

    // check that long is indeed 8 bytes on this machine
    assert(sizeof(long) == 8);

    // set global ids
    long gid;
    Tag global_id_tag;
    rval = mbint->tag_get_handle("HANDLEID", 1, MB_TYPE_HANDLE,
                                 global_id_tag, MB_TAG_CREAT|MB_TAG_DENSE); ERR;

    // gids for vertices, starting at 1 by moab convention
    handle = startv;
    for (int k = bounds[b].min[2]; k < bounds[b].max[2] + 1; k++)
    {
        for (int j = bounds[b].min[1]; j < bounds[b].max[1] + 1; j++)
        {
            for (int i = bounds[b].min[0]; i < bounds[b].max[0] + 1; i++)
            {
                gid = (long)1 + (long)i + (long)j * (mesh_size[0]) +
                    (long)k * (mesh_size[0]) * (mesh_size[1]);
                //         fprintf(stderr, "i,j,k = [%d %d %d] gid = %ld\n", i, j, k, gid);
                rval = mbint->tag_set_data(global_id_tag, &handle, 1, &gid); ERR;
                handle++;
            }
        }
    }

    // gids for cells, starting at 1 by moab convention
    handle = startc;
    for (int k = bounds[b].min[2]; k < bounds[b].max[2]; k++)
    {
        for (int j = bounds[b].min[1]; j < bounds[b].max[1]; j++)
        {
            for (int i = bounds[b].min[0]; i < bounds[b].max[0]; i++)
            {
                gid = (long)1 + (long)i + (long)j * (mesh_size[0] - 1) +
                    (long)k * (mesh_size[0] - 1) * (mesh_size[1] - 1);
                // debug
                //        fprintf(stderr, "i,j,k = [%d %d %d] gid = %ld\n", i, j, k, gid);
                rval = mbint->tag_set_data(global_id_tag, &handle, 1, &gid); ERR;
                handle++;
            }
        }
    }

    // update adjacencies (needed by moab)
    rval = iface->update_adjacencies(startc, num_hexes, 8, starth); ERR;

    // cleanup
    rval = mbint->release_interface(iface); ERR;

}
//
// create tet cells and vertices
//
void create_tets_and_verts(int *mesh_size,  // mesh size (i,j,k) number of vertices in each dim
                          Interface *mbint,                       // moab interface instance
                          EntityHandle *mesh_set,                 // moab parallel communicator
                          diy::RegularDecomposer<Bounds>* decomp, // diy decomposition
                          diy::Assigner* assign,                  // diy assignment
                          ParallelComm *mbpc)                     // moab communicator
{
    Core *mbcore = dynamic_cast<Core*>(mbint);
    EntityHandle vs, cs;
    ErrorCode rval;
    EntityHandle handle;
    vector<int> local_gids;
    int rank;
    MPI_Comm_rank(mbpc->comm(), &rank);
    assign->local_gids(rank, local_gids);
    int nblocks = local_gids.size();
    Bounds bounds[nblocks];
    int b; // local block number

    // get the read interface from moab
    ReadUtilIface *iface;
    rval = mbint->query_interface(iface); ERR;

    // block bounds
    for (int b = 0; b < nblocks; b++)
        decomp->fill_bounds(bounds[b], local_gids[b], false);

    // debug
//     fprintf(stderr, "nblocks = %d\n", nblocks);
//     fprintf(stderr, "bounds min = [%d %d %d] max = [%d %d %d]\n",
//             bounds[0].min[0], bounds[0].min[1], bounds[0].min[2],
//             bounds[0].max[0], bounds[0].max[1], bounds[0].max[2]);

    // todo: multiple blocks per process not really supported yet in this example
    b = 0;

    // the following method is based on the example in
    // moab/examples/old/FileRead.cpp, using the ReadUtilIface class

    // allocate a block of vertex handles and store xyz’s into them
    // returns a starting handle for the node sequence
    vector<double*> arrays;
    EntityHandle startv;
    int num_verts =
        (bounds[b].max[0] - bounds[b].min[0] + 1) *
        (bounds[b].max[1] - bounds[b].min[1] + 1) *
        (bounds[b].max[2] - bounds[b].min[2] + 1);
    rval = iface->get_node_coords(3, num_verts, 0, startv, arrays); ERR;

    // populate vertex arrays
    // vertices normalized to be in the range [0.0 - 1.0]
    int n = 0;
    for(int k = bounds[b].min[2]; k <= bounds[b].max[2]; k++)
    {
        for(int j = bounds[b].min[1]; j <= bounds[b].max[1]; j++)
        {
            for(int i = bounds[b].min[0]; i <= bounds[b].max[0]; i++)
            {
                arrays[0][n] = double(i) / (mesh_size[0] - 1);
                arrays[1][n] = double(j) / (mesh_size[1] - 1);
                arrays[2][n] = double(k) / (mesh_size[2] - 1);

                // debug
//                 fprintf(stderr, "vert[%d] = [%.2lf %.2lf %.2lf]\n",
//                         n, arrays[0][n], arrays[1][n], arrays[2][n]);

                n++;
            }
        }
    }

    // allocate connectivity array
    EntityHandle startc; // handle for start of cells
    EntityHandle *starth; // handle for start of connectivity
    int num_tets = 6 *  // each hex cell will be converted to 6 tets
        (bounds[b].max[0] - bounds[b].min[0]) *
        (bounds[b].max[1] - bounds[b].min[1]) *
        (bounds[b].max[2] - bounds[b].min[2]);
    rval = iface->get_element_connect(num_tets, 4, MBTET, 0, startc, starth); ERR;

    // populate the connectivity array
    n = 0;
    int m = 0;
    for (int k = bounds[b].min[2]; k <= bounds[b].max[2]; k++)
    {
        for (int j = bounds[b].min[1]; j <= bounds[b].max[1]; j++)
        {
            for (int i = bounds[b].min[0]; i <= bounds[b].max[0]; i++)
            {
                if (i < bounds[b].max[0] && j < bounds[b].max[1] &&
                    k < bounds[b].max[2])
                {
                    int A, B, C, D, E, F, G, H; // hex verts according to my diagram
                    D = n;
                    C = D + 1;
                    H = D + bounds[b].max[0] - bounds[b].min[0] + 1;
                    G = H + 1;
                    A = D + (bounds[b].max[0] - bounds[b].min[0] + 1) *
                        (bounds[b].max[1] - bounds[b].min[1] + 1);
                    B = A + 1;
                    E = A + bounds[b].max[0] - bounds[b].min[0] + 1;
                    F = E + 1;

                    // tet EDHG
                    starth[m++] = startv + E;
                    starth[m++] = startv + D;
                    starth[m++] = startv + H;
                    starth[m++] = startv + G;

                    // tet ABCF
                    starth[m++] = startv + A;
                    starth[m++] = startv + B;
                    starth[m++] = startv + C;
                    starth[m++] = startv + F;

                    // tet ADEF
                    starth[m++] = startv + A;
                    starth[m++] = startv + D;
                    starth[m++] = startv + E;
                    starth[m++] = startv + F;

                    // tet CGDF
                    starth[m++] = startv + C;
                    starth[m++] = startv + G;
                    starth[m++] = startv + D;
                    starth[m++] = startv + F;

                    // tet ACDF
                    starth[m++] = startv + A;
                    starth[m++] = startv + C;
                    starth[m++] = startv + D;
                    starth[m++] = startv + F;

                    // tet DGEF
                    starth[m++] = startv + D;
                    starth[m++] = startv + G;
                    starth[m++] = startv + E;
                    starth[m++] = startv + F;

                }

                n++;

            }
        }
    }

    // add vertices and cells to the mesh set
    Range vRange(startv, startv + num_verts - 1); // vertex range
    Range cRange(startc, startc + num_tets - 1); // cell range
    rval = mbint->add_entities(*mesh_set, vRange); ERR;
    rval = mbint->add_entities(*mesh_set, cRange); ERR;

    // set global ids
    long gid;
    Tag global_id_tag;
    rval = mbint->tag_get_handle("HANDLEID", 1, MB_TYPE_HANDLE,
                                 global_id_tag, MB_TAG_CREAT|MB_TAG_DENSE); ERR;

    // gids for vertices, starting at 1 by moab convention
    handle = startv;
    for (int k = bounds[b].min[2]; k < bounds[b].max[2] + 1; k++)
    {
        for (int j = bounds[b].min[1]; j < bounds[b].max[1] + 1; j++)
        {
            for (int i = bounds[b].min[0]; i < bounds[b].max[0] + 1; i++)
            {
                gid = (long)1 + (long)i + (long)j * (mesh_size[0]) +
                    (long)k * (mesh_size[0]) * (mesh_size[1]);
                // debug
//                 fprintf(stderr, "i,j,k = [%d %d %d] gid = %ld\n", i, j, k, gid);
                rval = mbint->tag_set_data(global_id_tag, &handle, 1, &gid); ERR;
                handle++;
            }
        }
    }

    // gids for cells, starting at 1 by moab convention
    handle = startc;
    for (int k = bounds[b].min[2]; k < bounds[b].max[2]; k++)
    {
        for (int j = bounds[b].min[1]; j < bounds[b].max[1]; j++)
        {
            for (int i = bounds[b].min[0]; i < bounds[b].max[0]; i++)
            {
                for (int t = 0; t < 6; t++) // 6 tets per grid space
                {
                    gid = (long)1 + (long)t +  (long)i * 6 + (long)j * 6 * (mesh_size[0] - 1) +
                        (long)k * 6 * (mesh_size[0] - 1) * (mesh_size[1] - 1);
                    // 	 fprintf(stderr, "t,i,j,k = [%d %d %d %d] gid = %ld\n", t, i, j, k, gid);
                    rval = mbint->tag_set_data(global_id_tag, &handle, 1, &gid); ERR;
                    handle++;
                }
            }
        }
    }

    // update adjacencies (needed by moab)
    rval = iface->update_adjacencies(startc, num_tets, 4, starth); ERR;

    // cleanup
    rval = mbint->release_interface(iface); ERR;

}
//
// resolve shared entities
//
void resolve_and_exchange(Interface *mbint,       // mbint: moab interface instance
                          EntityHandle *mesh_set, // mesh_set: moab mesh set
			  ParallelComm *mbpc)     // mbpc: moab parallel communicator
{
    ErrorCode rval;

    mbpc->partition_sets().insert(*mesh_set);
    Tag global_id_tag;
    rval = mbint->tag_get_handle("HANDLEID", 1, MB_TYPE_HANDLE, global_id_tag, MB_TAG_DENSE); ERR;
    rval = mbpc->resolve_shared_ents(*mesh_set, -1, -1, &global_id_tag); ERR;
}
