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

#ifndef _MESH_GEN
#define _MESH_GEN

#include "mpi.h"
#include <stddef.h>

#include "iMesh.h"
#include "MBiMesh.hpp"
#include "MBCore.hpp"
#include "MBRange.hpp"
#include "MBTagConventions.hpp"
#include "moab/ParallelComm.hpp"
#include "moab/HomXform.hpp"
#include "moab/ReadUtilIface.hpp"
#include "Coupler.hpp"

#include <diy/decomposition.hpp>

typedef     diy::ContinuousBounds       Bounds;

#define ERR {if(rval!=MB_SUCCESS)printf("MOAB error at line %d in %s\n", __LINE__, __FILE__);}

using namespace moab;

void hex_mesh_gen(int *mesh_size, Interface *mbint, EntityHandle *mesh_set,
		  ParallelComm *mbpc, diy::RegularDecomposer<Bounds>* decomp,
                  diy::Assigner* assign);
void tet_mesh_gen(int *mesh_size, Interface *mbint, EntityHandle *mesh_set,
		  ParallelComm *mbpc, diy::RegularDecomposer<Bounds>* decomp,
                  diy::Assigner* assign);
void create_hexes_and_verts(int *mesh_size, Interface *mbint, EntityHandle *mesh_set,
                            diy::RegularDecomposer<Bounds>* decomp,
                            diy::Assigner* assign, ParallelComm* mbpc);
void create_tets_and_verts(int *mesh_size, Interface *mbint, EntityHandle *mesh_set,
                           diy::RegularDecomposer<Bounds>* decomp,
                           diy::Assigner* assign, ParallelComm* mbpc);
void resolve_and_exchange(Interface *mbint, EntityHandle *mesh_set, ParallelComm *mbpc);

#endif
