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
// Copyright Notice
// + 2012 University of Chicago
// See COPYRIGHT in top-level directory.
//
//--------------------------------------------------------------------------

#ifndef _MESH_GEN
#define _MESH_GEN

#ifdef MOAB

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
#include "cian.h"

using namespace moab;

void hex_mesh_gen(int *mesh_size, Interface *mbint, EntityHandle *mesh_set,
		  ParallelComm *mbpc, int did);
void tet_mesh_gen(int *mesh_size, Interface *mbint, EntityHandle *mesh_set,
		  ParallelComm *mbpc, int did);
void create_hexes_and_verts(int *mesh_size, Interface *mbint, 
			    EntityHandle *mesh_set, int did);
void create_tets_and_verts(int *mesh_size, Interface *mbint,
			   EntityHandle *mesh_set, int did);
void resolve_and_exchange(Interface *mbint, EntityHandle *mesh_set,
			  ParallelComm *mbpc);

#endif

#endif
