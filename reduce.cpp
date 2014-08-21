//
// todo
//
// single threaded and multithreded versions of merge do not produce the 
//   correct result when k < 2. Need to pass ids of received items to
//   comppute function (tags are available, just not passed)
//
// multithreaded version of swap does not produce correct result either, 
//   presumably for same reason
//
//---------------------------------------------------------------------------
//
// testing DIY's reduction performance and comparing to MPI
//
// Tom Peterka
// Argonne National Laboratory
// 9700 S. Cass Ave.
// Argonne, IL 60439
// tpeterka@mcs.anl.gov
//
// (C) 2012 by Argonne National Laboratory.
// See COPYRIGHT in top-level directory.
//
//--------------------------------------------------------------------------
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
#include "diy.h"
#include <vector>
#include <algorithm>

using namespace std;

#define MAX_ROUNDS 20
#define MAX_ITEMS 8

struct gid_map_t {
  int gid; // gid
  int pos; // position in original list of gids
};

// globals
int nblocks; // my local number of blocks
int num_elems; // number of data elements per block
int op; // 1=normal, 0 = no op

// function prototypes
void GetArgs(int argc, char **argv, int &min_procs, int &min_elems, 
	     int &max_elems, int &nb, int &target_k, int &op);
void KValues(int tb, int k, int &nr, int *kv);
void Reduce(double *reduce_time, int run, float **in_data, MPI_Comm comm);
void ReduceScatter(double *reduce_scatter_time, int run, float **in_data, 
		   MPI_Comm comm);
void Merge(double *merge_time, int run, float **in_data, int rounds, 
	   int *k_values, MPI_Comm comm);
void Swap(double *swap_time, int run, float **in_data, int rounds, 
	  int *k_values, MPI_Comm comm);
void PrintResults(double *reduce_time, double *reduce_scatter_time, 
		  double *merge_time, double *swap_time, int min_procs,
		  int max_procs, int min_elems, int max_elems);
void ComputeMerge(char **items, int *gids, int nitems, int *hdr);
char *CreateItem(int *hdr);
void DestroyItem(void *);
char *CreateItemSize(int *hdr, int ne);
void CreateType(void *item, DIY_Datatype *dtype, int *hdr);
void ComputeSwap(char **items, int *gids, int nitems, int num_elems);
void *SendType(void *item, DIY_Datatype *dtype, int start_elem,
	       int num_elems);
void RecvType(void *item, DIY_Datatype *dtype, int num_elems);
void Sum(void *in, void *inout, int *len, MPI_Datatype *type);
void Over(void *in, void *inout, int *len, MPI_Datatype *type);
bool Compare(gid_map_t map1, gid_map_t map2);

//----------------------------------------------------------------------------
//
// main
//
int main(int argc, char **argv) {

  int dim = 3; // number of dimensions in the problem
  int tot_blocks; // total number of blocks
  int data_size[] = {10, 10, 10}; // data size (unused)
  int given[] = {0, 0, 0}; // constraints on blocking (none)
  int ghost[] = {0, 0, 0, 0, 0, 0}; // ghost in each -/+ direction
  int target_k; // target k-value
  int rounds; // number of rounds
  int k_values[MAX_ROUNDS]; // k-values

  int rank, groupsize; // MPI usual
  int min_procs; // minimum number of processes
  int max_procs; // maximum number of processes (groupsize of MPI_COMM_WORLD)
  int min_elems, max_elems; // min, max numbe of elements

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &max_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  GetArgs(argc, argv, min_procs, min_elems, max_elems, nblocks, target_k, op);

  int num_runs = (int)((log2(max_procs / min_procs) + 1) *
    (log2(max_elems / min_elems) + 1));

  // timing
  double reduce_time[num_runs];
  double reduce_scatter_time[num_runs];
  double merge_time[num_runs];
  double swap_time[num_runs];

  float **in_data; // data for each local block
  in_data = (float**)malloc(nblocks * sizeof(float *));
  float **bkp_data; // data for each local block
  bkp_data = (float**)malloc(nblocks * sizeof(float *));

  // iterate over processes
  int run = 0; // run number
  groupsize = min_procs;
  while (groupsize <= max_procs) {

    // form a new communicator
    MPI_Comm comm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_split(MPI_COMM_WORLD, (rank < groupsize), rank, &comm);
    if (rank >= groupsize) {
      MPI_Comm_free(&comm);
      groupsize *= 2;
      continue;
    }
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &groupsize);

    // initialize DIY
    tot_blocks = nblocks * groupsize;
    DIY_Init(dim, 1, comm);
    DIY_Decompose(ROUND_ROBIN_ORDER, data_size, tot_blocks, &nblocks, 
		  0, ghost, given, 0);

    // compute rounds and k-values
    KValues(tot_blocks, target_k, rounds, k_values);

    // iterate over number of elements
    num_elems = min_elems;
    while (num_elems <= max_elems) {

      // allocate input data
      for (int b = 0; b < nblocks; b++) {
	in_data[b] = (float *)malloc(num_elems * sizeof(float));
	bkp_data[b] = in_data[b];
      }

      // initialize input data
      for (int b = 0; b < nblocks; b++) { // all my blocks
	for (int i = 0; i < num_elems; i++)
	  in_data[b][i] = (float)((rank * nblocks + b) * num_elems + i) / 
	    (tot_blocks * num_elems);
      }

      // MPI reduce
      // only for one block per process
      if (tot_blocks == groupsize)
	Reduce(reduce_time, run, in_data, comm);

      // MPI reduce_scatter
      // only for one block per process
      if (tot_blocks == groupsize)
	ReduceScatter(reduce_scatter_time, run, in_data, comm);

      // DIY merge
      Merge(merge_time, run, in_data, rounds, k_values, comm);

      // re-initialize input data (in-place DIY merge disturbed it)
      for (int b = 0; b < nblocks; b++) { // all my blocks
	for (int i = 0; i < num_elems; i++)
	  in_data[b][i] = (float)((rank * nblocks + b) * num_elems + i) / 
	    (tot_blocks * num_elems);
      }

      // DIY swap
      Swap(swap_time, run, in_data, rounds, k_values, comm);

      num_elems *= 2; // double the number of elements every time
      run++;

      // cleanup
      for (int b = 0; b < nblocks; b++)
	free(bkp_data[b]); 

    } // elem iteration

    DIY_Finalize();
    groupsize *= 2; // double the number of processes every time
    MPI_Comm_free(&comm);

  } // proc iteration

  // cleanup
  free(in_data);
  free(bkp_data);

  // print results
  fflush(stderr);
  if (rank == 0)
    PrintResults(reduce_time, reduce_scatter_time, merge_time, swap_time, 
		 min_procs, max_procs, min_elems, max_elems);

  MPI_Finalize();

  return 0;

}
//----------------------------------------------------------------------------
//
// MPI reduce
//
// reduce_time: time (output)
// run: run number
// in_data: input data
// comm: current communicator
//
void Reduce(double *reduce_time, int run, float **in_data, MPI_Comm comm) {

  MPI_Op op; // custom operator
  MPI_Op_create(&Over, 0, &op); // nonommutative
  float *reduce_data = new float[num_elems];
  MPI_Barrier(comm);
  double t0 = MPI_Wtime();
  MPI_Reduce((void *)in_data[0], (void *)reduce_data, num_elems, 
	     MPI_FLOAT, op, 0, comm);
  MPI_Barrier(comm);
  reduce_time[run] = MPI_Wtime() - t0;

  // debug: print the reduced data
//   int rank;
//   MPI_Comm_rank(comm, &rank);
//   if (rank == 0) {
//     for (int i = 0; i < num_elems; i++)
//       fprintf(stderr, "reduce_data[%d] = %.3f\n", i, reduce_data[i]);
//   }

  delete[] reduce_data;
  MPI_Op_free(&op);

}
//----------------------------------------------------------------------------
//
// MPI reduce_scatter
//
// reduce_scatter_time: time (output)
// run: run number
// in_data: input data
// comm: current communicator
//
void ReduceScatter(double *reduce_scatter_time, int run, float **in_data, 
		   MPI_Comm comm) {

  int groupsize;
  MPI_Comm_size(comm, &groupsize);

  MPI_Op op; // custom operator
  MPI_Op_create(&Over, 0, &op); // noncommutative
  float *reduce_scatter_data = new float[num_elems];
  int counts[groupsize];
  int size = num_elems  / groupsize;
  for (int i = 0; i < groupsize; i++)
    counts[i] = size;
  counts[groupsize - 1] = num_elems - (groupsize - 1) * size; // remainder
  MPI_Barrier(comm);
  double t0 = MPI_Wtime();
  MPI_Reduce_scatter((void *)in_data[0], (void *)reduce_scatter_data, 
		     counts, MPI_FLOAT, op, comm);
  MPI_Barrier(comm);
  reduce_scatter_time[run] = MPI_Wtime() - t0;

  // debug: print the reduce-scattered data
//   for (int i = 0; i < counts[rank]; i++)
//     fprintf(stderr, "reduce_scatter_data[%d] = %.3f\n", 
// 	    i, reduce_scatter_data[i]);

  delete[] reduce_scatter_data;
  MPI_Op_free(&op);

}
//----------------------------------------------------------------------------
//
// DIY merge
//
// merge_time: time (output)
// run: run number
// in_data: input data (output in in_data[0], merged in-place)
// rounds, k_values: number of rounds and k-values per round
//
void Merge(double *merge_time, int run, float **in_data, int rounds,
	   int *k_values, MPI_Comm comm) {

  int nb_merged; // number of output merged blocks
  MPI_Barrier(comm);
  double t0 = MPI_Wtime();
  DIY_Merge_blocks(0, (char**)in_data, (int **)NULL, rounds, k_values, 
		   &ComputeMerge, &CreateItem, &DestroyItem, &CreateType, 
		   &nb_merged);
  MPI_Barrier(comm);
  merge_time[run] = MPI_Wtime() - t0;

  // debug: print the merged data
//   for (int b = 0; b < nb_merged; b++) {
//     for (int i = 0; i < num_elems; i++)
//       fprintf(stderr, "merge_data[%d][%d] = %.3f\n", b, i, in_data[0][i]);
//   }

}
//----------------------------------------------------------------------------
//
// DIY swap
//
// swap_time: time (output)
// run: run number
// in_data: input data
// rounds, k_values: number of rounds and k-values per round
//
void Swap(double *swap_time, int run, float **in_data, int rounds,
	  int *k_values, MPI_Comm comm) {

  int starts[nblocks]; // starts and lengths of results in each local block
  int sizes[nblocks];
  MPI_Barrier(comm);
  double t0 = MPI_Wtime();
  DIY_Swap_blocks(0, (char**)in_data, (int **)NULL, num_elems,
		  rounds, k_values, starts, sizes, &ComputeSwap, 
		  &CreateItemSize, &DestroyItem, &SendType, &RecvType);
  MPI_Barrier(comm);
  swap_time[run] = MPI_Wtime() - t0;

  // debug: print the swapped data
//   for (int b = 0; b < nblocks; b++) {
//     for (int i = 0; i < sizes[b]; i++)
//       fprintf(stderr, "swap_data[%d][%d] = %.3f\n", b, starts[b] + i, 
// 	      in_data[b][i]);
//   }

}
//----------------------------------------------------------------------------
//
// compute number of rounds and k-values
// fairly naive: factors into only target k-values and one more factor of the
// rest, whatever that is
//
// tb: total blocks
// k: target k-value
// nr: number of rounds (output)
// kv: k-values (output)
//
void KValues(int tb, int k, int &nr, int *kv) {

  nr = 0;
  int rem = tb;
  while(rem / k > 0 && rem % k == 0) {
    kv[nr] = k;
    rem /= k;
    nr++;
  }
  if (rem > 1) {
    kv[nr] = rem;
    nr++;
  }

  // sanity check
  assert(nr <= MAX_ROUNDS);
  int prod_k = 1;
  for (int i = 0; i < nr; i++)
    prod_k *= kv[i];
  assert(prod_k == tb);

}
//----------------------------------------------------------------------------
//
// print results
//
// reduce_time, reduce_scatter_time, merge_time, swap_time: times
// min_procs, max_procs: process range
// min_elems, max_elems: data range
//
void PrintResults(double *reduce_time, double *reduce_scatter_time, 
		  double *merge_time, double *swap_time, int min_procs,
		  int max_procs, int min_elems, int max_elems) {

  int elem_iter = 0; // element iteration number
  // number of element iterations
  int num_elem_iters = (int)(log2(max_elems / min_elems) + 1);
  int proc_iter = 0; // process iteration number

  fprintf(stderr, "----- Timing Results -----\n");

  // iterate over number of elements
  num_elems = min_elems;
  while (num_elems <= max_elems) {

    fprintf(stderr, "\n# num_elemnts = %d   size @ 4 bytes / element = %d KB\n",
	    num_elems, num_elems * 4 / 1024);
    fprintf(stderr, "# procs \t red_time \t red_scat_time \t merge_time "
	    "\t swap_time\n");

    // iterate over processes
    int groupsize = min_procs;
    proc_iter = 0;
    while (groupsize <= max_procs) {

      int i = proc_iter * num_elem_iters + elem_iter; // index into times
      fprintf(stderr, "%d \t\t %.3lf \t\t %.3lf \t\t %.3lf \t\t %.3lf\n", 
	      groupsize, reduce_time[i], reduce_scatter_time[i],  
	      merge_time[i], swap_time[i]);

      groupsize *= 2; // double the number of processes every time
      proc_iter++;

    } // proc iteration

    num_elems *= 2; // double the number of elements every time
    elem_iter++;

  } // elem iteration

  fprintf(stderr, "\n--------------------------\n\n");

}
//----------------------------------------------------------------------------
#ifndef OMP
//
// nonthreaded, original version
//
// user-defined callback function for merging an array of items
// in this example we comute the sum of individual blocks
//
// items: pointers to input items
// gids: global ids of items to be reduced
// nitems: total number of input items
// char * is used as a generic pointers to bytes, not necessarily to strings
// hdr: quantity information for items[0], unused
//
void ComputeMerge(char **items, int *gids, int nitems, int *hdr) {

  if (!op)
    return;

  // sort gids
  vector<gid_map_t> map;
  map.reserve(MAX_ITEMS);
  for (int i = 0; i < nitems; i++) {
    map[i].gid = gids[i];
    map[i].pos = i;
  }
  sort(map.begin(), map.begin() + nitems, Compare);

  int inout = map[nitems - 1].pos; // resulting item

  // over operator
  for (int k = nitems - 2; k >= 0; k--) {

    int in = map[k].pos; // incoming item

    for (int i = 0; i < num_elems / 4; i++) {

      ((float **)items)[inout][i * 4] = 
	(1.0f - ((float **)items)[in][i * 4 + 3]) * 
	((float **)items)[inout][i * 4] + 
	((float **)items)[in][i * 4];

      ((float **)items)[inout][i * 4 + 1] = 
	(1.0f - ((float **)items)[in][i * 4 + 3]) * 
	((float **)items)[inout][i * 4 + 1] + 
	((float **)items)[in][i * 4 + 1];

      ((float **)items)[inout][i * 4 + 2] = 
	(1.0f - ((float **)items)[in][i * 4 + 3]) * 
	((float **)items)[inout][i * 4 + 2] + 
	((float **)items)[in][i * 4 + 2];

      ((float **)items)[inout][i * 4 + 3] = 
	(1.0f - ((float **)items)[in][i * 4 + 3]) * 
	((float **)items)[inout][i * 4 + 3] + 
	((float **)items)[in][i * 4 + 3];

    }

  }

  // put result in items[0] (swap instead of copy so that pointers are not
  // duplicated and can be freed
  char *temp;
  temp = items[0];
  items[0] = items[inout];
  items[inout] = temp;

}
#endif
//----------------------------------------------------------------------------
#ifdef OMP
//
// threaded version
//
// user-defined callback function for merging an array of items
// in this example we comute the sum of individual blocks
//
// items: pointers to input items
// gids: global ids of items to be reduced
// nitems: total number of input items
// char * is used as a generic pointers to bytes, not necessarily to strings
// hdr: quantity information for items[0], unused
//
void ComputeMerge(char **items, int *gids, int nitems, int *hdr) {

  if (!op)
    return;

  // sort gids
  vector<gid_map_t> map;
  map.reserve(MAX_ITEMS);
  for (int i = 0; i < nitems; i++) {
    map[i].gid = gids[i];
    map[i].pos = i;
  }
  sort(map.begin(), map.begin() + nitems, Compare);

  int ngroups = nitems / 2; // number of groups in a tree level
  int nlevels = log2(nitems); // number of tree levels
  int d = 1; // distance between members in a group in a tree level
  int in, inout; // input and input/output items to actual computation

  for (int k = 0; k < nlevels; k++) { // tree levels

#pragma omp parallel for

    for (int j = 0; j < ngroups; j++) { // groups in a tree level

      int r = nitems - 1 - j * 2 * d; // resulting item
      int s = r - d; // source item
      inout = map[r].pos; // resulting item mapped to sorted list of gids
      in = map[s].pos; // source item mapped to sorted list of gids

      // over operator
      for (int i = 0; i < num_elems / 4; i++) {

	((float **)items)[inout][i * 4] = 
	  (1.0f - ((float **)items)[in][i * 4 + 3]) * 
	  ((float **)items)[inout][i * 4] + 
	  ((float **)items)[in][i * 4];

	((float **)items)[inout][i * 4 + 1] = 
	  (1.0f - ((float **)items)[in][i * 4 + 3]) * 
	  ((float **)items)[inout][i * 4 + 1] + 
	  ((float **)items)[in][i * 4 + 1];

	((float **)items)[inout][i * 4 + 2] = 
	  (1.0f - ((float **)items)[in][i * 4 + 3]) * 
	  ((float **)items)[inout][i * 4 + 2] + 
	  ((float **)items)[in][i * 4 + 2];

	((float **)items)[inout][i * 4 + 3] = 
	  (1.0f - ((float **)items)[in][i * 4 + 3]) * 
	  ((float **)items)[inout][i * 4 + 3] + 
	  ((float **)items)[in][i * 4 + 3];

      }

    } // groups in a tree level

    d *= 2;
    ngroups /= 2;

  } // tree levels

  // put result in items[0] (swap instead of copy so that pointers are not
  // duplicated and can be freed
  char *temp;
  temp = items[0];
  items[0] = items[inout];
  items[inout] = temp;

}
#endif
//----------------------------------------------------------------------------
//
// user-defined callback function for creating a received item
//
// hdr: quantity information for allocating custom parts of the item
//  (not used in this example)
// char * is used as a generic pointers to bytes, not necessarily to strings
//
// side effects: allocates the item
//
// returns: pointer to the item
//
char *CreateItem(int *hdr) {

  float *b = new float[num_elems]; 
  return (char *)b;

}
//----------------------------------------------------------------------------
//
// user-defined callback function for destroying a received item
//
// item: item to be destroyed
//
void DestroyItem(void *item) {

  delete[] (float *)item;

}
//----------------------------------------------------------------------------
//
// user-defined callback function for creating a received item
//  given a specified number of elements
//
// hdr: quantity information for allocating custom parts of the item
//  (not used in this example)
// char * is used as a generic pointers to bytes, not necessarily to strings
// ne: number of elements
//
// side effects: allocates the item
//
// returns: pointer to the item
//
char *CreateItemSize(int *hdr, int ne) {

  float *b = new float[ne];
  return (char *)b;

}
//----------------------------------------------------------------------------
//
// user-defined callback function for creating an MPI datatype for the
//   received item
//
// item: pointer to the item
// dtype: pointer to the datatype
// hdr: quantity information, unused
//
// side effects: commits the MPI datatype but DIY will cleanup datatype for you
//
void CreateType(void *item, DIY_Datatype *dtype, int *hdr) {

  DIY_Create_vector_datatype(num_elems, 1, DIY_FLOAT, dtype);

}
//----------------------------------------------------------------------------
#ifndef OMP
//
// unthreaded, original version
//
// user-defined callback function for reducing an array of items
// in this example we "compose" an image by adding element values
// user should write this function so the result of the reduction is in items[0]
//
// items: pointers to input and output items, reduced in place
//   char * is used as a generic pointers to bytes, not necessarily to strings
//   items are partial size (size of current active part)
// gids: global ids of items to be reduced
// nitems: number of items to reduce
// num_elems: current number of elements in item to reduce (gets smaller with
//  every round)
//
void ComputeSwap(char **items, int *gids, int nitems, int num_elems) {

  if (!op)
    return;

  // sort gids
  vector<gid_map_t> map;
  map.reserve(MAX_ITEMS);
  for (int i = 0; i < nitems; i++) {
    map[i].gid = gids[i];
    map[i].pos = i;
  }
  sort(map.begin(), map.begin() + nitems, Compare);

  int inout = map[nitems - 1].pos; // resulting item

  for (int k = nitems - 2; k >= 0; k--) {

    int in = map[k].pos; // incoming item

    for (int i = 0; i < num_elems / 4; i++) { // index in received items

      ((float **)items)[inout][i * 4] = 
	(1.0f - ((float **)items)[in][i * 4 + 3]) * 
	((float **)items)[inout][i * 4] + 
	((float **)items)[in][i * 4];

      ((float **)items)[inout][i * 4 + 1] = 
	(1.0f - ((float **)items)[in][i * 4 + 3]) * 
	((float **)items)[inout][i * 4 + 1] + 
	((float **)items)[in][i * 4 + 1];

      ((float **)items)[inout][i * 4 + 2] = 
	(1.0f - ((float **)items)[in][i * 4 + 3]) * 
	((float **)items)[inout][i * 4 + 2] + 
	((float **)items)[in][i * 4 + 2];

      ((float **)items)[inout][i * 4 + 3] = 
	(1.0f - ((float **)items)[in][i * 4 + 3]) * 
	((float **)items)[inout][i * 4 + 3] + 
	((float **)items)[in][i * 4 + 3];

    }

  }

  // put result in items[0] (swap instead of copy so that pointers are not
  // duplicated and can be freed
  char *temp;
  temp = items[0];
  items[0] = items[inout];
  items[inout] = temp;

}
#endif
//----------------------------------------------------------------------------
#ifdef OMP
//
// threaded version
//
// user-defined callback function for reducing an array of items
// in this example we "compose" an image by adding element values
// user should write this function so the result of the reduction is in items[0]
//
// items: pointers to input and output items, reduced in place
//   char * is used as a generic pointers to bytes, not necessarily to strings
//   items are partial size (size of current active part)
// gids: global ids of items to be reduced
// nitems: number of items to reduce
// num_elems: current number of elements in item to reduce (gets smaller with
//  every round)
//
void ComputeSwap(char **items, int *gids, int nitems, int num_elems) {

  if (!op)
    return;

  // sort gids
  vector<gid_map_t> map;
  map.reserve(MAX_ITEMS);
  for (int i = 0; i < nitems; i++) {
    map[i].gid = gids[i];
    map[i].pos = i;
  }
  sort(map.begin(), map.begin() + nitems, Compare);

  int ngroups = nitems / 2; // number of groups in a tree level
  int nlevels = log2(nitems); // number of tree levels
  int d = 1; // distance between members in a group in a tree level
  int in, inout; // input and input/output items to actual computation

  for (int k = 0; k < nlevels; k++) { // tree levels

#pragma omp parallel for

    for (int j = 0; j < ngroups; j++) { // groups in a tree level

      int r = nitems - 1 - j * 2 * d; // resulting item
      int s = r - d; // source item
      inout = map[r].pos; // resulting item mapped to sorted list of gids
      in = map[s].pos; // source item mapped to sorted list of gids

      // over operator
      for (int i = 0; i < num_elems / 4; i++) {

	((float **)items)[inout][i * 4] = 
	  (1.0f - ((float **)items)[in][i * 4 + 3]) * 
	  ((float **)items)[inout][i * 4] + 
	  ((float **)items)[in][i * 4];

	((float **)items)[inout][i * 4 + 1] = 
	  (1.0f - ((float **)items)[in][i * 4 + 3]) * 
	  ((float **)items)[inout][i * 4 + 1] + 
	  ((float **)items)[in][i * 4 + 1];

	((float **)items)[inout][i * 4 + 2] = 
	  (1.0f - ((float **)items)[in][i * 4 + 3]) * 
	  ((float **)items)[inout][i * 4 + 2] + 
	  ((float **)items)[in][i * 4 + 2];

	((float **)items)[inout][i * 4 + 3] = 
	  (1.0f - ((float **)items)[in][i * 4 + 3]) * 
	  ((float **)items)[inout][i * 4 + 3] + 
	  ((float **)items)[in][i * 4 + 3];

      }

    } // groups in a tree level

    d *= 2;
    ngroups /= 2;

  } // tree levels

  // put result in items[0] (swap instead of copy so that pointers are not
  // duplicated and can be freed
  char *temp;
  temp = items[0];
  items[0] = items[inout];
  items[inout] = temp;

}
#endif
//----------------------------------------------------------------------------
//
// user-defined callback function for creating an MPI datatype for sending
//   swapped item
//
// item: pointer to the item
// dtype: pointer to the datatype
// start_elem: starting element position to be sent
// num_elems: number of elements to be sent (less than number of elements
//   in the complete item
//
// side effects: commits the MPI datatype
//
// returns: base address associated with the datatype
//
void *SendType(void *item, DIY_Datatype *dtype, int start_elem,
	       int num_elems) {

  DIY_Create_vector_datatype(num_elems, 1, DIY_FLOAT, dtype);

  // user's job to compute the return address correctly by scaling the
  // pointer to item by the size of an element, in this case float
  return ((float *)item + start_elem);

}
//----------------------------------------------------------------------------
//
// user-defined callback function for creating an MPI datatype for receiving
//   swapped item
//
// item: pointer to the item
// dtype: pointer to the datatype
// num_elems: number of elements in the received datatyep (less than number
//   of elements in the complete item)
//
// side effects: commits the MPI datatype
//
void RecvType(void *item, DIY_Datatype *dtype, int num_elems) {

  DIY_Create_vector_datatype(num_elems, 1, DIY_FLOAT, dtype);

}
//----------------------------------------------------------------------------
//
// performs in plus inout
// inout is the result
// both in and inout have length len
//
void Sum(void *in, void *inout, int *len, MPI_Datatype *type) {

  // quiet the warnings
  type = type;

  for (int i = 0; i < *len; i++)
    ((float *)inout)[i] = ((float *)inout)[i] + ((float *)in)[i];

}
//----------------------------------------------------------------------------
//
// Over()
//
// performs in over inout
// inout is the result
// both in and inout have same size in pixels
//
void Over(void *in, void *inout, int *len, MPI_Datatype *type) {

  // quiet the warnings
  type = type;

  if (!op)
    return;

  for (int i = 0; i < *len / 4; i++) {

    ((float *)inout)[i * 4] = 
      (1.0f - ((float *)in)[i * 4 + 3]) * ((float *)inout)[i * 4] + 
      ((float *)in)[i * 4];

    ((float *)inout)[i * 4 + 1] = 
      (1.0f - ((float *)in)[i * 4 + 3]) * ((float *)inout)[i * 4 + 1] + 
      ((float *)in)[i * 4 + 1];

    ((float *)inout)[i * 4 + 2] = 
      (1.0f - ((float *)in)[i * 4 + 3]) * ((float *)inout)[i * 4 + 2] + 
      ((float *)in)[i * 4 + 2];

    ((float *)inout)[i * 4 + 3] = 
      (1.0f - ((float *)in)[i * 4 + 3]) * ((float *)inout)[i * 4 + 3] + 
      ((float *)in)[i * 4 + 3];

  }

}
//----------------------------------------------------------------------------
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
// side effects: allocates kv
//
void GetArgs(int argc, char **argv, int &min_procs, 
	     int &min_elems, int &max_elems, int &nb, int &target_k, int &op) {

  assert(argc >= 7);

  min_procs = atoi(argv[1]);
  min_elems = atoi(argv[2]);
  max_elems = atoi(argv[3]);
  nb = atoi(argv[4]);
  target_k = atoi(argv[5]);
  op = atoi(argv[6]);

  // cheack there is at least four elements (eg., one pixel) per block
  int max_procs;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &max_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(min_elems >= 4 *nb * max_procs); // at least one element per block

  if (rank == 0)
    fprintf(stderr, "min_procs = %d min_elems = %d max_elems = %d nb = %d "
	    "target_k = %d\n", min_procs, min_elems, max_elems, nb, target_k);


}
//----------------------------------------------------------------------------
//
// comparison function for sorting gid map
//
// returns true if map1.gid < map2.gid, else false
//
bool Compare(gid_map_t map1, gid_map_t map2) {

  return (map1.gid < map2.gid);

}
//----------------------------------------------------------------------------
