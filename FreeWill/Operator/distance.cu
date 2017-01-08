/*
 * distance.cu
 * Copyright (C) 2016- CloudBrain <byzhang@>
 */

#include "distance.h"

__global__ void hamming_distance(KEY_T* keys, uint32_t *values, const uint32_t *query,
    cudaTextureObject_t tex, unsigned int tex_height, int num_dim, int num_data_per_block) {
  int tu = blockDim.x * blockIdx.x;
  int tv = threadIdx.x;

  if (tu < tex_height && tv < num_data_per_block) {
    extern __shared__ uint32_t query_local[];

    if (tv < num_dim) {
      query_local[tv] = query[tv];
    }

    __syncthreads();

    KEY_T count = 0;

    for (int i = 0; i<num_dim; ++i) {
      unsigned int m = tex2D<unsigned int>(tex, tv * num_dim + i, tu);
      count += __popc(m ^ query_local[i]);

    }

    unsigned int id = tu*num_data_per_block + tv;

    keys[id] = count;
    values[id] = id;
  }
}
