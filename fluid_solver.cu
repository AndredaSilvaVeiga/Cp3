#include "fluid_solver.h"
#include <cmath>
#include <omp.h>
#include <cuda.h>
#include <stdio.h>
#include <float.h>


#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20


__global__ void add_source_kernel(int size, float *d_x, float *d_s, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

  if (i < size) {
    d_x[i] += dt * d_s[i];
  }
}

void add_source_call_kernel(int M, int N, int O, float *d_x, float *d_s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  int threads_per_block = 256;
  int blocks_grid = (size + threads_per_block - 1) / threads_per_block;

  add_source_kernel<<<blocks_grid, threads_per_block>>>(size, d_x, d_s, dt);
  cudaDeviceSynchronize();
}


__global__ void set_bnd_kernel(int M, int N, int O, int b, float *d_x) {
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;

    if (i >= 1 && i <= M && j >= 1 && j <= N) {
        d_x[IX(i, j, 0)] = b == 3 ? -d_x[IX(i, j, 1)] : d_x[IX(i, j, 1)];
        d_x[IX(i, j, O + 1)] = b == 3 ? -d_x[IX(i, j, O)] : d_x[IX(i, j, O)];
    }

    if (i >= 1 && i <= N && j >= 1 && j <= O) {
        d_x[IX(0, i, j)] = b == 1 ? -d_x[IX(1, i, j)] : d_x[IX(1, i, j)];
        d_x[IX(M + 1, i, j)] = b == 1 ? -d_x[IX(M, i, j)] : d_x[IX(M, i, j)];
    }

    if (i >= 1 && i <= M && j >= 1 && j <= O) {
        d_x[IX(i, 0, j)] = b == 2 ? -d_x[IX(i, 1, j)] : d_x[IX(i, 1, j)];
        d_x[IX(i, N + 1, j)] = b == 2 ? -d_x[IX(i, N, j)] : d_x[IX(i, N, j)];
    }
}

__global__ void set_corners_kernel(int M, int N, int O, float *d_x) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_x[IX(0, 0, 0)] =         0.33f * (d_x[IX(1, 0, 0)] + d_x[IX(0, 1, 0)] + d_x[IX(0, 0, 1)]);
        d_x[IX(M + 1, 0, 0)] =     0.33f * (d_x[IX(M, 0, 0)] + d_x[IX(M + 1, 1, 0)] + d_x[IX(M + 1, 0, 1)]);
        d_x[IX(0, N + 1, 0)] =     0.33f * (d_x[IX(1, N + 1, 0)] + d_x[IX(0, N, 0)] + d_x[IX(0, N + 1, 1)]);
        d_x[IX(M + 1, N + 1, 0)] = 0.33f * (d_x[IX(M, N + 1, 0)] + d_x[IX(M + 1, N, 0)] + d_x[IX(M + 1, N + 1, 1)]);
    }
}

void set_bnd_call_kernels(int M, int N, int O, int b, float *d_x) {
    dim3 threads_per_block(16, 16); 
    dim3 blocks_grid((M + threads_per_block.x - 1) / threads_per_block.x,
                     (N + threads_per_block.y - 1) / threads_per_block.y); 

    set_bnd_kernel<<<blocks_grid, threads_per_block>>>(M, N, O, b, d_x);
    cudaDeviceSynchronize();

    set_corners_kernel<<<1, 1>>>(M, N, O, d_x);
    cudaDeviceSynchronize();
}


// Sequential Addressing
__global__ void reduce_max(float *input, float *output) {
  extern __shared__ float shared_data[];

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  shared_data[tid] = input[idx];
  __syncthreads();

  for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) {
        shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
    }
     __syncthreads();
  }

  if(tid == 0) {
	  output[blockIdx.x] = shared_data[0];
  }	
}


__global__ void lin_solver_kernel(float *d_x, float *d_x0, int N, int M, int O, float a, float c, float *d_change_array, int sign) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if((i + j + k) % 2 != sign) return;

  if (i <= M && j <= N && k <= O) {
    int idx = IX(i, j, k);
    float  old_x = d_x[idx];
    d_x[idx] = (d_x0[idx] + a *(d_x[IX(i - 1, j, k)] + d_x[IX(i + 1, j, k)] +
                            d_x[IX(i, j - 1, k)] + d_x[IX(i, j + 1, k)] +
                            d_x[IX(i, j, k - 1)] + d_x[IX(i, j, k + 1)])) / c;
	  float change = fabsf(d_x[idx] - old_x);
	  d_change_array[idx] = change;
  }
}


void lin_solve(int M, int N, int O, int b, float *d_x, float *d_x0, float a, float c) {
  float *d_change_array, *d_change_reduction, *h_change_reduction;
  int size = (M + 2) * (N + 2) * (O + 2);

  dim3 threads_per_block(8, 8, 8);
  dim3 num_blocks((M + threads_per_block.x - 1) / threads_per_block.x,
                   (N + threads_per_block.y - 1) / threads_per_block.y,
                   (O + threads_per_block.z - 1) / threads_per_block.z);

  int threads_per_block_reduction = 1024;
  int num_blocks_reduction  = (M * N * O + threads_per_block_reduction - 1) / threads_per_block_reduction; 

  cudaMalloc((void **)&d_change_array, size * sizeof(float)); 
  cudaMalloc((void **)&d_change_reduction, num_blocks_reduction * sizeof(float)); 
  h_change_reduction = (float *)malloc(num_blocks_reduction * sizeof(float));

  int l = 0;
  float max_change;
  float tol = 1e-7;
  do {      
    max_change = 0.0f;

    //Phase 1
    lin_solver_kernel<<<num_blocks, threads_per_block>>>(d_x, d_x0, N, M, O, a, c, d_change_array, 0);
    //Phase 2
    lin_solver_kernel<<<num_blocks, threads_per_block>>>(d_x, d_x0, N, M, O, a, c, d_change_array, 1);

    // Reduction GPU
    reduce_max<<<num_blocks_reduction, threads_per_block_reduction, threads_per_block_reduction * sizeof(float)>>>(d_change_array, d_change_reduction);  
 
    // Reduction CPU
    cudaMemcpy(h_change_reduction, d_change_reduction, num_blocks_reduction * sizeof(float), cudaMemcpyDeviceToHost);
    
    #pragma omp parallel for reduction(max:max_change)
    for(int i = 0; i <= num_blocks_reduction; i++) {
    	    max_change = fmaxf(max_change, h_change_reduction[i]);
    }

    set_bnd_call_kernels(M, N, O, b, d_x);
   
    } while (++l < LINEARSOLVERTIMES && max_change > tol);


  // Free memory
  cudaFree(d_change_array);
  cudaFree(d_change_reduction);
  free(h_change_reduction);
}


void diffuse(int M, int N, int O, int b, float *d_x, float *d_x0, float diff, float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, d_x, d_x0, a, 1 + 6 * a);
}


__global__ void advect_kernel(int M, int N, int O, float dt, float *d_d, float *d_d0, float *d_u, float *d_v, float *d_w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

        float x = i - dtX * d_u[IX(i, j, k)];
        float y = j - dtY * d_v[IX(i, j, k)];
        float z = k - dtZ * d_w[IX(i, j, k)];

        // Clamp to grid boundaries
        if (x < 0.5f)
            x = 0.5f;
        if (x > M + 0.5f)
            x = M + 0.5f;
        if (y < 0.5f)
            y = 0.5f;
        if (y > N + 0.5f)
            y = N + 0.5f;
        if (z < 0.5f)
            z = 0.5f;
        if (z > O + 0.5f)
            z = O + 0.5f;

        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1;
        int k0 = (int)z, k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1;
        float u1 = z - k0, u0 = 1 - u1;

        d_d[IX(i, j, k)] =
            s0 * (t0 * (u0 * d_d0[IX(i0, j0, k0)] + u1 * d_d0[IX(i0, j0, k1)]) +
                  t1 * (u0 * d_d0[IX(i0, j1, k0)] + u1 * d_d0[IX(i0, j1, k1)])) +
            s1 * (t0 * (u0 * d_d0[IX(i1, j0, k0)] + u1 * d_d0[IX(i1, j0, k1)]) +
                  t1 * (u0 * d_d0[IX(i1, j1, k0)] + u1 * d_d0[IX(i1, j1, k1)]));
    }
}


void advect_call_kernel(int M, int N, int O, int b, float *d_d, float *d_d0, float *d_u, float *d_v, float *d_w, float dt) {
    dim3 threads_per_block(8, 8, 8);
    dim3 blocks_grid((M + threads_per_block.x - 1) / threads_per_block.x,
                    (N + threads_per_block.y - 1) / threads_per_block.y,
                    (O + threads_per_block.z - 1) / threads_per_block.z);

    advect_kernel<<<blocks_grid, threads_per_block>>>(M, N, O, dt, d_d, d_d0, d_u, d_v, d_w);
    cudaDeviceSynchronize();

    set_bnd_call_kernels(M, N, O, b, d_d);
}


__global__ void project_1_kernel(int M, int N, int O, float *d_u, float *d_v, float *d_w, float *d_p, float *d_div) {

  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O) {
    int idx = IX(i, j, k);
    d_div[idx] = -0.5f *
               ((d_u[IX(i + 1, j, k)] - d_u[IX(i - 1, j, k)]) +
                (d_v[IX(i, j + 1, k)] - d_v[IX(i, j - 1, k)]) +
                (d_w[IX(i, j, k + 1)] - d_w[IX(i, j, k - 1)])) /
               MAX(M, MAX(N, O));

    d_p[idx] = 0.0f;
  }
}


__global__ void project_2_kernel(int M, int N, int O, float *d_u, float *d_v, float *d_w, float *d_p) {

  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O) {
    int idx = IX(i, j, k);
    d_u[idx] += -0.5f * (d_p[IX(i + 1, j, k)] - d_p[IX(i - 1, j, k)]);
    d_v[idx] += -0.5f * (d_p[IX(i, j + 1, k)] - d_p[IX(i, j - 1, k)]);
    d_w[idx] += -0.5f * (d_p[IX(i, j, k + 1)] - d_p[IX(i, j, k - 1)]);
  }

}


void project_call_kernels(int M, int N, int O, float *d_u, float *d_v, float *d_w, float *d_p, float *d_div) {
  dim3 threads_per_block(8,8,8);
  dim3 blocks_grid((M + threads_per_block.x - 1) / threads_per_block.x,
                   (N + threads_per_block.y - 1) / threads_per_block.y,
                   (O + threads_per_block.z - 1) / threads_per_block.z);
 
  project_1_kernel<<<blocks_grid, threads_per_block>>>(M, N, O, d_u, d_v, d_w, d_p, d_div);
  cudaDeviceSynchronize();

  set_bnd_call_kernels(M, N, O, 0, d_div);
  set_bnd_call_kernels(M, N, O, 0, d_p);
  
  lin_solve(M, N, O, 0, d_p, d_div, 1, 6);

  project_2_kernel<<<blocks_grid, threads_per_block>>>(M, N, O, d_u, d_v, d_w, d_p);
  cudaDeviceSynchronize();

  set_bnd_call_kernels(M, N, O, 1, d_u);
  set_bnd_call_kernels(M, N, O, 2, d_v);
  set_bnd_call_kernels(M, N, O, 3, d_w);
}


// Step function for density
void dens_step(int M, int N, int O, float *d_x, float *d_x0, float *d_u, float *d_v, float *d_w, float diff, float dt) {

  add_source_call_kernel(M, N, O, d_x, d_x0, dt);
  SWAP(d_x0, d_x);
  diffuse(M, N, O, 0, d_x, d_x0, diff, dt);
  SWAP(d_x0, d_x);
  advect_call_kernel(M, N, O, 0, d_x, d_x0, d_u, d_v, d_w, dt);

}


// Step function for velocity
void vel_step(int M, int N, int O, float *d_u, float *d_v, float *d_w, float *d_u0, float *d_v0, float *d_w0, float visc, float dt) {


  add_source_call_kernel(M, N, O, d_u, d_u0, dt);
  add_source_call_kernel(M, N, O, d_v, d_v0, dt);
  add_source_call_kernel(M, N, O, d_w, d_w0, dt);
  SWAP(d_u0, d_u);
  diffuse(M, N, O, 1, d_u, d_u0, visc, dt);
  SWAP(d_v0, d_v);
  diffuse(M, N, O, 2, d_v, d_v0, visc, dt);
  SWAP(d_w0, d_w);
  diffuse(M, N, O, 3, d_w, d_w0, visc, dt);
  project_call_kernels(M, N, O, d_u, d_v, d_w, d_u0, d_v0);
  SWAP(d_u0, d_u);
  SWAP(d_v0, d_v);
  SWAP(d_w0, d_w);
  advect_call_kernel(M, N, O, 1, d_u, d_u0, d_u0, d_v0, d_w0, dt);
  advect_call_kernel(M, N, O, 2, d_v, d_v0, d_u0, d_v0, d_w0, dt);
  advect_call_kernel(M, N, O, 3, d_w, d_w0, d_u0, d_v0, d_w0, dt);
  project_call_kernels(M, N, O, d_u, d_v, d_w, d_u0, d_v0);

}
