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


// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
}


// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float *x) {
  int i, j;

  // Set boundary on faces
  #pragma omp parallel 
  {
  #pragma omp for collapse(2) 
  for (j = 1; j <= N; j++) {
  for (i = 1; i <= M; i++) {

      x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
      x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
    }
  }

  #pragma omp for collapse(2)
  for (j = 1; j <= O; j++) { 
    for (i = 1; i <= N; i++) {
      x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
      x[IX(M + 1, i, j)] = b == 1 ? -x[IX(M, i, j)] : x[IX(M, i, j)];
    }
  }

  #pragma omp for collapse(2)
  for (j = 1; j <= O; j++) {
    for (i = 1; i <= M; i++) {
      x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
      x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
    }
  }
  }
  // Set corners
  x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
  x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
  x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
  x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                    x[IX(M + 1, N + 1, 1)]);
}


__global__ void reduce_max(float *input, float *output, int n) {
  extern __shared__ float shared_data[];

  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

 // Tirar esta linha se usar o bloco de codigo seguinte
  shared_data[tid] = input[idx];

 //
 // -> Problemas de memoria com esta parte 
 //    (se tiver incluido o resultado final dá 0,
 //     se não tiver dá reultado perto do desejado,
 //     mas com memory leaks) 
 //
 // if(idx < n) {
 //	shared_data[tid] = input[idx];
 // } else {
 //	shared_data[tid] = -INFINITY;
 // }
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


__global__ void lin_solver_kernel(float *x, float *x0,int N, int M, int O, float a, float c, float *change_array, int sign) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  //if((i + j + k) % 2 != sign) return;

  if (i <= M && j <= N && k <= O) {
    int idx = IX(i, j, k);
    float  old_x = x[idx];
    x[idx] = (x0[idx] + a *(x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                            x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                            x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
	  float change = fabsf(x[idx] - old_x);
	  change_array[idx] = change;
  }
}


// Linear solve for implicit methods (diffusion)
// red-black solver with convergence check
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
  float *d_x, *d_x0, *d_max, *d_change_array;
  int size = (M + 2) * (N + 2) * (O + 2);

  // Alloc memory and copy to device
  cudaMalloc((void **)&d_x, size * sizeof(float));
  cudaMalloc((void **)&d_x0, size * sizeof(float));
  cudaMalloc((void **)&d_change_array, size * sizeof(float));
  cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice); 
  cudaMemcpy(d_x0, x0, size * sizeof(float), cudaMemcpyHostToDevice);

  // Threads and blocks def
  dim3 threads_per_block(8, 8, 8);
  dim3 num_blocks((M + threads_per_block.x - 1) / threads_per_block.x,
                   (N + threads_per_block.y - 1) / threads_per_block.y,
                   (O + threads_per_block.z - 1) / threads_per_block.z);

  int threads_per_block_reduction = 256;
  int num_blocks_reduction = (size + threads_per_block_reduction - 1) / threads_per_block_reduction; 
  cudaMalloc(&d_max, num_blocks_reduction * sizeof(float));
  float *h_max = (float *)malloc(num_blocks_reduction * sizeof(float));

  int l = 0;
  float max_change;
  float tol = 1e-7;
  do {      
    max_change = 0.0f;

    //Phase 1
    lin_solver_kernel<<<num_blocks, threads_per_block>>>(d_x, d_x0, N, M, O, a, c, d_change_array, 0);
    cudaError_t err_1 = cudaGetLastError();
    if (err_1 != cudaSuccess) {
      printf("Kernel launch failed 1. Error: %s\n", cudaGetErrorString(err_1));
    }

    //Phase 2
    lin_solver_kernel<<<num_blocks, threads_per_block>>>(d_x, d_x0, N, M, O, a, c, d_change_array, 1);
    cudaError_t err_2 = cudaGetLastError();
    if (err_2 != cudaSuccess) {
      printf("Kernel launch failed 2. Error: %s\n", cudaGetErrorString(err_2));
    }
    
      
    // Reduction GPU
    reduce_max<<<num_blocks_reduction, threads_per_block_reduction, threads_per_block_reduction * sizeof(float)>>>(d_change_array, d_max, size);  
   
    // Reduction CPU
    cudaMemcpy(h_max, d_max, num_blocks_reduction * sizeof(float), cudaMemcpyDeviceToHost);
    

    //
    // Redução final feita no CPU -> melhoria: fazer a redução total no GPU
    //    

    #pragma omp parallel for reduction(max:max_change)
    for(int i = 0; i < num_blocks_reduction; i++) {
	    max_change = fmaxf(max_change, h_max[i]);
    }

    set_bnd(M, N, O, b, x);
    } while (++l < LINEARSOLVERTIMES && max_change > tol);


  cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

  // Free memory
  cudaFree(d_x);
  cudaFree(d_x0);
  cudaFree(d_change_array);
  cudaFree(d_max);
  free(h_max);
}


// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}


// Advection step (uses velocity field to move quantities)
// Spatial Locality Done
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

  #pragma omp parallel for collapse(3)
  for (int k = 1; k <= O; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {

        float x = i - dtX * u[IX(i, j, k)];
        float y = j - dtY * v[IX(i, j, k)];
        float z = k - dtZ * w[IX(i, j, k)];

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

        d[IX(i, j, k)] =
            s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                  t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
            s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                  t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
      }
    }
  }
  set_bnd(M, N, O, b, d);
}


// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
// Spatial Locality Done
void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {

  #pragma omp parallel for collapse(3)
  for (int k = 1; k <= O; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {

        div[IX(i, j, k)] =
            -0.5f *
            (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
             v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) /
            MAX(M, MAX(N, O));
        p[IX(i, j, k)] = 0;
      }
    }
  }

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

  #pragma omp parallel for collapse(3)
  for (int k = 1; k <= O; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= M; i++) {
       int idx = IX(i, j, k);

        u[idx] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[idx] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[idx] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
      }
    }
  }
  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}


// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {
  add_source(M, N, O, x, x0, dt);
  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}


// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt) {
  add_source(M, N, O, u, u0, dt);
  add_source(M, N, O, v, v0, dt);
  add_source(M, N, O, w, w0, dt);
  SWAP(u0, u);
  diffuse(M, N, O, 1, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse(M, N, O, 2, v, v0, visc, dt);
  SWAP(w0, w);
  diffuse(M, N, O, 3, w, w0, visc, dt);
  project(M, N, O, u, v, w, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  SWAP(w0, w);
  advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project(M, N, O, u, v, w, u0, v0);
}
