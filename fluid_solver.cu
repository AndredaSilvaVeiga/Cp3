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
  
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  add_source_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, x, s,dt);
}

__global__ void add_source_kernel(int M, int N, int O, float *x, float *s,float dt) {


  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int size = (M + 2) * (N + 2) * (O + 2);

  if (idx < size){
    x[idx] += dt * s[idx];
  }
}

// Não funciona
// Set boundary conditions - Kernel
__global__ void set_bnd_kernel(int M, int N, int O, int b, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i > 0 && i <= M && j > 0 && j <= N) {
    x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
    x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
  }

  if (j > 0 && j <= N && k > 0 && k <= O) { 
    x[IX(0, j, k)] = b == 1 ? -x[IX(1, j, k)] : x[IX(1, j, k)];
    x[IX(M + 1, j, k)] = b == 1 ? -x[IX(M, j, k)] : x[IX(M, j, k)];
  }

  if (i > 0 && i <= M && k > 0 && k <= O) {
    x[IX(i, 0, k)] = b == 2 ? -x[IX(i, 1, k)] : x[IX(i, 1, k)];
    x[IX(i, N + 1, k)] = b == 2 ? -x[IX(i, N, k)] : x[IX(i, N, k)];
  }

  if (i == 0 && j == 0) {
    x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
  }
  if (i == M + 1 && j == 0) {
    x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
  }
  if (i == 0 && j == N + 1) {
    x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
  }
  if (i == M + 1 && j == N + 1) {
    x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
  }
}

void setbnd_cuda(int M, int N, int O, int b, float *x){
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks ((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

  set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M,N,O,b,x);
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


__global__ void lin_solver_kernel(float *x, float *x0,int N, int M, int O, float a, float c, float *change_array, int sign) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if((i + j + k) % 2 != sign) return;

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
  //float *d_x, *d_x0; 
  float *d_change_array, *d_change_reduction, *h_change_reduction;
  int size = (M + 2) * (N + 2) * (O + 2);

  //Threads and blocks def
  dim3 threads_per_block(8, 8, 8);
  dim3 num_blocks((M + threads_per_block.x - 1) / threads_per_block.x,
                   (N + threads_per_block.y - 1) / threads_per_block.y,
                   (O + threads_per_block.z - 1) / threads_per_block.z);

  int threads_per_block_reduction = 1024;
  int num_blocks_reduction  = (M * N * O + threads_per_block_reduction - 1) / threads_per_block_reduction; 

  // Alloc memory and copy to device
  //cudaMalloc((void **)&d_x, size * sizeof(float));
  //cudaMalloc((void **)&d_x0, size * sizeof(float));  //estes

  cudaMalloc((void **)&d_change_array, size * sizeof(float)); 
  cudaMalloc((void **)&d_change_reduction, num_blocks_reduction * sizeof(float)); 
  h_change_reduction = (float *)malloc(num_blocks_reduction * sizeof(float));

  // Transfer data to device
  //cudaMemcpy(d_x0, x0, size * sizeof(float), cudaMemcpyHostToDevice);   //este

  int l = 0;
  float max_change;
  float tol = 1e-7;
  do {      
    max_change = 0.0f;
    //cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice); 

    //Phase 1
    lin_solver_kernel<<<num_blocks, threads_per_block>>>(x, x0, N, M, O, a, c, d_change_array, 0);
    cudaError_t err_1 = cudaGetLastError();
    if (err_1 != cudaSuccess) {
      printf("Kernel launch failed 1. Error: %s\n", cudaGetErrorString(err_1));
    }

    //Phase 2
    lin_solver_kernel<<<num_blocks, threads_per_block>>>(x, x0, N, M, O, a, c, d_change_array, 1);
    cudaError_t err_2 = cudaGetLastError();
    if (err_2 != cudaSuccess) {
      printf("Kernel launch failed 2. Error: %s\n", cudaGetErrorString(err_2));
    }    
      
    // Reduction GPU
    reduce_max<<<num_blocks_reduction, threads_per_block_reduction, threads_per_block_reduction * sizeof(float)>>>(d_change_array, d_change_reduction);  
 
    // Reduction CPU
    cudaMemcpy(h_change_reduction, d_change_reduction, num_blocks_reduction * sizeof(float), cudaMemcpyDeviceToHost);
    #pragma omp parallel for reduction(max:max_change)
    for(int i = 0; i <= num_blocks_reduction; i++) {
    	    max_change = fmaxf(max_change, h_change_reduction[i]);
    }

    setbnd_cuda(M, N, O, b, x);
    //cudaMemcpy(x, d_x, size * sizeof(float), cudaMemcpyDeviceToHost); 
    //set_bnd(M, N, O, b, x);
   
    } while (++l < LINEARSOLVERTIMES && max_change > tol);
  
  //cudaMemcpy(x, d_x, size * sizeof(float), cudaMemcpyDeviceToHost);  //este
  // Free memory
  //cudaFree(d_x);
  //cudaFree(d_x0);
  cudaFree(d_change_array);
  cudaFree(d_change_reduction);
  free(h_change_reduction);
}


// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}


// Advection step (uses velocity field to move quantities)
// Spatial Locality Done
__global__ void advect_Kernel (int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

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

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt){

  dim3 threadsPerBlock(8,8,8);
  dim3 gridDim((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
               (O + threadsPerBlock.z - 1) / threadsPerBlock.z);
  advect_Kernel<<<gridDim, threadsPerBlock>>>(M, N, O, b, d, d0, u, v, w, dt);

  set_bnd(M, N, O, b, d);
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
// Spatial Locality Done
void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {


  dim3 threadsPerBlock(8,8,8);
  dim3 gridDim((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
               (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

 
project1_kernel<<<gridDim, threadsPerBlock>>>(M, N, O, u, v, w, p, div);

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

project2_kernel<<<gridDim, threadsPerBlock>>>(M, N, O, u, v, w, p);

  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}


__global__ void project1_kernel(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {

  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O) {
    int idx = IX(i, j, k);
    div[idx] = -0.5f *
               ((u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)]) +
                (v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)]) +
                (w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)])) /
               MAX(M, MAX(N, O));

    p[idx] = 0.0f;
  }
}


__global__ void project2_kernel(int M, int N, int O, float *u, float *v, float *w, float *p,) {

  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O) {
    int idx = IX(i, j, k);
    u[idx] += -0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
    v[idx] += -0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
    w[idx] += -0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
  }

}

// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {

  int size = (M + 2) * (N + 2) * (O + 2);
  size *= sizeof(float);

  cudaMalloc(&u_k, size);
  cudaMalloc(&v_k, size);
  cudaMalloc(&w_k, size);
  cudaMalloc(&x_k, size);
  cudaMalloc(&x0_k, size);

  cudaMemcpy(u_k, u, size , cudaMemcpyHostToDevice);
  cudaMemcpy(v_k, v, size , cudaMemcpyHostToDevice);
  cudaMemcpy(w_k, w, size , cudaMemcpyHostToDevice);
  cudaMemcpy(x_k, x, size , cudaMemcpyHostToDevice);
  cudaMemcpy(x0_k, x0, size , cudaMemcpyHostToDevice);

  add_source(M, N, O, x_k, x0_k, dt);
  SWAP(x0_k, x_k);
  diffuse(M, N, O, 0, x_k, x0_k, diff, dt);
  SWAP(x0_k, x_k);
  advect(M, N, O, 0, x_k, x0_k, u_k, v_k, w_k, dt);

  cudaMemcpy(x, x_k, size , cudaMemcpyDeviceToHost);
  cudaMemcpy(x0, x0_k, size , cudaMemcpyDeviceToHost);
  cudaMemcpy(u, u_k, size , cudaMemcpyDeviceToHost);
  cudaMemcpy(v, v_k, size , cudaMemcpyDeviceToHost);
  cudaMemcpy(w, w_k, size , cudaMemcpyDeviceToHost);
}


// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt) {


  int size = (M + 2) * (N + 2) * (O + 2);
  size *= sizeof(float);

  cudaMalloc(&u_k, size);
  cudaMalloc(&u0_k, size);
  cudaMalloc(&v_k, size);
  cudaMalloc(&v0_k, size);
  cudaMalloc(&w_k, size);
  cudaMalloc(&w0_k, size);

  cudaMemcpy(u_k, u, size , cudaMemcpyHostToDevice);
  cudaMemcpy(u0_k, u0, size , cudaMemcpyHostToDevice);
  cudaMemcpy(v_k, v, size , cudaMemcpyHostToDevice);
  cudaMemcpy(v0_k, v0, size , cudaMemcpyHostToDevice);
  cudaMemcpy(w_k, w, size , cudaMemcpyHostToDevice);
  cudaMemcpy(w0_k, w0, size , cudaMemcpyHostToDevice);

  add_source(M, N, O, u_k, u0_k, dt);
  add_source(M, N, O, v_k, v0_k, dt);
  add_source(M, N, O, w_k, w0_k, dt);
  SWAP(u0_k, u_k);
  diffuse(M, N, O, 1, u_k, u0_k, visc, dt);
  SWAP(v0_k v_k);
  diffuse(M, N, O, 2, v_k, v0_k, visc, dt);
  SWAP(w0, w);
  diffuse(M, N, O, 3, w_k, w0_k, visc, dt);
  project(M, N, O, u_k, v_k, w_k, u0_k, v0_k);
  SWAP(u0, u_k);
  SWAP(v0, v_k);
  SWAP(w0, w_k);
  advect(M, N, O, 1, u_k, u0_k, u0_k, v0_k, w0_k, dt);
  advect(M, N, O, 2, v_k, v0_k, u0_k, v0_k, w0_k, dt);
  advect(M, N, O, 3, w_k, w0_k, u0_k, v0_k, w0_k, dt);
  project(M, N, O, u_k, v_k, w_k, u0_k, v0_k);


  cudaMemcpy(u, u_k, size , cudaMemcpyDeviceToHost);
  cudaMemcpy(v, v_k, size , cudaMemcpyDeviceToHost);
  cudaMemcpy(w, w_k, size , cudaMemcpyDeviceToHost);
  cudaMemcpy(u, u0_k, size , cudaMemcpyDeviceToHost);
  cudaMemcpy(v, v0_k, size , cudaMemcpyDeviceToHost);
  cudaMemcpy(w, w0_k, size , cudaMemcpyDeviceToHost);
}
