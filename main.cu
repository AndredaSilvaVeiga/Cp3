#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda.h>
#include <omp.h>

// 168
#define SIZE 168

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

// Fluid simulation arrays
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;
static float *dens, *dens_prev, *h_dens;

// Function to allocate simulation data
int allocate_data() {
  int size = (M + 2) * (N + 2) * (O + 2);

  cudaMalloc((void**)&u, size * sizeof(float));
  cudaMalloc((void**)&v, size * sizeof(float));
  cudaMalloc((void**)&w, size * sizeof(float));
  cudaMalloc((void**)&u_prev, size * sizeof(float));
  cudaMalloc((void**)&v_prev, size * sizeof(float));
  cudaMalloc((void**)&w_prev, size * sizeof(float));
  cudaMalloc((void**)&dens, size * sizeof(float));
  cudaMalloc((void**)&dens_prev, size * sizeof(float));

  h_dens = (float *)malloc(size * sizeof(float));

  //dens = static_cast<float*>(aligned_alloc(32, size * sizeof(float)));
  //dens_prev = static_cast<float*>(aligned_alloc(32, size * sizeof(float)));

  if (!u || !v || !w || !u_prev || !v_prev || !w_prev || !dens || !dens_prev) {
    std::cerr << "Cannot allocate memory" << std::endl;
    return 0;
  }
  return 1;
}

// Function to clear the data (set all to zero)
void clear_data() {
  int size = (M + 2) * (N + 2) * (O + 2);

  cudaMemset(u, 0, size * sizeof(float));
  cudaMemset(v, 0, size * sizeof(float));
  cudaMemset(w, 0, size * sizeof(float));
  cudaMemset(u_prev, 0, size * sizeof(float));
  cudaMemset(v_prev, 0, size * sizeof(float));
  cudaMemset(w_prev, 0, size * sizeof(float));
  cudaMemset(dens, 0, size * sizeof(float));
  cudaMemset(dens_prev, 0, size * sizeof(float));

  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
     h_dens[i] = 0.0f;
  }
}

// Free allocated memory
void free_data() {

  cudaFree(u);
  cudaFree(v);
  cudaFree(w);
  cudaFree(u_prev);
  cudaFree(v_prev);
  cudaFree(w_prev);
  cudaFree(dens);
  cudaFree(dens_prev);

  free(h_dens);
}


__global__ void apply_events_kernel(int M, int N, int O, float *dens, float *u, float *v, float *w, Event *events, int num_events) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_events) {
        Event event = events[idx];
        int i = SIZE / 2, j = SIZE / 2, k = SIZE / 2;

        if (event.type == ADD_SOURCE) {
            dens[IX(i, j, k)] = event.density;
        } else if (event.type == APPLY_FORCE) {
            u[IX(i, j, k)] = event.force.x;
            v[IX(i, j, k)] = event.force.y;
            w[IX(i, j, k)] = event.force.z;
        }
    }
}

void apply_events(const std::vector<Event> &events) {
    Event *d_events;
    cudaMalloc((void **)&d_events, events.size() * sizeof(Event));
    cudaMemcpy(d_events, events.data(), events.size() * sizeof(Event), cudaMemcpyHostToDevice);

    int blocks_size = 128;
    apply_events_kernel<<<1, blocks_size>>>(M, N, O, dens, u, v, w, d_events, events.size());
    cudaFree(d_events);
}


// Function to sum the total density
float sum_density(float *dens_array) {
  float total_density = 0.0f;
  int size = (M + 2) * (N + 2) * (O + 2);

  //#pragma omp parallel for reduction(+:total_density) schedule(static)
  for (int i = 0; i < size; i++) {
    total_density += dens_array[i];
  }
  return total_density;
}

// Simulation loop
void simulate(EventManager &eventManager, int timesteps) {
  for (int t = 0; t < timesteps; t++) {
    // Get the events for the current timestep
    std::vector<Event> events = eventManager.get_events_at_timestamp(t);

    // Apply events to the simulation
    apply_events(events);

    //Perform the simulation steps
    vel_step(M, N, O, u, v, w, u_prev, v_prev, w_prev, visc, dt);
    dens_step(M, N, O, dens, dens_prev, u, v, w, diff, dt);
    cudaDeviceSynchronize();
  }
}

int main() {
  // Initialize EventManager
  EventManager eventManager;
  eventManager.read_events("events.txt");

  // Get the total number of timesteps from the event file
  int timesteps = eventManager.get_total_timesteps();

  // Allocate and clear data
  if (!allocate_data())
    return -1;
  clear_data();

  // Run simulation with events
  simulate(eventManager, timesteps);


  int size = (M + 2) * (N + 2) * (O + 2);
  cudaMemcpy(h_dens, dens, size * sizeof(float), cudaMemcpyDeviceToHost);


  // Print total density at the end of simulation
  float total_density = sum_density(h_dens);
  std::cout << "Total density after " << timesteps
            << " timesteps: " << total_density << std::endl;

  // Free memory
  free_data();

  return 0;
}
