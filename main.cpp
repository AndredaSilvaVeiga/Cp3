#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>
#include <cstdlib>


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
static float *dens, *dens_prev;

// Function to allocate simulation data
int allocate_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  u = static_cast<float*>(aligned_alloc(32, size * sizeof(float)));
  v = static_cast<float*>(aligned_alloc(32, size * sizeof(float)));
  w = static_cast<float*>(aligned_alloc(32, size * sizeof(float)));
  u_prev = static_cast<float*>(aligned_alloc(32, size * sizeof(float)));
  v_prev = static_cast<float*>(aligned_alloc(32, size * sizeof(float)));
  w_prev = static_cast<float*>(aligned_alloc(32, size * sizeof(float)));
  dens = static_cast<float*>(aligned_alloc(32, size * sizeof(float)));
  dens_prev = static_cast<float*>(aligned_alloc(32, size * sizeof(float)));

  if (!u || !v || !w || !u_prev || !v_prev || !w_prev || !dens || !dens_prev) {
    std::cerr << "Cannot allocate memory" << std::endl;
    return 0;
  }
  return 1;
}

// Function to clear the data (set all to zero)
void clear_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    u[i] = v[i] = w[i] = u_prev[i] = v_prev[i] = w_prev[i] = dens[i] =
        dens_prev[i] = 0.0f;
  }
}

// Free allocated memory
void free_data() {
  free(u);
  free(v);
  free(w);
  free(u_prev);
  free(v_prev);
  free(w_prev);
  free(dens);
  free(dens_prev);
}

// Apply events (source or force) for the current timestep
void apply_events(const std::vector<Event> &events) {
  for (const auto &event : events) {
    if (event.type == ADD_SOURCE) {
      // Apply density source at the center of the grid
      int i = M / 2, j = N / 2, k = O / 2;
      dens[IX(i, j, k)] = event.density;
    } else if (event.type == APPLY_FORCE) {
      // Apply forces based on the event's vector (fx, fy, fz)
      int i = M / 2, j = N / 2, k = O / 2;
      u[IX(i, j, k)] = event.force.x;
      v[IX(i, j, k)] = event.force.y;
      w[IX(i, j, k)] = event.force.z;
    }
  }
}

// Function to sum the total density
float sum_density() {
  float total_density = 0.0f;
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    total_density += dens[i];
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

    // Perform the simulation steps
    vel_step(M, N, O, u, v, w, u_prev, v_prev, w_prev, visc, dt);
    dens_step(M, N, O, dens, dens_prev, u, v, w, diff, dt);
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

  // Print total density at the end of simulation
  float total_density = sum_density();
  std::cout << "Total density after " << timesteps
            << " timesteps: " << total_density << std::endl;

  // Free memory
  free_data();

  return 0;
}
