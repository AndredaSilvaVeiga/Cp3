#!/bin/sh
#
#BATCH --exclusive 	  # exclusive node for the job
#SBATCH --time=02:00      # allocation for 2 minutes
#SBATCH --partition=day
#SBATCH --cpus-per-task=20
#SBATCH --constraint=k20

export OMP_NUM_THREADS=20
nvprof ./fluid_sim
