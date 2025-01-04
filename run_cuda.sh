#!/bin/sh
#
#BATCH --exclusive 	  # exclusive node for the job
#SBATCH --time=5:00      # allocation for 2 minutes
#SBATCH --partition=day
#SBATCH --cpus-per-task=20
#SBATCH --constraint=k20
#SBATCH --ntasks=1

module load gcc/7.2.0
module load cuda/11.3.1

make

export OMP_NUM_THREADS=20
nvprof ./fluid_sim
