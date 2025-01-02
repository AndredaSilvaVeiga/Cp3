#!/bin/sh
#
#BATCH --exclusive        # exclusive node for the job
#SBATCH --time=05:00	  # allocation for 2 minutes
#SBATCH --partition=day
#SBATCH --ntasks=1
#SBATCH --constraint=k20

module load gcc/7.2.0
module load cuda/11.3.1
make
time ./fluid_sim
