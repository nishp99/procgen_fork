#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=1
#SBATCH --error=/nfs/nhome/live/npatel/procgen_experiments/procgen_fork/scripts/utils/results/outputs/%j_0_log.err
#SBATCH --job-name=submitit
#SBATCH --mem=1GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/nfs/nhome/live/npatel/procgen_experiments/procgen_fork/scripts/utils/results/outputs/%j_0_log.out
#SBATCH --signal=USR1@90
#SBATCH --time=60
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --output /nfs/nhome/live/npatel/procgen_experiments/procgen_fork/scripts/utils/results/outputs/%j_%t_log.out --error /nfs/nhome/live/npatel/procgen_experiments/procgen_fork/scripts/utils/results/outputs/%j_%t_log.err --unbuffered /nfs/nhome/live/npatel/.conda/envs/procgen_fork_local/bin/python -u -m submitit.core._submit /nfs/nhome/live/npatel/procgen_experiments/procgen_fork/scripts/utils/results/outputs