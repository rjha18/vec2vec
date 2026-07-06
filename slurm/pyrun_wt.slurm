#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="gpu-high"
#SBATCH --job-name=pyrun
#SBATCH --requeue
#SBATCH --error=out/error/%x.%j.err
#SBATCH --output=out/output/%x.%j.out

# Worktree runner: sbatch slurm/pyrun_wt.slurm scripts/foo.py --arg ...
# (same as slurm/pyrun.slurm in the main checkout but cd's to this worktree)
set -eo pipefail
cd /home/rdj58/code/supervised_disc/.claude/worktrees/fable-identifiability-experiments
source activate inversion
export TQDM_DISABLE=1
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "Starting job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo "Command: python $*"
python "$@"
