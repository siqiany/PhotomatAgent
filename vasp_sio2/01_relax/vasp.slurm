#!/bin/bash
#SBATCH -J sio2_relax
#SBATCH -p kshcnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -t 10
#SBATCH -o %j.out
#SBATCH -e %j.err
set -euo pipefail
ulimit -s unlimited
module purge
set +e
source /public/home/scniv4a4go/apprepo/vasp/5.4.4-ioptcell_intelmpi2017_hdf5_libxc/scripts/env.sh
set -e
command -v vasp_std >/dev/null
srun --mpi=pmi2 vasp_std
