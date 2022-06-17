#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-im
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --array=1-10
#SBATCH -c 28
#SBATCH --mem=100000000
#SBATCH --job-name="SS_GMRFpy"
#SBATCH --output=SS_GMRFpy.out
#SBATCH --mail-user=martinob@ntnu.no 
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

module purge
module use /cluster/work/perhov/eb/modules/all
module load SciPy-bundle/2021.05-foss-2021a
module load R/4.1.0-foss-2021a
module load CMake/3.20.1-GCCcore-10.3.0
module load SuiteSparse/5.10.1-foss-2021a-METIS-5.1.0
module load NLopt/2.7.0-GCCcore-10.3.0
source /cluster/home/martinob/
srun python3 srunfitNA.py $SLURM_ARRAY_TASK_ID