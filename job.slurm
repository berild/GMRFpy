#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=martinob
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --array=1-10
#SBATCH --mem=1000000000
#SBATCH --job-name="SS_GMRFpy"
#SBATCH --output=SS_GMRFpy.out
#SBATCH --mail-user=martinob@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module use /cluster/work/perhov/eb/modules/all
module load SciPy-bundle/2021.05-foss-2021a
module load R/4.1.0-foss-2021a
module load CMake/3.20.1-GCCcore-10.3.0
module load SuiteSparse/5.10.1-foss-2021a-METIS-5.1.0
module load NLopt/2.7.0-GCCcore-10.3.0
source ../pyenv/bin/activate
srun python3 srunfitNA.py $SLURM_ARRAY_TASK_ID

uname -a