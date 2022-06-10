#!/bin/bash
# Ask the user for their name
echo Which model to run?
read varname1
echo From which number?
read varname2
echo Running model $varname1 from number $varname2
module use /cluster/work/perhov/eb/modules/all
module load SciPy-bundle/2021.05-foss-2021a
module load R/4.1.0-foss-2021a
module load CMake/3.20.1-GCCcore-10.3.0
module load SuiteSparse/5.10.1-foss-2021a-METIS-5.1.0
module load NLopt/2.7.0-GCCcore-10.3.0
source ../pyenb/bin/activate

python3 srunfit.py $varname1 $varname2