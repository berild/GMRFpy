#!/bin/bash
# Ask the user for their name
echo Which model to run?
read varname1
echo From which number?
read varname2
echo Running model $varname1 from number $varname2
source ../env/bin/activate

python3 srunfit.py $varname1 $varname2