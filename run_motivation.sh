#!/bin/bash
# Get the CMNIST Results

declare -i num_trainings=50;

cd empirical_motivation;
OLDIFS=$IFS;
IFS=',';

for i in 00112,'0 0 1 1 2 3' 30112,'3 0 1 1 2 3' 32000,'3 2 0 0 0 3' 32102,'3 2 1 0 2 3' 32112,'3 2 1 1 2 3';
do
       	set -- $i;
	cd $1;
	python ../dense.py $2 $num_trainings;
	cd ..;
done;
python replot.py
cd ..
IFS=$OLDIFS
